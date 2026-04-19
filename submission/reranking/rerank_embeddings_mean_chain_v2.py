"""Combined reranking pipeline: mean-chain followed by graph rerank v2."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

DEFAULT_MAIN_METRIC_KS = (5, 10, 20, 45)
DEFAULT_METRIC_KS = DEFAULT_MAIN_METRIC_KS


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    if "embeddings" not in payload:
        raise KeyError(f"{path} is missing required key 'embeddings'")
    if "audio_ids" not in payload:
        raise KeyError(f"{path} is missing required key 'audio_ids'")
    return payload


def _speaker_ids_from_audio_ids(audio_ids: list[str]) -> list[str]:
    speakers: list[str] = []
    for audio_id in audio_ids:
        token = str(audio_id).strip()
        if not token:
            raise ValueError("Encountered empty audio_id")
        speakers.append(token.split("/", 1)[0])
    return speakers


def _build_speaker_labels(speaker_ids: list[str]) -> np.ndarray:
    unique = sorted(set(speaker_ids))
    mapping = {speaker_id: idx for idx, speaker_id in enumerate(unique)}
    return np.asarray([mapping[speaker_id] for speaker_id in speaker_ids], dtype=np.int32)


def _speaker_counts(speaker_labels: np.ndarray) -> np.ndarray:
    return np.bincount(speaker_labels)


def _precision_at_ks_from_neighbors(
    nbrs: np.ndarray,
    speaker_labels: np.ndarray,
    ks: list[int],
) -> dict[int, float]:
    if not ks:
        return {}

    counts = _speaker_counts(speaker_labels)
    max_k = int(max(ks))
    if nbrs.shape[1] < max_k:
        raise ValueError(f"Neighbor graph has K={nbrs.shape[1]} but metrics need up to {max_k}")

    rel = (speaker_labels[nbrs[:, :max_k]] == speaker_labels[:, None]).astype(np.float32, copy=False)
    cumsum_hits = np.cumsum(rel, axis=1)
    per_query_counts = counts[speaker_labels]

    metrics: dict[int, float] = {}
    for k in ks:
        valid_queries = per_query_counts >= 2
        num_valid = int(valid_queries.sum())
        if num_valid == 0:
            raise ValueError(f"No query has enough same-speaker utterances for precision@{k}")
        metrics[k] = float(np.mean(cumsum_hits[valid_queries, k - 1] / float(k)))
    return metrics


def _empty_metrics(ks: list[int]) -> dict[int, float | None]:
    return {int(k): None for k in ks}


def _serialize_metrics(metrics: dict[int, float | None]) -> dict[str, float | None]:
    return {
        f"precision@{k}": (None if v is None else float(v))
        for k, v in metrics.items()
    }


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return emb / norms


def _compute_candidate_pool(
    embeddings: np.ndarray,
    *,
    pool_k: int,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(embeddings.shape[0])
    emb = torch.as_tensor(embeddings, dtype=torch.float32)
    emb_dev = emb.to(device)

    nbrs = np.empty((n, pool_k), dtype=np.int32)
    sims_out = np.empty((n, pool_k), dtype=np.float32)
    t0 = time.time()
    progress = tqdm(
        range(0, n, chunk_size),
        total=(n + chunk_size - 1) // chunk_size,
        desc="candidate pool",
        unit="chunk",
    )
    for start in progress:
        end = min(start + chunk_size, n)
        sims = emb_dev[start:end] @ emb_dev.T
        row_idx = torch.arange(end - start, device=device)
        sims[row_idx, torch.arange(start, end, device=device)] = float("-inf")
        top = torch.topk(sims, k=pool_k, dim=1)
        nbrs[start:end] = top.indices.cpu().numpy().astype(np.int32, copy=False)
        sims_out[start:end] = top.values.cpu().numpy().astype(np.float32, copy=False)
        dt = max(time.time() - t0, 1e-6)
        progress.set_postfix(items=end, rate=f"{end / dt:.0f}/s")
    return nbrs, sims_out


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom <= 0.0:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / denom).astype(np.float32, copy=False)


def _blend_scores(
    cand_emb: np.ndarray,
    *,
    base_scores: np.ndarray,
    query_emb: np.ndarray,
    probe_emb: np.ndarray,
    weight_base: float,
    weight_query: float,
    weight_probe: float,
) -> np.ndarray:
    scores = np.zeros(cand_emb.shape[0], dtype=np.float32)
    if weight_base:
        scores += float(weight_base) * base_scores
    if weight_query:
        scores += float(weight_query) * (cand_emb @ query_emb)
    if weight_probe:
        scores += float(weight_probe) * (cand_emb @ probe_emb)
    return scores


def _stage1_rerank(
    embeddings: np.ndarray,
    base_nbrs: np.ndarray,
    base_scores: np.ndarray,
    *,
    out_k: int,
    mean_steps: int,
) -> np.ndarray:
    n, pool_k = base_nbrs.shape
    if out_k > pool_k:
        raise ValueError(f"Need pool_k >= out_k, got pool_k={pool_k}, out_k={out_k}")

    out = np.empty((n, out_k), dtype=np.int32)
    progress = tqdm(range(n), total=n, desc="mean_chain rerank", unit="q")
    for q in progress:
        pool = base_nbrs[q]
        query_emb = embeddings[q]
        cand_emb = embeddings[pool]

        selected: list[int] = []
        used = np.zeros(pool_k, dtype=bool)
        current_emb = query_emb
        running_sum = np.zeros_like(query_emb)
        n_steps = min(mean_steps, out_k, pool_k)
        for step in range(n_steps):
            step_scores = _blend_scores(
                cand_emb,
                base_scores=base_scores[q],
                query_emb=query_emb,
                probe_emb=current_emb,
                weight_base=0.0,
                weight_query=0.0,
                weight_probe=1.0,
            )
            step_scores[used] = float("-inf")

            best_idx = int(np.argmax(step_scores))
            best_id = int(pool[best_idx])
            selected.append(best_id)
            used[best_idx] = True
            running_sum += embeddings[best_id]

            if step + 1 >= n_steps:
                continue
            neighbor_vec = embeddings[best_id]
            merged = 0.5 * query_emb + 0.5 * neighbor_vec
            current_emb = _normalize_vector(merged) if np.any(merged) else query_emb

        if len(selected) < out_k:
            selected_set = set(selected)
            for cand in pool.tolist():
                cand_int = int(cand)
                if cand_int not in selected_set:
                    selected.append(cand_int)
                    selected_set.add(cand_int)
                    if len(selected) == out_k:
                        break

        out[q] = np.asarray(selected[:out_k], dtype=np.int32)
        if (q + 1) % 256 == 0 or (q + 1) == n:
            progress.set_postfix(selected=len(selected), last_q=q + 1)
    return out


def _greedy_mean_chain_rerank(
    embeddings: np.ndarray,
    base_nbrs: np.ndarray,
    base_scores: np.ndarray,
    *,
    out_k: int,
    mean_steps: int,
) -> np.ndarray:
    return _stage1_rerank(
        embeddings,
        base_nbrs,
        base_scores,
        out_k=out_k,
        mean_steps=mean_steps,
    )


def _build_reverse_index(nbrs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = nbrs.shape
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = nbrs.ravel().astype(np.int32, copy=False)
    pos = np.tile(np.arange(k, dtype=np.int16), n)
    order = np.argsort(cols, kind="stable")
    cols_sorted = cols[order]
    rows_sorted = rows[order]
    pos_sorted = pos[order]
    counts = np.bincount(cols_sorted, minlength=n)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, rows_sorted, pos_sorted


def _graph_stats(nbrs: np.ndarray, name: str) -> np.ndarray:
    n, k = nbrs.shape
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = nbrs.ravel().astype(np.int32, copy=False)
    graph = csr_matrix((np.ones(n * k, np.float32), (rows, cols)), shape=(n, n))
    recip_rate = graph.multiply(graph.T).nnz / max(graph.nnz, 1)
    in_deg = np.asarray(graph.sum(axis=0)).ravel().astype(np.int32, copy=False)
    print(
        f"[{name}] recip={recip_rate:.4f}  "
        f"in_deg: median={np.median(in_deg):.0f} "
        f"p95={np.percentile(in_deg, 95):.0f} "
        f"max={in_deg.max():.0f}"
    )
    return in_deg


def _rerank_pass(
    embeddings: np.ndarray,
    nbrs: np.ndarray,
    in_deg: np.ndarray,
    *,
    avg_sims: np.ndarray,
    w_rank: float,
    w_recip: float,
    w_common: float,
    w_hub: float,
    w_cosine: float,
    w_mutual: float,
    w_csls: float,
    pool_hop_depth: int,
) -> np.ndarray:
    n, k = nbrs.shape
    rank_w = (k - np.arange(k, dtype=np.float32)) / float(k)
    offsets, rev_rows, rev_pos = _build_reverse_index(nbrs)

    with np.errstate(divide="ignore"):
        hub_pen_all = np.log(in_deg.astype(np.float32) / float(k))
    hub_pen_all = np.clip(hub_pen_all, a_min=0.0, a_max=None)

    out = np.empty_like(nbrs)
    t0 = time.time()
    progress = tqdm(range(n), total=n, desc="rerank pass", unit="q")
    for q in progress:
        row_q = nbrs[q]
        pool_set = set(int(x) for x in row_q.tolist())
        frontier = [int(x) for x in row_q.tolist()]
        for _ in range(max(pool_hop_depth, 1)):
            next_frontier: list[int] = []
            for nb_int in frontier:
                neighbors = nbrs[nb_int]
                pool_set.update(int(x) for x in neighbors.tolist())
                next_frontier.extend(int(x) for x in neighbors.tolist())
            frontier = next_frontier
        pool_set.discard(q)

        if not pool_set:
            out[q] = row_q
            continue

        pool_arr = np.fromiter(pool_set, dtype=np.int32, count=len(pool_set))
        pool_index = {int(c): idx for idx, c in enumerate(pool_arr.tolist())}
        rank_score = np.zeros(pool_arr.size, dtype=np.float32)
        recip_score = np.zeros(pool_arr.size, dtype=np.float32)
        common_score = np.zeros(pool_arr.size, dtype=np.float32)
        mutual_score = np.zeros(pool_arr.size, dtype=np.float32)
        query_cos = embeddings[pool_arr] @ embeddings[q]
        csls_score = (2.0 * query_cos - avg_sims[q] - avg_sims[pool_arr]).astype(np.float32, copy=False)

        for pos_q, nb in enumerate(row_q):
            nb_int = int(nb)
            idx_nb = pool_index.get(nb_int)
            if idx_nb is not None:
                rank_score[idx_nb] = rank_w[pos_q]

            start = int(offsets[nb_int])
            end = int(offsets[nb_int + 1])
            if start == end:
                continue
            q_weight = rank_w[pos_q]
            rows_slice = rev_rows[start:end]
            pos_slice = rev_pos[start:end]
            for row_c, pos_c in zip(rows_slice.tolist(), pos_slice.tolist()):
                row_c = int(row_c)
                if row_c == q:
                    continue
                idx_c = pool_index.get(row_c)
                if idx_c is None:
                    continue
                common_score[idx_c] += q_weight * float(rank_w[int(pos_c)]) / float(k)

        start_q = int(offsets[q])
        end_q = int(offsets[q + 1])
        if start_q != end_q:
            rows_slice = rev_rows[start_q:end_q]
            pos_slice = rev_pos[start_q:end_q]
            for row_c, pos_c in zip(rows_slice.tolist(), pos_slice.tolist()):
                idx_c = pool_index.get(int(row_c))
                if idx_c is not None:
                    recip_score[idx_c] = rank_w[int(pos_c)]
                    mutual_score[idx_c] = 1.0

        scores = (
            w_rank * rank_score
            + w_recip * recip_score
            + w_common * common_score
            - w_hub * hub_pen_all[pool_arr]
            + w_cosine * query_cos
            + w_mutual * mutual_score
            + w_csls * csls_score
        )

        if pool_arr.size <= k:
            order = np.argsort(-scores, kind="stable")
            top = pool_arr[order][:k]
            if top.size < k:
                have = set(top.tolist())
                fill = [int(x) for x in row_q if int(x) not in have][: k - top.size]
                top = np.concatenate([top, np.asarray(fill, dtype=np.int32)])
        else:
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx], kind="stable")]
            top = pool_arr[idx]
        out[q] = top[:k]
        if (q + 1) % 256 == 0 or (q + 1) == n:
            dt = max(time.time() - t0, 1e-6)
            progress.set_postfix(rate=f"{(q + 1) / dt:.0f}/s")

    return out


def _community_postprocess(
    nbrs: np.ndarray,
    *,
    community_min: int,
    community_max: int,
) -> np.ndarray:
    n, k = nbrs.shape
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = nbrs.ravel().astype(np.int32, copy=False)
    graph = csr_matrix((np.ones(n * k, np.float32), (rows, cols)), shape=(n, n))
    mutual = graph.multiply(graph.T)
    sym = mutual + mutual.T
    _, labels = connected_components(sym, directed=False)

    comp_members: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        comp_members[int(lab)].append(i)

    sizes = np.bincount(labels)
    print(
        f"    communities: total={len(comp_members)}  "
        f"singleton={(sizes == 1).sum()}  "
        f"size 2..11={((sizes >= 2) & (sizes <= 11)).sum()}  "
        f"size 12..50={((sizes >= 12) & (sizes <= 50)).sum()}  "
        f"size >50={(sizes > 50).sum()}  max={sizes.max()}"
    )

    row_sets = [frozenset(int(x) for x in row) for row in nbrs]
    out = nbrs.copy()
    overridden = 0

    progress = tqdm(comp_members.values(), total=len(comp_members), desc="community", unit="comp")
    for members in progress:
        sz = len(members)
        if sz < community_min or sz > community_max:
            continue
        for q in members:
            q_set = row_sets[q]
            others = [m for m in members if m != q]

            def strength(m: int) -> tuple[float, int]:
                score = 0.0
                if m in q_set:
                    score += 1.0
                if q in row_sets[m]:
                    score += 1.0
                score += 0.1 * len(q_set & row_sets[m])
                return (-score, m)

            others.sort(key=strength)
            if len(others) >= k:
                new_row = others[:k]
            else:
                seen = set(others)
                extras = [int(x) for x in nbrs[q] if int(x) != q and int(x) not in seen]
                new_row = others + extras[: k - len(others)]
                if len(new_row) < k:
                    pad = 0
                    while len(new_row) < k:
                        if pad != q and pad not in new_row:
                            new_row.append(pad)
                        pad += 1

            out[q] = np.asarray(new_row[:k], dtype=np.int32)
            overridden += 1
        progress.set_postfix(overridden=overridden)

    print(f"    overridden by community: {overridden}/{n}")
    return out


def _validate(nbrs: np.ndarray) -> None:
    n, k = nbrs.shape
    for i in range(n):
        row = nbrs[i]
        if len(set(row.tolist())) != k:
            raise ValueError(f"row {i}: duplicate indices {row.tolist()}")
        if (row < 0).any():
            raise ValueError(f"row {i}: negative index")
        if i in set(row.tolist()):
            raise ValueError(f"row {i}: self-index")


def _resolve_output_prefix(input_path: Path, output_prefix: Path | None) -> Path:
    if output_prefix is not None:
        return output_prefix.expanduser().resolve()
    stem = input_path.name
    if stem.endswith(".pkl"):
        stem = stem[:-4]
    return (input_path.parent / f"{stem}_mean_chain_v2").resolve()


def _format_neighbors_csv_filepath(raw: str, filepath_prefix: str | None) -> str:
    """Match submission layout, e.g. ``test_public/000000.flac`` when prefix is ``test_public``."""
    raw = str(raw).strip().strip('"')
    if not filepath_prefix:
        return raw
    pfx = filepath_prefix.strip().rstrip("/")
    if not pfx:
        return raw
    norm = raw.replace("\\", "/")
    if norm.startswith(f"{pfx}/"):
        return norm
    base = Path(raw).name
    return f"{pfx}/{base}"


def _write_neighbors_csv(
    payload: dict[str, Any],
    nbrs: np.ndarray,
    path: Path,
    filepath_prefix: str | None = None,
) -> None:
    filepaths = payload.get("trial_rel_paths") or payload.get("audio_paths")
    if filepaths is None:
        raise KeyError("Payload has neither 'trial_rel_paths' nor 'audio_paths'")
    rows = ["filepath,neighbours"]
    for filepath, row in zip(filepaths, nbrs):
        out_fp = _format_neighbors_csv_filepath(filepath, filepath_prefix)
        rows.append(f'{out_fp},"{",".join(str(int(x)) for x in row)}"')
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_pkl", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-pool-k", type=int, default=200)
    parser.add_argument("--mean-steps", type=int, default=40)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--w-rank", type=float, default=1.0)
    parser.add_argument("--w-recip", type=float, default=1.5)
    parser.add_argument("--w-common", type=float, default=2.0)
    parser.add_argument("--w-hub", type=float, default=0.3)
    parser.add_argument("--community-min", type=int, default=2)
    parser.add_argument("--community-max", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--knn-chunk-size", type=int, default=1024)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip precision@k evaluation for unlabeled embeddings.",
    )
    parser.add_argument(
        "--neighbors-filepath-prefix",
        type=str,
        default=None,
        help="If set (e.g. test_public), CSV filepath column is '<prefix>/<basename>' "
        "(submission layout). Ignored when the path already starts with '<prefix>/'.",
    )
    args = parser.parse_args()

    input_path = args.embeddings_pkl.expanduser().resolve()
    payload = _load_payload(input_path)
    embeddings = _normalize_embeddings(np.asarray(payload["embeddings"], dtype=np.float32))
    audio_ids = [str(x) for x in payload["audio_ids"]]
    speaker_ids = _speaker_ids_from_audio_ids(audio_ids)
    speaker_labels = _build_speaker_labels(speaker_ids)
    metric_ks = [int(k) for k in DEFAULT_METRIC_KS if int(k) <= int(args.k)]

    if args.candidate_pool_k < args.k:
        raise ValueError("--candidate-pool-k must be >= --k")

    output_prefix = _resolve_output_prefix(
        input_path,
        args.output_prefix or (input_path.parent / f"{input_path.stem}_mean_chain_v2"),
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {input_path}")
    print(
        f"items={len(audio_ids)} speakers={len(set(speaker_ids))} "
        f"k={args.k} pool_k={args.candidate_pool_k} mean_steps={args.mean_steps} "
        f"n_iter={args.n_iter}"
    )

    print("building candidate pool ...")
    base_nbrs, base_scores = _compute_candidate_pool(
        embeddings,
        pool_k=int(args.candidate_pool_k),
        device=str(args.device),
        chunk_size=int(args.knn_chunk_size),
    )
    baseline_nbrs = base_nbrs[:, : int(args.k)]
    baseline_metrics = (
        _empty_metrics(metric_ks)
        if args.skip_metrics
        else _precision_at_ks_from_neighbors(baseline_nbrs, speaker_labels, metric_ks)
    )
    _graph_stats(baseline_nbrs, "baseline")

    print("running mean-chain ...")
    mean_chain_nbrs = _greedy_mean_chain_rerank(
        embeddings,
        base_nbrs,
        base_scores,
        out_k=int(args.k),
        mean_steps=int(args.mean_steps),
    )
    mean_chain_metrics = (
        _empty_metrics(metric_ks)
        if args.skip_metrics
        else _precision_at_ks_from_neighbors(mean_chain_nbrs, speaker_labels, metric_ks)
    )
    _graph_stats(mean_chain_nbrs, "mean_chain")

    avg_sims = np.zeros(embeddings.shape[0], dtype=np.float32)
    cur = mean_chain_nbrs.copy()
    for it in range(1, int(args.n_iter) + 1):
        print(f"iter {it}: v2 rerank pass ...")
        in_deg = np.bincount(cur.ravel(), minlength=cur.shape[0]).astype(np.int32, copy=False)
        cur = _rerank_pass(
            embeddings,
            cur,
            in_deg,
            avg_sims=avg_sims,
            w_rank=float(args.w_rank),
            w_recip=float(args.w_recip),
            w_common=float(args.w_common),
            w_hub=float(args.w_hub),
            w_cosine=0.0,
            w_mutual=0.0,
            w_csls=0.0,
            pool_hop_depth=1,
        )
        _graph_stats(cur, f"iter {it}")

    print("community post-process ...")
    cur = _community_postprocess(
        cur,
        community_min=int(args.community_min),
        community_max=int(args.community_max),
    )
    _graph_stats(cur, "final")

    print("validating ...")
    _validate(cur)
    print("  OK")

    final_metrics = (
        _empty_metrics(metric_ks)
        if args.skip_metrics
        else _precision_at_ks_from_neighbors(cur, speaker_labels, metric_ks)
    )
    if args.skip_metrics:
        print("precision@k metrics skipped")
    else:
        for k in metric_ks:
            print(
                f"precision@{k}: baseline={baseline_metrics[k]:.6f} "
                f"mean_chain={mean_chain_metrics[k]:.6f} "
                f"final={final_metrics[k]:.6f}"
            )

    changed_vs_baseline = (
        np.sort(cur, axis=1) != np.sort(baseline_nbrs, axis=1)
    ).any(axis=1).sum()
    changed_vs_mean_chain = (
        np.sort(cur, axis=1) != np.sort(mean_chain_nbrs, axis=1)
    ).any(axis=1).sum()
    print(f"rows with set change vs baseline: {changed_vs_baseline}/{len(audio_ids)}")
    print(f"rows with set change vs mean-chain: {changed_vs_mean_chain}/{len(audio_ids)}")

    csv_path = output_prefix.with_suffix(".neighbors.csv")
    metrics_path = output_prefix.with_suffix(".metrics.json")
    _write_neighbors_csv(
        payload,
        cur,
        csv_path,
        filepath_prefix=args.neighbors_filepath_prefix,
    )
    summary = {
        "input_embeddings_pkl": str(input_path),
        "output_neighbors_csv": str(csv_path),
        "neighbors_filepath_prefix": args.neighbors_filepath_prefix,
        "num_items": int(len(audio_ids)),
        "num_speakers": int(len(set(speaker_ids))),
        "k": int(args.k),
        "candidate_pool_k": int(args.candidate_pool_k),
        "mean_steps": int(args.mean_steps),
        "n_iter": int(args.n_iter),
        "weights": {
            "rank": float(args.w_rank),
            "recip": float(args.w_recip),
            "common": float(args.w_common),
            "hub": float(args.w_hub),
        },
        "baseline_metrics": _serialize_metrics(baseline_metrics),
        "mean_chain_metrics": _serialize_metrics(mean_chain_metrics),
        "final_metrics": _serialize_metrics(final_metrics),
        "rows_with_set_change_vs_baseline": int(changed_vs_baseline),
        "rows_with_set_change_vs_mean_chain": int(changed_vs_mean_chain),
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote neighbors: {csv_path}")
    print(f"wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
