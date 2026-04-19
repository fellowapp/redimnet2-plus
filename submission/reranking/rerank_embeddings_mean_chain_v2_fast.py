"""Объединённый ре-ранкер: mean-chain + графовый re-rank v2.

Вход — pickle с ``{embeddings, audio_ids}`` (например, из ``run_embeddings.py`` или
``run_embeddings_trt.py``). Выход — CSV с топ-K ближайшими соседями для каждого
файла и (опционально) метриками качества. Используется `numba` для ускорения
тяжёлых циклов, если установлена."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context, shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

DEFAULT_MAIN_METRIC_KS = (5, 10, 20, 45)
DEFAULT_METRIC_KS = DEFAULT_MAIN_METRIC_KS
_RERANK_WORKER_STATE: dict[str, Any] = {}


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


def _is_nonzero_weight(value: float) -> bool:
    return abs(float(value)) > 0.0


def _compute_candidate_pool(
    embeddings: np.ndarray,
    *,
    pool_k: int,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(embeddings.shape[0])
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    emb_dev = torch.as_tensor(emb, dtype=torch.float32, device=device)
    emb_dev_t = emb_dev.T

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
        sims = emb_dev[start:end] @ emb_dev_t
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


if njit is not None:

    @njit(cache=False)
    def _stage1_rerank_numba_chunk(
        embeddings: np.ndarray,
        base_nbrs_chunk: np.ndarray,
        *,
        query_start: int,
        out_k: int,
        mean_steps: int,
    ) -> np.ndarray:
        n_chunk, pool_k = base_nbrs_chunk.shape
        dim = embeddings.shape[1]
        out = np.empty((n_chunk, out_k), dtype=np.int32)
        fill_order = np.arange(pool_k, dtype=np.int32)
        half = np.float32(0.5)
        neg_inf = np.float32(-np.inf)

        for q_local in range(n_chunk):
            q = query_start + q_local
            pool = base_nbrs_chunk[q_local]
            query_emb = embeddings[q]
            cand_emb = embeddings[pool]

            used = np.zeros(pool_k, dtype=np.uint8)
            current_emb = query_emb
            current_buf = np.empty(dim, dtype=np.float32)
            n_steps = min(mean_steps, out_k, pool_k)
            selected = np.empty(out_k, dtype=np.int32)
            selected_count = 0

            for step in range(n_steps):
                step_scores = cand_emb @ current_emb
                for idx in range(pool_k):
                    if used[idx] != 0:
                        step_scores[idx] = neg_inf

                best_idx = int(np.argmax(step_scores))
                best_id = int(pool[best_idx])
                selected[selected_count] = best_id
                selected_count += 1
                used[best_idx] = 1

                if step + 1 >= n_steps:
                    continue

                neighbor_vec = embeddings[best_id]
                any_nonzero = False
                for dim_idx in range(dim):
                    val = half * query_emb[dim_idx] + half * neighbor_vec[dim_idx]
                    current_buf[dim_idx] = val
                    if val != 0.0:
                        any_nonzero = True

                if any_nonzero:
                    denom = float(np.linalg.norm(current_buf))
                    if denom <= 0.0:
                        for dim_idx in range(dim):
                            current_buf[dim_idx] = 0.0
                    else:
                        scaled = current_buf / denom
                        for dim_idx in range(dim):
                            current_buf[dim_idx] = np.float32(scaled[dim_idx])
                    current_emb = current_buf
                else:
                    current_emb = query_emb

            if selected_count < out_k:
                fill_count = 0
                for idx in range(pool_k):
                    fill_idx = fill_order[idx]
                    if used[fill_idx] == 0:
                        selected[selected_count + fill_count] = pool[fill_idx]
                        fill_count += 1
                        if selected_count + fill_count >= out_k:
                            break

            out[q_local] = selected

        return out


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

    if njit is not None:
        out = np.empty((n, out_k), dtype=np.int32)
        chunk_size = 4096
        progress = tqdm(total=n, desc="mean_chain rerank", unit="q")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            out[start:end] = _stage1_rerank_numba_chunk(
                embeddings,
                base_nbrs[start:end],
                query_start=start,
                out_k=out_k,
                mean_steps=mean_steps,
            )
            progress.update(end - start)
            progress.set_postfix(last_q=end, selected=out_k)
        progress.close()
        return out

    out = np.empty((n, out_k), dtype=np.int32)
    fill_order = np.arange(pool_k, dtype=np.int32)
    progress = tqdm(range(n), total=n, desc="mean_chain rerank", unit="q")
    for q in progress:
        pool = base_nbrs[q]
        query_emb = embeddings[q]
        cand_emb = embeddings[pool]

        used = np.zeros(pool_k, dtype=bool)
        current_emb = query_emb
        n_steps = min(mean_steps, out_k, pool_k)
        selected = np.empty(out_k, dtype=np.int32)
        selected_count = 0
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
            selected[selected_count] = best_id
            selected_count += 1
            used[best_idx] = True

            if step + 1 >= n_steps:
                continue
            neighbor_vec = embeddings[best_id]
            merged = 0.5 * query_emb + 0.5 * neighbor_vec
            current_emb = _normalize_vector(merged) if np.any(merged) else query_emb

        if selected_count < out_k:
            fill_idx = fill_order[~used][: out_k - selected_count]
            selected[selected_count:] = pool[fill_idx]

        out[q] = selected
        if (q + 1) % 256 == 0 or (q + 1) == n:
            progress.set_postfix(selected=selected_count, last_q=q + 1)
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


def _create_shared_array(array: np.ndarray) -> tuple[shared_memory.SharedMemory, dict[str, Any]]:
    arr = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    desc = {
        "name": shm.name,
        "shape": arr.shape,
        "dtype": arr.dtype.str,
    }
    return shm, desc


def _attach_shared_array(desc: dict[str, Any]) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(name=str(desc["name"]))
    arr = np.ndarray(tuple(desc["shape"]), dtype=np.dtype(str(desc["dtype"])), buffer=shm.buf)
    return shm, arr


def _set_rerank_worker_state(
    embeddings: np.ndarray,
    *,
    embeddings_shm: shared_memory.SharedMemory | None,
    nbrs: np.ndarray,
    offsets: np.ndarray,
    rev_rows: np.ndarray,
    rev_pos: np.ndarray,
    hub_pen_all: np.ndarray,
    avg_sims: np.ndarray,
    rank_w: np.ndarray,
    weights: tuple[float, float, float, float, float, float, float],
) -> None:
    _RERANK_WORKER_STATE.clear()
    _RERANK_WORKER_STATE.update(
        {
            "embeddings_shm": embeddings_shm,
            "embeddings": embeddings,
            "nbrs": nbrs,
            "offsets": offsets,
            "rev_rows": rev_rows,
            "rev_pos": rev_pos,
            "hub_pen_all": hub_pen_all,
            "avg_sims": avg_sims,
            "rank_w": rank_w,
            "weights": weights,
        }
    )


def _init_rerank_worker(
    embeddings_desc: dict[str, Any],
    nbrs: np.ndarray,
    offsets: np.ndarray,
    rev_rows: np.ndarray,
    rev_pos: np.ndarray,
    hub_pen_all: np.ndarray,
    avg_sims: np.ndarray,
    rank_w: np.ndarray,
    weights: tuple[float, float, float, float, float, float, float],
) -> None:
    shm, embeddings = _attach_shared_array(embeddings_desc)
    _set_rerank_worker_state(
        embeddings,
        embeddings_shm=shm,
        nbrs=nbrs,
        offsets=offsets,
        rev_rows=rev_rows,
        rev_pos=rev_pos,
        hub_pen_all=hub_pen_all,
        avg_sims=avg_sims,
        rank_w=rank_w,
        weights=weights,
    )


if njit is not None:

    @njit(cache=False)
    def _rerank_default_score_terms_numba(
        q: int,
        row_q: np.ndarray,
        pool_arr: np.ndarray,
        lookup: np.ndarray,
        offsets: np.ndarray,
        rev_rows: np.ndarray,
        rev_pos: np.ndarray,
        rank_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = int(row_q.shape[0])
        pool_size = int(pool_arr.shape[0])
        rank_score = np.zeros(pool_size, dtype=np.float32)
        recip_score = np.zeros(pool_size, dtype=np.float32)
        common_score = np.zeros(pool_size, dtype=np.float32)

        for pos_q in range(k):
            nb_int = int(row_q[pos_q])
            idx_nb = int(lookup[nb_int])
            if idx_nb != -1:
                rank_score[idx_nb] = rank_w[pos_q]

            start = int(offsets[nb_int])
            end = int(offsets[nb_int + 1])
            if start == end:
                continue

            q_weight = rank_w[pos_q]
            for idx_local in range(start, end):
                row_c = int(rev_rows[idx_local])
                if row_c == q:
                    continue
                idx_c = int(lookup[row_c])
                if idx_c == -1:
                    continue
                pos_c = int(rev_pos[idx_local])
                rhs = np.float32(q_weight * float(rank_w[pos_c]) / float(k))
                common_score[idx_c] += rhs

        start_q = int(offsets[q])
        end_q = int(offsets[q + 1])
        if start_q != end_q:
            for idx_local in range(start_q, end_q):
                idx_c = int(lookup[int(rev_rows[idx_local])])
                if idx_c != -1:
                    recip_score[idx_c] = rank_w[int(rev_pos[idx_local])]

        return rank_score, recip_score, common_score


if njit is not None:

    @njit(cache=False)
    def _community_rank_component(member_rows: np.ndarray, members_arr: np.ndarray) -> np.ndarray:
        sz, k = member_rows.shape
        out = np.empty((sz, sz - 1), dtype=np.int32)

        for i in range(sz):
            q = int(members_arr[i])
            ids = np.empty(sz - 1, dtype=np.int32)
            strengths = np.empty(sz - 1, dtype=np.int16)
            count = 0
            row_q = member_rows[i]

            for j in range(sz):
                if i == j:
                    continue

                m = int(members_arr[j])
                row_m = member_rows[j]
                strength = 0

                for idx in range(k):
                    if int(row_q[idx]) == m:
                        strength += 10
                    if int(row_m[idx]) == q:
                        strength += 10

                common = 0
                for idx_q in range(k):
                    val_q = int(row_q[idx_q])
                    for idx_m in range(k):
                        if val_q == int(row_m[idx_m]):
                            common += 1
                            break
                strength += common

                ids[count] = m
                strengths[count] = np.int16(strength)
                count += 1

            for pos in range(1, count):
                cur_id = ids[pos]
                cur_strength = strengths[pos]
                prev = pos - 1
                while prev >= 0:
                    prev_strength = strengths[prev]
                    prev_id = ids[prev]
                    if prev_strength > cur_strength:
                        break
                    if prev_strength == cur_strength and prev_id < cur_id:
                        break
                    ids[prev + 1] = prev_id
                    strengths[prev + 1] = prev_strength
                    prev -= 1
                ids[prev + 1] = cur_id
                strengths[prev + 1] = cur_strength

            out[i] = ids

        return out


def _rerank_pass_rows_default_numba(
    embeddings: np.ndarray,
    nbrs: np.ndarray,
    offsets: np.ndarray,
    rev_rows: np.ndarray,
    rev_pos: np.ndarray,
    hub_pen_all: np.ndarray,
    rank_w: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    w_rank: float,
    w_recip: float,
    w_common: float,
    w_hub: float,
) -> np.ndarray:
    n, k = nbrs.shape
    out = np.empty((end_idx - start_idx, k), dtype=np.int32)
    lookup = np.full(n, -1, dtype=np.int32)

    for out_row, q in enumerate(range(start_idx, end_idx)):
        row_q = nbrs[q]
        row_q_list = [int(x) for x in row_q]
        pool_set = set(row_q_list)
        next_frontier: list[int] = []
        for nb_int in row_q_list:
            neighbors = nbrs[nb_int]
            for cand in neighbors:
                cand_int = int(cand)
                pool_set.add(cand_int)
                next_frontier.append(cand_int)
        pool_set.discard(q)

        if not pool_set:
            out[out_row] = row_q
            continue

        pool_arr = np.fromiter(pool_set, dtype=np.int32, count=len(pool_set))
        lookup[pool_arr] = np.arange(pool_arr.size, dtype=np.int32)
        rank_score, recip_score, common_score = _rerank_default_score_terms_numba(
            q,
            row_q,
            pool_arr,
            lookup,
            offsets,
            rev_rows,
            rev_pos,
            rank_w,
        )
        lookup[pool_arr] = -1

        scores = (
            w_rank * rank_score
            + w_recip * recip_score
            + w_common * common_score
            - w_hub * hub_pen_all[pool_arr]
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
        out[out_row] = top[:k]

    return out


def _rerank_pass_range(task: tuple[int, int]) -> tuple[int, np.ndarray]:
    start_idx, end_idx = task
    embeddings = _RERANK_WORKER_STATE["embeddings"]
    nbrs = _RERANK_WORKER_STATE["nbrs"]
    offsets = _RERANK_WORKER_STATE["offsets"]
    rev_rows = _RERANK_WORKER_STATE["rev_rows"]
    rev_pos = _RERANK_WORKER_STATE["rev_pos"]
    hub_pen_all = _RERANK_WORKER_STATE["hub_pen_all"]
    avg_sims = _RERANK_WORKER_STATE["avg_sims"]
    rank_w = _RERANK_WORKER_STATE["rank_w"]
    w_rank, w_recip, w_common, w_hub, w_cosine, w_mutual, w_csls = _RERANK_WORKER_STATE["weights"]

    k = int(nbrs.shape[1])
    use_cosine = _is_nonzero_weight(w_cosine)
    use_csls = _is_nonzero_weight(w_csls)
    use_mutual = _is_nonzero_weight(w_mutual)
    use_default_numba = njit is not None and (not use_cosine) and (not use_csls) and (not use_mutual)

    if use_default_numba:
        out = _rerank_pass_rows_default_numba(
            embeddings,
            nbrs,
            offsets,
            rev_rows,
            rev_pos,
            hub_pen_all,
            rank_w,
            start_idx=start_idx,
            end_idx=end_idx,
            w_rank=float(w_rank),
            w_recip=float(w_recip),
            w_common=float(w_common),
            w_hub=float(w_hub),
        )
        shm = _RERANK_WORKER_STATE.get("embeddings_shm")
        if shm is not None:
            shm.close()
            _RERANK_WORKER_STATE["embeddings_shm"] = None
        return start_idx, out

    out = np.empty((end_idx - start_idx, k), dtype=np.int32)

    for out_row, q in enumerate(range(start_idx, end_idx)):
        row_q = nbrs[q]
        row_q_list = [int(x) for x in row_q]
        pool_set = set(row_q_list)
        frontier = row_q_list
        next_frontier: list[int] = []
        for nb_int in frontier:
            neighbors = nbrs[nb_int]
            for cand in neighbors:
                cand_int = int(cand)
                pool_set.add(cand_int)
                next_frontier.append(cand_int)
        pool_set.discard(q)

        if not pool_set:
            out[out_row] = row_q
            continue

        pool_arr = np.fromiter(pool_set, dtype=np.int32, count=len(pool_set))
        pool_index = {int(c): idx for idx, c in enumerate(pool_arr)}
        rank_score = np.zeros(pool_arr.size, dtype=np.float32)
        recip_score = np.zeros(pool_arr.size, dtype=np.float32)
        common_score = np.zeros(pool_arr.size, dtype=np.float32)
        mutual_score = np.zeros(pool_arr.size, dtype=np.float32) if use_mutual else None
        query_cos = (embeddings[pool_arr] @ embeddings[q]) if (use_cosine or use_csls) else None
        csls_score = (
            (2.0 * query_cos - avg_sims[q] - avg_sims[pool_arr]).astype(np.float32, copy=False)
            if use_csls
            else None
        )

        for pos_q, nb_int in enumerate(row_q_list):
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
            for idx_local in range(rows_slice.shape[0]):
                row_c = int(rows_slice[idx_local])
                if row_c == q:
                    continue
                idx_c = pool_index.get(row_c)
                if idx_c is None:
                    continue
                pos_c = int(pos_slice[idx_local])
                common_score[idx_c] += q_weight * float(rank_w[pos_c]) / float(k)

        start_q = int(offsets[q])
        end_q = int(offsets[q + 1])
        if start_q != end_q:
            rows_slice = rev_rows[start_q:end_q]
            pos_slice = rev_pos[start_q:end_q]
            for idx_local in range(rows_slice.shape[0]):
                idx_c = pool_index.get(int(rows_slice[idx_local]))
                if idx_c is not None:
                    recip_score[idx_c] = rank_w[int(pos_slice[idx_local])]
                    if mutual_score is not None:
                        mutual_score[idx_c] = 1.0

        query_cos_term = query_cos if query_cos is not None else 0.0
        mutual_term = mutual_score if mutual_score is not None else 0.0
        csls_term = csls_score if csls_score is not None else 0.0
        scores = (
            w_rank * rank_score
            + w_recip * recip_score
            + w_common * common_score
            - w_hub * hub_pen_all[pool_arr]
            + w_cosine * query_cos_term
            + w_mutual * mutual_term
            + w_csls * csls_term
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
        out[out_row] = top[:k]

    shm = _RERANK_WORKER_STATE.get("embeddings_shm")
    if shm is not None:
        shm.close()
        _RERANK_WORKER_STATE["embeddings_shm"] = None
    return start_idx, out


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
    cpu_workers: int = 1,
) -> np.ndarray:
    n, k = nbrs.shape
    rank_w = (k - np.arange(k, dtype=np.float32)) / float(k)
    offsets, rev_rows, rev_pos = _build_reverse_index(nbrs)
    with np.errstate(divide="ignore"):
        hub_pen_all = np.log(in_deg.astype(np.float32) / float(k))
    hub_pen_all = np.clip(hub_pen_all, a_min=0.0, a_max=None)
    use_cosine = _is_nonzero_weight(w_cosine)
    use_csls = _is_nonzero_weight(w_csls)
    use_mutual = _is_nonzero_weight(w_mutual)
    use_hub = _is_nonzero_weight(w_hub)

    if int(cpu_workers) > 1:
        worker_count = max(1, int(cpu_workers))
        step = (n + worker_count - 1) // worker_count
        tasks = [(start, min(start + step, n)) for start in range(0, n, step)]
        out = np.empty_like(nbrs)
        weights = (
            float(w_rank),
            float(w_recip),
            float(w_common),
            float(w_hub),
            float(w_cosine),
            float(w_mutual),
            float(w_csls),
        )
        use_fork = os.name == "posix" and sys.platform.startswith("linux")
        if use_fork:
            _set_rerank_worker_state(
                embeddings,
                embeddings_shm=None,
                nbrs=nbrs,
                offsets=offsets,
                rev_rows=rev_rows,
                rev_pos=rev_pos,
                hub_pen_all=hub_pen_all,
                avg_sims=avg_sims,
                rank_w=rank_w,
                weights=weights,
            )
            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=get_context("fork"),
                ) as executor:
                    for start_idx, chunk in executor.map(_rerank_pass_range, tasks):
                        out[start_idx : start_idx + chunk.shape[0]] = chunk
                return out
            finally:
                _RERANK_WORKER_STATE.clear()

        emb_shm, emb_desc = _create_shared_array(embeddings)
        try:
            ctx = get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=ctx,
                initializer=_init_rerank_worker,
                initargs=(
                    emb_desc,
                    nbrs,
                    offsets,
                    rev_rows,
                    rev_pos,
                    hub_pen_all,
                    avg_sims,
                    rank_w,
                    weights,
                ),
            ) as executor:
                for start_idx, chunk in executor.map(_rerank_pass_range, tasks):
                    out[start_idx : start_idx + chunk.shape[0]] = chunk
            return out
        finally:
            emb_shm.close()
            emb_shm.unlink()

    use_default_numba = njit is not None and pool_hop_depth == 1 and (not use_cosine) and (not use_csls) and (not use_mutual)
    if use_default_numba:
        out = np.empty_like(nbrs)
        t0 = time.time()
        chunk_size = 4096
        progress = tqdm(total=n, desc="rerank pass", unit="q")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            out[start:end] = _rerank_pass_rows_default_numba(
                embeddings,
                nbrs,
                offsets,
                rev_rows,
                rev_pos,
                hub_pen_all,
                rank_w,
                start_idx=start,
                end_idx=end,
                w_rank=float(w_rank),
                w_recip=float(w_recip),
                w_common=float(w_common),
                w_hub=float(w_hub),
            )
            progress.update(end - start)
            dt = max(time.time() - t0, 1e-6)
            progress.set_postfix(rate=f"{end / dt:.0f}/s")
        progress.close()
        return out

    out = np.empty_like(nbrs)
    t0 = time.time()
    progress = tqdm(range(n), total=n, desc="rerank pass", unit="q")
    for q in progress:
        row_q = nbrs[q]
        row_q_list = [int(x) for x in row_q]
        pool_set = set(row_q_list)
        frontier = row_q_list
        for _ in range(max(pool_hop_depth, 1)):
            next_frontier: list[int] = []
            for nb_int in frontier:
                neighbors = nbrs[nb_int]
                for cand in neighbors:
                    cand_int = int(cand)
                    pool_set.add(cand_int)
                    next_frontier.append(cand_int)
            frontier = next_frontier
        pool_set.discard(q)

        if not pool_set:
            out[q] = row_q
            continue

        pool_arr = np.fromiter(pool_set, dtype=np.int32, count=len(pool_set))
        pool_index = {int(c): idx for idx, c in enumerate(pool_arr)}
        rank_score = np.zeros(pool_arr.size, dtype=np.float32)
        recip_score = np.zeros(pool_arr.size, dtype=np.float32)
        common_score = np.zeros(pool_arr.size, dtype=np.float32)
        reciprocal_hits = np.zeros(pool_arr.size, dtype=np.float32) if use_mutual else None
        query_cos = (embeddings[pool_arr] @ embeddings[q]) if (use_cosine or use_csls) else None
        csls_score = (
            (2.0 * query_cos - avg_sims[q] - avg_sims[pool_arr]).astype(np.float32, copy=False)
            if use_csls
            else None
        )

        for pos_q, nb_int in enumerate(row_q_list):
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
            for idx_local in range(rows_slice.shape[0]):
                row_c = int(rows_slice[idx_local])
                if row_c == q:
                    continue
                idx_c = pool_index.get(row_c)
                if idx_c is None:
                    continue
                pos_c = int(pos_slice[idx_local])
                common_score[idx_c] += q_weight * float(rank_w[pos_c]) / float(k)

        start_q = int(offsets[q])
        end_q = int(offsets[q + 1])
        if start_q != end_q:
            rows_slice = rev_rows[start_q:end_q]
            pos_slice = rev_pos[start_q:end_q]
            for idx_local in range(rows_slice.shape[0]):
                idx_c = pool_index.get(int(rows_slice[idx_local]))
                if idx_c is not None:
                    recip_score[idx_c] = rank_w[int(pos_slice[idx_local])]
                    if reciprocal_hits is not None:
                        reciprocal_hits[idx_c] = 1.0

        if use_cosine or use_csls:
            query_cos_term = query_cos
        else:
            query_cos_term = 0.0
        mutual_term = reciprocal_hits if reciprocal_hits is not None else 0.0
        csls_term = csls_score if csls_score is not None else 0.0
        hub_term = hub_pen_all[pool_arr] if use_hub else 0.0
        scores = (
            w_rank * rank_score
            + w_recip * recip_score
            + w_common * common_score
            - w_hub * hub_term
            + w_cosine * query_cos_term
            + w_mutual * mutual_term
            + w_csls * csls_term
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

    out = nbrs.copy()
    overridden = 0

    progress = tqdm(comp_members.values(), total=len(comp_members), desc="community", unit="comp")
    for members in progress:
        sz = len(members)
        if sz < community_min or sz > community_max:
            continue
        ranked_members = None
        if njit is not None and sz > 1:
            members_arr = np.asarray(members, dtype=np.int32)
            ranked_members = _community_rank_component(nbrs[members_arr], members_arr)
        for q in members:
            if ranked_members is not None:
                q_idx = members.index(q)
                others = ranked_members[q_idx].tolist()
            else:
                row_q = nbrs[q]
                others = [m for m in members if m != q]

                def strength(m: int) -> tuple[int, int]:
                    row_m = nbrs[m]
                    score10 = 0
                    for val in row_q:
                        if int(val) == m:
                            score10 += 10
                            break
                    for val in row_m:
                        if int(val) == q:
                            score10 += 10
                            break
                    common = 0
                    for val_q in row_q:
                        for val_m in row_m:
                            if int(val_q) == int(val_m):
                                common += 1
                                break
                    score10 += common
                    return (-score10, m)

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
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="CPU workers for rerank passes in the fast script.",
    )
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

    n_items = len(audio_ids)
    if int(args.candidate_pool_k) < int(args.k):
        bumped = int(args.k) + 100
        capped = min(bumped, n_items)
        print(
            f"[info] --candidate-pool-k ({args.candidate_pool_k}) < --k ({args.k}); "
            f"using candidate_pool_k=min(k+100, n)={capped}.",
            flush=True,
        )
        args.candidate_pool_k = capped
    if int(args.candidate_pool_k) < int(args.k):
        raise ValueError(
            f"Need candidate_pool_k >= k after adjustment; got pool_k={args.candidate_pool_k}, "
            f"k={args.k}, n={n_items}."
        )

    output_prefix = _resolve_output_prefix(
        input_path,
        args.output_prefix or (input_path.parent / f"{input_path.stem}_mean_chain_v2"),
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    intermediate_output_prefix = output_prefix.with_name(
        f"INTERMEDIATE_SAVE_YOU_SHOULD_WAIT_FOR_FINAL_{output_prefix.name}"
    )
    phase_times: dict[str, float] = {}

    print(f"loading {input_path}")
    print(
        f"items={len(audio_ids)} speakers={len(set(speaker_ids))} "
        f"k={args.k} pool_k={args.candidate_pool_k} mean_steps={args.mean_steps} "
        f"n_iter={args.n_iter}"
    )

    print("building candidate pool ...")
    t_phase = time.perf_counter()
    base_nbrs, base_scores = _compute_candidate_pool(
        embeddings,
        pool_k=int(args.candidate_pool_k),
        device=str(args.device),
        chunk_size=int(args.knn_chunk_size),
    )
    phase_times["candidate_pool"] = time.perf_counter() - t_phase
    baseline_nbrs = base_nbrs[:, : int(args.k)]
    baseline_metrics = (
        _empty_metrics(metric_ks)
        if args.skip_metrics
        else _precision_at_ks_from_neighbors(baseline_nbrs, speaker_labels, metric_ks)
    )
    _graph_stats(baseline_nbrs, "baseline")

    print("running mean-chain ...")
    t_phase = time.perf_counter()
    mean_chain_nbrs = _greedy_mean_chain_rerank(
        embeddings,
        base_nbrs,
        base_scores,
        out_k=int(args.k),
        mean_steps=int(args.mean_steps),
    )
    phase_times["mean_chain"] = time.perf_counter() - t_phase
    mean_chain_metrics = (
        _empty_metrics(metric_ks)
        if args.skip_metrics
        else _precision_at_ks_from_neighbors(mean_chain_nbrs, speaker_labels, metric_ks)
    )
    _graph_stats(mean_chain_nbrs, "mean_chain")
    intermediate_csv_path = intermediate_output_prefix.with_suffix(".neighbors.csv")
    t_phase = time.perf_counter()
    _write_neighbors_csv(
        payload,
        mean_chain_nbrs,
        intermediate_csv_path,
        filepath_prefix=args.neighbors_filepath_prefix,
    )
    phase_times["write_intermediate_csv"] = time.perf_counter() - t_phase
    print(f"wrote intermediate mean-chain neighbors: {intermediate_csv_path}")

    avg_sims = np.zeros(embeddings.shape[0], dtype=np.float32)
    cur = mean_chain_nbrs.copy()
    rerank_total = 0.0
    skip_v2_rerank = int(args.k) > 100
    if skip_v2_rerank:
        print(
            "[info] k>100: skipping v2 graph-rerank stage (iterative _rerank_pass); "
            "running community on mean-chain output.",
            flush=True,
        )
        phase_times["rerank_total"] = 0.0
    else:
        for it in range(1, int(args.n_iter) + 1):
            print(f"iter {it}: v2 rerank pass ...")
            in_deg = np.bincount(cur.ravel(), minlength=cur.shape[0]).astype(np.int32, copy=False)
            t_iter = time.perf_counter()
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
                cpu_workers=int(args.cpu_workers),
            )
            iter_time = time.perf_counter() - t_iter
            rerank_total += iter_time
            phase_times[f"rerank_iter_{it}"] = iter_time
            _graph_stats(cur, f"iter {it}")
        phase_times["rerank_total"] = rerank_total

    print("community post-process ...")
    t_phase = time.perf_counter()
    cur = _community_postprocess(
        cur,
        community_min=int(args.community_min),
        community_max=int(args.community_max),
    )
    phase_times["community"] = time.perf_counter() - t_phase
    _graph_stats(cur, "final")

    print("validating ...")
    t_phase = time.perf_counter()
    _validate(cur)
    phase_times["validate"] = time.perf_counter() - t_phase
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
    t_phase = time.perf_counter()
    _write_neighbors_csv(
        payload,
        cur,
        csv_path,
        filepath_prefix=args.neighbors_filepath_prefix,
    )
    phase_times["write_csv"] = time.perf_counter() - t_phase
    summary = {
        "input_embeddings_pkl": str(input_path),
        "intermediate_mean_chain_neighbors_csv": str(intermediate_csv_path),
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
        "cpu_workers": int(args.cpu_workers),
        "v2_rerank_skipped_k_gt_50": bool(skip_v2_rerank),
        "baseline_metrics": _serialize_metrics(baseline_metrics),
        "mean_chain_metrics": _serialize_metrics(mean_chain_metrics),
        "final_metrics": _serialize_metrics(final_metrics),
        "rows_with_set_change_vs_baseline": int(changed_vs_baseline),
        "rows_with_set_change_vs_mean_chain": int(changed_vs_mean_chain),
        "phase_times_seconds": {name: float(value) for name, value in phase_times.items()},
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote neighbors: {csv_path}")
    print(f"wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
