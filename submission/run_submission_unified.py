"""Unified submission inference with multi-instance GPU embedding extraction and reranking."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from asv.data.collate import clone_frozen_frontend, compute_spectrogram_batch
from asv.data.decode import load_with_torchcodec
from asv.models.assemble import build_model
from benchmarking import checkpoint_loading
from submission.reranking.run_embeddings import _discover_audio_files
from submission.reranking import rerank_embeddings_mean_chain_v2_fast as rerank_fast


DEFAULT_MODEL_CONFIG = _REPO_ROOT / "configs" / "model" / "redimnet2_b6.yaml"
DEFAULT_MAX_DURATION_SEC = 60.0
# Weights are always resolved from the repository ./weights directory (override with --weights-dir).
DEFAULT_WEIGHTS_DIR = (_REPO_ROOT / "weights").resolve()
# Default CSV path at repository root (Docker image uses /workspace/output.csv).
DEFAULT_OUTPUT_CSV = (_REPO_ROOT / "output.csv").resolve()


def _clip_to_max_length(waveform: torch.Tensor, max_len: int) -> torch.Tensor:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if waveform.ndim == 0:
        raise ValueError("expected at least 1D waveform")
    if waveform.shape[-1] <= max_len:
        return waveform
    return waveform[..., :max_len]


def _iter_weight_candidates(search_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if search_root.is_file() and search_root.name in checkpoint_loading._WEIGHT_CANDIDATES:
        candidates.append(search_root.resolve())
        return candidates
    if search_root.is_dir():
        if (search_root / "training_state.json").is_file() or (search_root / "latest.json").is_file():
            candidates.append(search_root.resolve())
    for path in search_root.rglob("*"):
        if not path.exists():
            continue
        if path.is_file() and path.name in checkpoint_loading._WEIGHT_CANDIDATES:
            candidates.append(path.resolve())
            continue
        if path.is_dir():
            if (path / "training_state.json").is_file():
                candidates.append(path.resolve())
                continue
            if (path / "latest.json").is_file():
                candidates.append(path.resolve())
    return candidates


def _autodetect_weights_path(search_root: Path) -> Path:
    root = search_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Weights directory does not exist: {root}")
    candidates = _iter_weight_candidates(root)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint or weight files found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_weights_source(weights_dir: Path) -> Path:
    return _autodetect_weights_path(weights_dir.expanduser().resolve())


def _infer_redimnet2_encoder_overrides(
    model_cfg: Any,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    encoder_cfg = getattr(model_cfg, "encoder", None)
    if encoder_cfg is None:
        return {}
    if str(encoder_cfg.get("_target_", "")) != "asv.models.redimnet2.ReDimNet2Encoder":
        return {}

    head_weight = state_dict.get("encoder.model.backbone.head.weight")
    if head_weight is None or not hasattr(head_weight, "ndim"):
        return {}

    return {
        "out_channels": int(head_weight.shape[0]),
        "return_2d_output": bool(head_weight.ndim == 4),
    }


def _merge_encoder_overrides(model_cfg: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return model_cfg
    cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=False))
    existing = cfg_copy.encoder.get("model_overrides")
    existing_dict = (
        OmegaConf.to_container(existing, resolve=False)
        if existing is not None
        else {}
    )
    merged = dict(existing_dict or {})
    merged.update(overrides)
    cfg_copy.encoder.model_overrides = merged
    return cfg_copy


class _DiscoveredAudioDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        audio_paths: list[str],
        sample_rate: int,
        max_num_samples: int,
    ) -> None:
        self._audio_paths = audio_paths
        self._sample_rate = int(sample_rate)
        self._max_num_samples = int(max_num_samples)

    def __len__(self) -> int:
        return len(self._audio_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = Path(self._audio_paths[index]).expanduser().resolve()
        waveform, _ = load_with_torchcodec(str(path), sample_rate=self._sample_rate)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected [T] or [C,T] waveform, got {tuple(waveform.shape)} for {path}"
            )
        waveform = _clip_to_max_length(waveform, self._max_num_samples)
        return {"waveform": waveform, "index": index}


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError(f"Expected batch_size=1, got {len(batch)} items")
    item = batch[0]
    waveforms = item["waveform"].unsqueeze(0).unsqueeze(0)
    indices = torch.tensor([int(item["index"])], dtype=torch.long)
    return {"inputs": waveforms, "indices": indices}


def _build_base_model(
    *,
    weights_source: Path,
    model_config: Path,
) -> tuple[torch.nn.Module, Path, str, list[str], list[str]]:
    state_dict, kind, embedded_model_config, source_info = checkpoint_loading.load_state_dict_from_path(
        weights_source.expanduser().resolve()
    )
    resolved_weights = Path(source_info["weight_path"])

    if embedded_model_config is not None:
        model_cfg = OmegaConf.create(embedded_model_config)
    else:
        model_cfg = checkpoint_loading.load_model_config(model_config.expanduser().resolve())
    inferred_overrides = _infer_redimnet2_encoder_overrides(model_cfg, state_dict)
    model_cfg = _merge_encoder_overrides(model_cfg, inferred_overrides)
    model_cfg.classifier = OmegaConf.create(
        {"_target_": "asv.models.classifier.IdentityClassifier"}
    )
    asv_model = build_model(model_cfg)
    incompatible = asv_model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    blocking_missing = [key for key in missing if not key.startswith("classifier.")]
    blocking_unexpected = [key for key in unexpected if not key.startswith("classifier.")]
    if blocking_missing or blocking_unexpected:
        raise RuntimeError(
            "Checkpoint load failed because non-classifier parameters did not match the model config. "
            f"Missing keys: {blocking_missing}. Unexpected keys: {blocking_unexpected}."
        )

    asv_model.eval()
    for parameter in asv_model.parameters():
        parameter.requires_grad_(False)
    return asv_model.cpu(), resolved_weights, kind, missing, unexpected


@dataclass
class _ModelInstance:
    model: torch.nn.Module
    spec_frontend: torch.nn.Module
    stream: torch.cuda.Stream | None


def _build_model_instances(
    *,
    base_model: torch.nn.Module,
    device: torch.device,
    count: int,
) -> list[_ModelInstance]:
    instances: list[_ModelInstance] = []
    for _ in range(count):
        model = copy.deepcopy(base_model).to(device=device)
        model.eval()
        spec_frontend = clone_frozen_frontend(model.encoder.spec_frontend, device=device)
        stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        instances.append(
            _ModelInstance(
                model=model,
                spec_frontend=spec_frontend,
                stream=stream,
            )
        )
    return instances


@torch.inference_mode()
def _run_instance_batch(
    instance: _ModelInstance,
    inputs_cpu: torch.Tensor,
    indices_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if instance.stream is not None:
        with torch.cuda.stream(instance.stream):
            inputs = inputs_cpu.to(device=device, non_blocking=True)
            spec = compute_spectrogram_batch(inputs, instance.spec_frontend, training=False)
            embeds = instance.model(spec, return_embeddings=True)
            embeds = F.normalize(embeds.detach().float(), dim=1)
            embeds_cpu = embeds.cpu()
        instance.stream.synchronize()
    else:
        inputs = inputs_cpu.to(device=device)
        spec = compute_spectrogram_batch(inputs, instance.spec_frontend, training=False)
        embeds = instance.model(spec, return_embeddings=True)
        embeds_cpu = F.normalize(embeds.detach().float(), dim=1).cpu()
    return indices_cpu.clone(), embeds_cpu


def _store_completed_batch(
    future: Future[tuple[torch.Tensor, torch.Tensor]],
    embeddings_by_index: dict[int, torch.Tensor],
) -> None:
    indices_cpu, embeds_cpu = future.result()
    for row_idx in range(indices_cpu.shape[0]):
        item_index = int(indices_cpu[row_idx].item())
        if item_index not in embeddings_by_index:
            embeddings_by_index[item_index] = embeds_cpu[row_idx]


def _compute_embeddings(
    instances: list[_ModelInstance],
    loader: DataLoader,
    *,
    device: torch.device,
    num_items: int,
) -> torch.Tensor:
    if not instances:
        raise ValueError("at least one model instance is required")

    raw_model = instances[0].model
    embed_dim = int(getattr(raw_model.encoder, "embed_dim"))
    embeddings_by_index: dict[int, torch.Tensor] = {}
    progress = tqdm(loader, desc="embeddings", unit="batch")

    if len(instances) == 1:
        for batch in progress:
            indices_cpu, embeds_cpu = _run_instance_batch(
                instances[0],
                batch["inputs"],
                batch["indices"],
                device,
            )
            for row_idx in range(indices_cpu.shape[0]):
                embeddings_by_index[int(indices_cpu[row_idx].item())] = embeds_cpu[row_idx]
    else:
        pending: set[Future[tuple[torch.Tensor, torch.Tensor]]] = set()
        max_pending = max(2, len(instances) * 2)
        next_instance = 0
        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            for batch in progress:
                future = executor.submit(
                    _run_instance_batch,
                    instances[next_instance],
                    batch["inputs"],
                    batch["indices"],
                    device,
                )
                pending.add(future)
                next_instance = (next_instance + 1) % len(instances)
                while len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for finished in done:
                        _store_completed_batch(finished, embeddings_by_index)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for finished in done:
                    _store_completed_batch(finished, embeddings_by_index)

    missing = [idx for idx in range(num_items) if idx not in embeddings_by_index]
    if missing:
        preview = missing[:10]
        raise RuntimeError(
            f"Missing embeddings for {len(missing)} items: "
            f"{preview}{' ...' if len(missing) > len(preview) else ''}"
        )
    ordered = [embeddings_by_index[idx] for idx in range(num_items)]
    if not ordered:
        return torch.empty((0, embed_dim), dtype=torch.float32)
    return torch.stack(ordered, dim=0)


def _resolve_neighbor_params(
    *,
    num_items: int,
    requested_k: int,
) -> tuple[int, int]:
    max_neighbors = max(0, int(num_items) - 1)
    effective_k = min(max(0, int(requested_k)), max_neighbors)
    if effective_k == 0:
        return 0, 0
    effective_pool_k = min(effective_k + 200, max_neighbors)
    return effective_k, effective_pool_k


def _write_empty_neighbors_csv(output_csv: Path, rel_paths: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filepath", "neighbours"])
        for rel in rel_paths:
            writer.writerow([rel, ""])


def _run_rerank_pipeline(
    *,
    embeddings: torch.Tensor,
    rel_paths: list[str],
    output_csv: Path,
    k: int,
    mean_steps: int,
    n_iter: int,
    w_rank: float,
    w_recip: float,
    w_common: float,
    w_hub: float,
    community_min: int,
    community_max: int,
    device: str,
    knn_chunk_size: int,
    cpu_workers: int,
) -> None:
    effective_k, effective_pool_k = _resolve_neighbor_params(
        num_items=len(rel_paths),
        requested_k=k,
    )
    if effective_k == 0:
        _write_empty_neighbors_csv(output_csv, rel_paths)
        return

    emb = rerank_fast._normalize_embeddings(embeddings.cpu().numpy().astype(np.float32, copy=False))
    print(
        f"[info] rerank params: k={effective_k}, candidate_pool_k={effective_pool_k}, "
        f"mean_steps={mean_steps}, n_iter={n_iter}",
        flush=True,
    )

    base_nbrs, base_scores = rerank_fast._compute_candidate_pool(
        emb,
        pool_k=effective_pool_k,
        device=device,
        chunk_size=int(knn_chunk_size),
    )
    cur = rerank_fast._greedy_mean_chain_rerank(
        emb,
        base_nbrs,
        base_scores,
        out_k=effective_k,
        mean_steps=int(mean_steps),
    )
    rerank_fast._graph_stats(cur, "mean_chain")

    avg_sims = np.zeros(emb.shape[0], dtype=np.float32)
    skip_v2_rerank = effective_k > 501
    if skip_v2_rerank:
        print(
            "[info] k>501: skipping v2 graph-rerank stage and using community post-process only.",
            flush=True,
        )
    else:
        for iteration in range(1, int(n_iter) + 1):
            print(f"iter {iteration}: v2 rerank pass ...", flush=True)
            in_deg = np.bincount(cur.ravel(), minlength=cur.shape[0]).astype(np.int32, copy=False)
            cur = rerank_fast._rerank_pass(
                emb,
                cur,
                in_deg,
                avg_sims=avg_sims,
                w_rank=float(w_rank),
                w_recip=float(w_recip),
                w_common=float(w_common),
                w_hub=float(w_hub),
                w_cosine=0.0,
                w_mutual=0.0,
                w_csls=0.0,
                pool_hop_depth=1,
                cpu_workers=int(cpu_workers),
            )
            rerank_fast._graph_stats(cur, f"iter {iteration}")

    cur = rerank_fast._community_postprocess(
        cur,
        community_min=int(community_min),
        community_max=int(community_max),
    )
    rerank_fast._graph_stats(cur, "final")
    rerank_fast._validate(cur)

    payload = {
        "trial_rel_paths": rel_paths,
        "audio_paths": rel_paths,
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rerank_fast._write_neighbors_csv(payload, cur, output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Neighbor CSV output path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help=(
            "Directory to scan recursively for the latest checkpoint/weight file "
            f"(default: {DEFAULT_WEIGHTS_DIR})."
        ),
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mean-steps", type=int, default=40)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--w-rank", type=float, default=1.0)
    parser.add_argument("--w-recip", type=float, default=1.5)
    parser.add_argument("--w-common", type=float, default=2.0)
    parser.add_argument("--w-hub", type=float, default=0.3)
    parser.add_argument("--community-min", type=int, default=2)
    parser.add_argument("--community-max", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--gpu-model-instances", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    parser.add_argument("--knn-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")
    if args.gpu_model_instances <= 0:
        raise SystemExit("--gpu-model-instances must be >= 1.")
    if args.max_duration_sec <= 0:
        raise SystemExit("--max-duration-sec must be > 0.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, but --device points to CUDA.")

    model_instance_count = int(args.gpu_model_instances)
    if device.type != "cuda" and model_instance_count > 1:
        print(
            "[warn] multiple model instances are only useful on CUDA; falling back to 1 instance.",
            flush=True,
        )
        model_instance_count = 1

    audio_root = args.audio_root.expanduser().resolve()
    audio_paths, _audio_ids, rel_paths = _discover_audio_files(audio_root)

    max_num_samples = int(round(float(args.max_duration_sec) * int(args.sample_rate)))
    dataset = _DiscoveredAudioDataset(
        audio_paths=audio_paths,
        sample_rate=args.sample_rate,
        max_num_samples=max_num_samples,
    )
    loader_kw: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": _collate,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    loader = DataLoader(**loader_kw)

    weights_source = _resolve_weights_source(args.weights_dir)
    base_model, resolved_weights, kind, missing, unexpected = _build_base_model(
        weights_source=weights_source,
        model_config=args.model_config,
    )
    print(f"[info] loaded {kind} weights from {resolved_weights}", flush=True)
    if missing:
        print(
            f"[warn] missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}",
            flush=True,
        )
    if unexpected:
        print(
            f"[warn] unexpected keys ({len(unexpected)}): "
            f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}",
            flush=True,
        )

    print(
        f"[info] discovered {len(audio_paths)} audio files under {audio_root}",
        flush=True,
    )
    print(
        f"[info] building {model_instance_count} model instance(s) on {device} "
        "with per-file variable-length inference",
        flush=True,
    )
    instances = _build_model_instances(
        base_model=base_model,
        device=device,
        count=model_instance_count,
    )

    embeddings = _compute_embeddings(
        instances,
        loader,
        device=device,
        num_items=len(rel_paths),
    )
    del instances
    del base_model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    _run_rerank_pipeline(
        embeddings=embeddings,
        rel_paths=rel_paths,
        output_csv=args.output_csv.expanduser().resolve(),
        k=args.k,
        mean_steps=args.mean_steps,
        n_iter=args.n_iter,
        w_rank=args.w_rank,
        w_recip=args.w_recip,
        w_common=args.w_common,
        w_hub=args.w_hub,
        community_min=args.community_min,
        community_max=args.community_max,
        device=str(device),
        knn_chunk_size=args.knn_chunk_size,
        cpu_workers=int(args.num_workers),
    )
    print(f"Wrote {len(rel_paths)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
