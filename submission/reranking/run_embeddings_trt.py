"""Инференс эмбеддингов через TensorRT-движок (waveform → embedding, фиксированный shape).

Запечённый ``.plan`` имеет фиксированные ``batch_size`` × ``num_samples`` — они взяты
из ``export_checkpoint_to_tensorrt.py`` / ``ASV_EXPORT_SUMMARY``. Каждый utterance
центр-кроп'ится (или паддится нулями справа) до ``num_samples`` при ``--sample-rate``,
а результат L2-нормализуется — так же, как в ``run_embeddings.py``, чтобы дальше
можно было одним и тем же ре-ранкером пользоваться.

Декодирование аудио идёт через PyTorch ``DataLoader`` (по умолчанию
``--num-workers=4``), так что чтение не блокирует основной поток. Сам TensorRT
всё ещё обрабатывает один батч за раз на GPU."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from asv.data.decode import load_with_torchcodec
from benchmarking.benchmark_onnx_speed import TrtRunner, load_trt_engine
from benchmarking.export_redimnet2_to_tensorrt import require_cuda_device
from benchmarking.trt_smoke_test import resolve_run_config


def _load_run_embeddings_helpers() -> Any:
    return importlib.import_module("submission.reranking.run_embeddings")


def _prepare_waveform(
    waveform_1d: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Center-crop or right-pad to *num_samples*; return shape [1, T] (matches ONNX export)."""
    w = waveform_1d.detach().float().flatten()
    t = int(w.numel())
    if t >= num_samples:
        start = (t - num_samples) // 2
        w = w[start : start + num_samples]
    else:
        pad = torch.zeros(num_samples - t, dtype=w.dtype, device=w.device)
        w = torch.cat([w, pad], dim=0)
    return w.view(1, -1)


class _TrtWaveformDataset(Dataset[dict[str, torch.Tensor]]):
    """Decode one utterance per index (runs in DataLoader worker processes)."""

    def __init__(self, audio_paths: list[str], *, sample_rate: int, num_samples: int) -> None:
        self._paths = audio_paths
        self._sample_rate = int(sample_rate)
        self._num_samples = int(num_samples)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = str(Path(self._paths[index]).expanduser().resolve())
        wf, _ = load_with_torchcodec(path, sample_rate=self._sample_rate)
        if wf.ndim == 2:
            wf = wf.mean(dim=0)
        row = _prepare_waveform(wf, self._num_samples).squeeze(0)
        return {"waveform": row}


def _collate_trt_waveforms(batch: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([item["waveform"] for item in batch], dim=0)


def main() -> None:
    _re = _load_run_embeddings_helpers()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="export_summary.json from TensorRT export (default: ASV_EXPORT_SUMMARY or "
        "<repo>/artifacts/export_summary.json).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant key in summary (default: ASV_DEFAULT_VARIANT or frontend_fp32_model_fp16).",
    )
    parser.add_argument("--engine-path", type=Path, default=None, help="Override engine path.")
    parser.add_argument(
        "--trt-batch-size",
        type=int,
        default=None,
        help="Required with --engine-path (must match how the engine was built).",
    )
    parser.add_argument(
        "--trt-num-samples",
        type=int,
        default=None,
        help="Required with --engine-path (waveform length the engine expects).",
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--voxceleb-root", type=Path, default=None)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument(
        "--skip-eer",
        action="store_true",
        help="Skip VoxCeleb trial EER (required for --audio-root).",
    )
    parser.add_argument(
        "--trial-files",
        type=Path,
        nargs="*",
        default=None,
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=int(os.environ.get("ASV_GPU_ID", "0")),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("ASV_TRT_NUM_WORKERS", "4")),
        help="DataLoader workers for parallel audio decode (0 = main thread only). Default 4 or ASV_TRT_NUM_WORKERS.",
    )
    args = parser.parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0.")

    summary_default = Path(
        os.environ.get("ASV_EXPORT_SUMMARY", str(_REPO_ROOT / "artifacts" / "export_summary.json"))
    )
    summary_path = (args.summary_json or summary_default).expanduser().resolve()
    variant = args.variant or os.environ.get(
        "ASV_DEFAULT_VARIANT", "frontend_fp32_model_fp16"
    )

    if (args.manifest_path is None) == (args.audio_root is None):
        raise SystemExit("Specify exactly one of --manifest-path or --audio-root.")
    if args.manifest_path is not None and args.voxceleb_root is None:
        raise SystemExit("--voxceleb-root is required with --manifest-path.")
    if args.audio_root is not None and args.voxceleb_root is not None:
        raise SystemExit("--voxceleb-root is not used with --audio-root; omit it.")
    if args.audio_root is not None and not args.skip_eer:
        raise SystemExit("Use --skip-eer with --audio-root (trial EER needs manifest layout).")

    manifest_path: Path | None = None
    audio_root: Path | None = None
    voxceleb_root: Path | None = None
    if args.manifest_path is not None:
        manifest_path = args.manifest_path.expanduser().resolve()
        voxceleb_root = args.voxceleb_root.expanduser().resolve()  # type: ignore[union-attr]
        audio_paths, audio_ids, trial_rel_paths = _re._read_manifest_records(
            manifest_path=manifest_path,
            voxceleb_root=voxceleb_root,
        )
        if not audio_paths:
            raise SystemExit(f"No rows found in {manifest_path}")
    else:
        audio_root = args.audio_root.expanduser().resolve()  # type: ignore[union-attr]
        audio_paths, audio_ids, trial_rel_paths = _re._discover_audio_files(audio_root)

    if len(set(trial_rel_paths)) != len(trial_rel_paths):
        raise RuntimeError("Duplicate utterance relative paths / ids in the input set")

    if args.engine_path is not None:
        if args.trt_batch_size is None or args.trt_num_samples is None:
            raise SystemExit(
                "With --engine-path, also pass --trt-batch-size and --trt-num-samples "
                "(must match the TensorRT build)."
            )

    ns = argparse.Namespace(
        engine_path=args.engine_path.resolve() if args.engine_path else None,
        summary_json=summary_path if args.engine_path is None else None,
        variant=variant,
        batch_size=args.trt_batch_size,
        num_samples=args.trt_num_samples,
        seed=0,
        gpu_id=args.gpu_id,
    )
    engine_path, trt_batch, num_samples = resolve_run_config(ns)

    device = require_cuda_device(args.gpu_id)
    trt, _runtime, engine = load_trt_engine(engine_path)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")
    runner = TrtRunner(trt, engine, context, device)

    dataset = _TrtWaveformDataset(
        audio_paths,
        sample_rate=args.sample_rate,
        num_samples=num_samples,
    )
    loader_kw: dict[str, Any] = {
        "batch_size": trt_batch,
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "collate_fn": _collate_trt_waveforms,
        "pin_memory": True,
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kw)

    embed_list: list[torch.Tensor] = []
    for batch_cpu in tqdm(loader, desc="trt_embeddings", unit="batch"):
        bsz = int(batch_cpu.shape[0])
        batch_dev = batch_cpu.to(device, non_blocking=True)
        out_np, _elapsed = runner.run(batch_dev)
        slice_out = torch.from_numpy(out_np[:bsz]).float()
        slice_out = F.normalize(slice_out, dim=1)
        embed_list.append(slice_out.cpu())

    embeddings = torch.cat(embed_list, dim=0)
    assert embeddings.shape[0] == len(audio_paths)

    eers: dict[str, float] = {}
    if not args.skip_eer:
        assert voxceleb_root is not None
        trial_files = (
            [path.expanduser().resolve() for path in args.trial_files]
            if args.trial_files
            else _re._default_trial_files(voxceleb_root)
        )
        eers = _re._compute_trial_eers(
            embeddings=embeddings,
            voxceleb_root=voxceleb_root,
            trial_files=trial_files,
            trial_rel_paths=trial_rel_paths,
        )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": "tensorrt",
        "trt_engine_path": str(engine_path),
        "export_summary": str(summary_path),
        "trt_variant": variant,
        "trt_batch_size": int(trt_batch),
        "trt_num_samples": int(num_samples),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "audio_root": str(audio_root) if audio_root is not None else None,
        "voxceleb_root": str(voxceleb_root) if voxceleb_root is not None else None,
        "sample_rate": int(args.sample_rate),
        "dataloader_num_workers": int(args.num_workers),
        "audio_paths": audio_paths,
        "audio_ids": audio_ids,
        "trial_rel_paths": trial_rel_paths,
        "embeddings": embeddings.numpy(),
        "eers": eers,
    }
    with args.output_pkl.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] wrote TensorRT embeddings to {args.output_pkl}")
    if not args.skip_eer:
        eer_json_path = args.output_pkl.with_suffix(".eers.json")
        eer_json_path.write_text(json.dumps(eers, indent=2), encoding="utf-8")
        print(f"[done] wrote EER summary to {eer_json_path}")


if __name__ == "__main__":
    main()
