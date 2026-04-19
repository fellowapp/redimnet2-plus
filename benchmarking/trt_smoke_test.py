#!/usr/bin/env python3
"""CUDA smoke test for a serialized TensorRT engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.benchmark_onnx_speed import load_trt_engine
from benchmarking.export_redimnet2_to_tensorrt import (
    build_sample_waveforms,
    require_cuda_device,
    run_tensorrt_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a TensorRT engine and run a mock inference.")
    parser.add_argument("--engine_path", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="frontend_fp32_model_fp16",
        help="Variant key used with --summary_json.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="With --engine_path: waveform batch (required). With --summary_json: optional; if set, must match export batch_size.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="With --engine_path: waveform length (required). With --summary_json: optional; if set, must match export num_samples.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _infer_dims_from_summary(summary: dict[str, Any], variant_entry: dict[str, Any], variant_key: str) -> tuple[int, int]:
    """Batch/time dims used for the TensorRT build (same as export)."""
    root_bs, root_ns = summary.get("batch_size"), summary.get("num_samples")
    report = variant_entry.get("report")
    ex_bs = ex_ns = None
    if isinstance(report, dict):
        export_meta = report.get("export")
        if isinstance(export_meta, dict):
            ex_bs, ex_ns = export_meta.get("batch_size"), export_meta.get("num_samples")

    if root_bs is not None and root_ns is not None:
        bs, ns = int(root_bs), int(root_ns)
        if ex_bs is not None and int(ex_bs) != bs:
            raise SystemExit(
                f"Summary root batch_size ({bs}) disagrees with variant `{variant_key}` "
                f"report.export.batch_size ({int(ex_bs)})."
            )
        if ex_ns is not None and int(ex_ns) != ns:
            raise SystemExit(
                f"Summary root num_samples ({ns}) disagrees with variant `{variant_key}` "
                f"report.export.num_samples ({int(ex_ns)})."
            )
        return bs, ns

    if not isinstance(report, dict):
        raise SystemExit(
            f"Variant `{variant_key}` has no embedded `report` in the summary; "
            "cannot infer batch_size/num_samples. Re-export or pass --engine_path with explicit dims."
        )
    export_meta = report.get("export")
    if not isinstance(export_meta, dict):
        raise SystemExit(f"Variant `{variant_key}` report has no `export` section.")
    try:
        return int(export_meta["batch_size"]), int(export_meta["num_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Could not read batch_size/num_samples from variant `{variant_key}` export metadata."
        ) from exc


def resolve_run_config(args: argparse.Namespace) -> tuple[Path, int, int]:
    """Return engine path and input shape dims that match the serialized engine."""
    if args.engine_path is not None:
        if args.batch_size is None or args.num_samples is None:
            raise SystemExit("--engine_path requires explicit --batch_size and --num_samples.")
        return args.engine_path.resolve(), args.batch_size, args.num_samples
    if args.summary_json is None:
        raise SystemExit("Provide either --engine_path or --summary_json.")
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    variants = summary.get("variants", {})
    if args.variant not in variants:
        raise SystemExit(f"Variant `{args.variant}` is not present in {args.summary_json}.")
    variant_entry = variants[args.variant]
    engine_path = Path(variant_entry["engine_path"]).resolve()
    batch_size, num_samples = _infer_dims_from_summary(summary, variant_entry, args.variant)
    if args.batch_size is not None and args.batch_size != batch_size:
        raise SystemExit(
            f"--batch_size {args.batch_size} does not match export batch_size {batch_size} in summary."
        )
    if args.num_samples is not None and args.num_samples != num_samples:
        raise SystemExit(
            f"--num_samples {args.num_samples} does not match export num_samples {num_samples} in summary."
        )
    return engine_path, batch_size, num_samples


def main() -> None:
    args = parse_args()
    engine_path, batch_size, num_samples = resolve_run_config(args)
    if not engine_path.is_file():
        raise SystemExit(f"TensorRT engine not found: {engine_path}")

    device = require_cuda_device(args.gpu_id)
    trt, _runtime, engine = load_trt_engine(engine_path)
    waveforms = build_sample_waveforms(batch_size, num_samples, args.seed)
    outputs, output_dtype, input_names, output_names, context = run_tensorrt_inference(
        trt,
        engine,
        waveforms,
        device,
    )
    del context
    if outputs.ndim != 2:
        raise SystemExit(f"Expected [B, D] embeddings, got {tuple(outputs.shape)}")
    if outputs.shape[0] != batch_size:
        raise SystemExit(
            f"Expected batch size {batch_size}, got {outputs.shape[0]} from {engine_path}"
        )
    if not torch.isfinite(outputs).all():
        raise SystemExit("TensorRT smoke test produced non-finite outputs.")

    report: dict[str, Any] = {
        "engine_path": str(engine_path),
        "batch_size": batch_size,
        "num_samples": num_samples,
        "output_shape": list(outputs.shape),
        "output_dtype": output_dtype,
        "input_names": input_names,
        "output_names": output_names,
        "gpu_id": args.gpu_id,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
