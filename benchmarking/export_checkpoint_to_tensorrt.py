#!/usr/bin/env python3
"""Build FP32 and mixed-precision TensorRT engines from a local checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.checkpoint_loading import checkpoint_artifact_stem


@dataclass
class VariantSpec:
    key: str
    export_dtype: str
    engine_precision: str


VARIANTS = (
    VariantSpec(key="fp32", export_dtype="fp32", engine_precision="fp32"),
    VariantSpec(
        key="frontend_fp32_model_fp16",
        export_dtype="fp16",
        engine_precision="fp16",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export two TensorRT variants from a local checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="b6", choices=["b6"])
    parser.add_argument("--model_config", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=6 * 16000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--torch_reference_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--inspect_engine", action="store_true")
    parser.add_argument("--workspace_gib", type=float, default=4.0)
    parser.add_argument(
        "--builder_optimization_level",
        type=int,
        default=5,
        choices=[0, 1, 2, 3, 4, 5],
    )
    parser.add_argument("--max_aux_streams", type=int, default=0)
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Reuse existing ONNX/engine artifacts if present.",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Summary JSON path. Default: <output_dir>/export_summary.json",
    )
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    tag = checkpoint_artifact_stem(args.checkpoint)
    return (REPO_ROOT / "benchmarking" / "outputs" / f"trt_export_{tag}").resolve()


def variant_paths(
    output_dir: Path,
    *,
    model_name: str,
    checkpoint_tag: str,
    variant: VariantSpec,
    batch_size: int,
    num_samples: int,
) -> tuple[Path, Path, Path]:
    stem = f"redimnet2_{model_name}_{checkpoint_tag}_{variant.key}"
    onnx_path = output_dir / f"{stem}.onnx"
    engine_path = output_dir / f"{stem}_bs{batch_size}_s{num_samples}.plan"
    report_path = output_dir / f"{stem}.report.json"
    return onnx_path, engine_path, report_path


def run_export_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    checkpoint_tag = checkpoint_artifact_stem(args.checkpoint)
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_json.resolve() if args.report_json is not None else output_dir / "export_summary.json"

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_artifact_stem": checkpoint_tag,
        "model_name": args.model_name,
        "model_config": str(args.model_config.resolve()) if args.model_config is not None else None,
        "output_dir": str(output_dir),
        "batch_size": int(args.batch_size),
        "num_samples": int(args.num_samples),
        "variants": {},
    }

    exporter_script = REPO_ROOT / "benchmarking" / "export_redimnet2_to_tensorrt.py"
    for variant in VARIANTS:
        onnx_path, engine_path, report_path = variant_paths(
            output_dir,
            model_name=args.model_name,
            checkpoint_tag=checkpoint_tag,
            variant=variant,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
        )
        if args.reuse_existing and engine_path.is_file() and report_path.is_file():
            summary["variants"][variant.key] = {
                "command": None,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "onnx_path": str(onnx_path),
                "engine_path": str(engine_path),
                "report_json": str(report_path),
                "export_dtype": variant.export_dtype,
                "engine_precision": variant.engine_precision,
                "report": load_json(report_path),
                "reused_existing": True,
            }
            continue
        command = [
            sys.executable,
            str(exporter_script),
            "--model_name",
            args.model_name,
            "--checkpoint",
            str(args.checkpoint),
            "--engine_precision",
            variant.engine_precision,
            "--export_dtype",
            variant.export_dtype,
            "--onnx_path",
            str(onnx_path),
            "--engine_path",
            str(engine_path),
            "--report_json",
            str(report_path),
            "--batch_size",
            str(args.batch_size),
            "--num_samples",
            str(args.num_samples),
            "--gpu_id",
            str(args.gpu_id),
            "--seed",
            str(args.seed),
            "--torch_reference_device",
            args.torch_reference_device,
            "--workspace_gib",
            str(args.workspace_gib),
            "--builder_optimization_level",
            str(args.builder_optimization_level),
            "--max_aux_streams",
            str(args.max_aux_streams),
        ]
        if args.model_config is not None:
            command.extend(["--model_config", str(args.model_config)])
        if args.inspect_engine:
            command.append("--inspect_engine")
        if args.reuse_existing and onnx_path.is_file():
            command.append("--reuse_onnx")

        result = run_export_command(command)
        variant_summary: dict[str, Any] = {
            "command": command,
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "onnx_path": str(onnx_path),
            "engine_path": str(engine_path),
            "report_json": str(report_path),
            "export_dtype": variant.export_dtype,
            "engine_precision": variant.engine_precision,
            "reused_existing": False,
        }
        if result.returncode == 0 and report_path.is_file():
            variant_summary["report"] = load_json(report_path)
        summary["variants"][variant.key] = variant_summary
        if result.returncode != 0:
            summary["status"] = "failed"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            raise SystemExit(result.returncode)

    summary["status"] = "ok"
    summary["variant_order"] = [variant.key for variant in VARIANTS]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote export summary to {summary_path}")


if __name__ == "__main__":
    main()
