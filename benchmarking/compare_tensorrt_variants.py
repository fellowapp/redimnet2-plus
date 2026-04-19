#!/usr/bin/env python3
"""Export and benchmark FP32 vs mixed-precision TensorRT variants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.checkpoint_loading import checkpoint_artifact_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export two TensorRT variants, benchmark them, and compare EER."
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
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--utterance_list", type=Path, required=True)
    parser.add_argument("--audio_root", type=Path, required=True)
    parser.add_argument(
        "--list_format",
        type=str,
        default="voxceleb_trials",
        choices=["voxceleb_trials", "voxceleb_trials_pooled", "plain"],
    )
    parser.add_argument("--extra_voxceleb_o_audio_root", type=Path, default=None)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=REPO_ROOT / "benchmarking" / "outputs" / "speakerid-challenge-baseline-main",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="builtin",
        choices=["builtin", "model_dir"],
    )
    parser.add_argument("--benchmark_batch_size", type=int, default=32)
    parser.add_argument("--benchmark_num_workers", type=int, default=0)
    parser.add_argument("--trt_inflight_batches", type=int, default=2)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunk_seconds", type=float, default=6.0)
    parser.add_argument("--random_chunk", action="store_true", default=False)
    parser.add_argument("--no_dedupe", action="store_true", default=False)
    parser.add_argument("--max_utterances", type=int, default=None)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_num_threads", type=int, default=None)
    parser.add_argument("--eer_tolerance", type=float, default=0.005)
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Final comparison summary path. Default: <output_dir>/comparison_summary.json",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    tag = checkpoint_artifact_stem(args.checkpoint)
    return (REPO_ROOT / "benchmarking" / "outputs" / f"trt_compare_{tag}").resolve()


def extract_eer(report: dict[str, Any]) -> float:
    verification = report.get("verification", {})
    pooled_eer = verification.get("pooled_eer")
    if pooled_eer is not None:
        return float(pooled_eer)
    eer = verification.get("eer")
    if eer is not None:
        return float(eer)
    raise KeyError("Benchmark report does not contain `verification.pooled_eer` or `verification.eer`.")


def extract_timing(report: dict[str, Any]) -> dict[str, float]:
    timings = report.get("timings", {})
    return {
        "wall_time_s": float(timings.get("wall_time_s", 0.0)),
        "compute_time_s": float(timings.get("compute_time_s", 0.0)),
        "utterances_per_s_wall": float(timings.get("utterances_per_s_wall", 0.0)),
        "rtf": float(timings.get("rtf", 0.0)),
    }


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_summary_path = output_dir / "export_summary.json"
    summary_path = args.report_json.resolve() if args.report_json is not None else output_dir / "comparison_summary.json"

    export_command = [
        sys.executable,
        str(REPO_ROOT / "benchmarking" / "export_checkpoint_to_tensorrt.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--model_name",
        args.model_name,
        "--output_dir",
        str(output_dir),
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
        "--report_json",
        str(export_summary_path),
    ]
    if args.model_config is not None:
        export_command.extend(["--model_config", str(args.model_config)])
    if args.inspect_engine:
        export_command.append("--inspect_engine")
    if args.reuse_existing:
        export_command.append("--reuse_existing")

    export_result = run_command(export_command)
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "output_dir": str(output_dir),
        "eer_tolerance": float(args.eer_tolerance),
        "commands": {"export": export_command},
        "export": {
            "returncode": int(export_result.returncode),
            "stdout": export_result.stdout,
            "stderr": export_result.stderr,
            "report_json": str(export_summary_path),
        },
    }
    if export_result.returncode != 0:
        summary["status"] = "export_failed"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        raise SystemExit(export_result.returncode)

    export_summary = load_json(export_summary_path)
    summary["export"]["summary"] = export_summary

    benchmark_script = REPO_ROOT / "benchmarking" / "benchmark_onnx_speed.py"
    benchmark_reports: dict[str, dict[str, Any]] = {}
    benchmark_commands: dict[str, list[str]] = {}
    for variant_key in ("fp32", "frontend_fp32_model_fp16"):
        engine_path = Path(export_summary["variants"][variant_key]["engine_path"])
        benchmark_report_path = output_dir / f"benchmark_{variant_key}.json"
        command = [
            sys.executable,
            str(benchmark_script),
            "--backend",
            "trt",
            "--trt_engine_path",
            str(engine_path),
            "--utterance_list",
            str(args.utterance_list),
            "--audio_root",
            str(args.audio_root),
            "--list_format",
            args.list_format,
            "--model_dir",
            str(args.model_dir),
            "--preprocessor",
            args.preprocessor,
            "--batch_size",
            str(args.benchmark_batch_size),
            "--num_workers",
            str(args.benchmark_num_workers),
            "--trt_inflight_batches",
            str(args.trt_inflight_batches),
            "--sample_rate",
            str(args.sample_rate),
            "--chunk_seconds",
            str(args.chunk_seconds),
            "--warmup_batches",
            str(args.warmup_batches),
            "--device",
            args.device,
            "--report_json",
            str(benchmark_report_path),
        ]
        if args.extra_voxceleb_o_audio_root is not None:
            command.extend(
                ["--extra_voxceleb_o_audio_root", str(args.extra_voxceleb_o_audio_root)]
            )
        if args.random_chunk:
            command.append("--random_chunk")
        if args.no_dedupe:
            command.append("--no_dedupe")
        if args.max_utterances is not None:
            command.extend(["--max_utterances", str(args.max_utterances)])
        if args.torch_num_threads is not None:
            command.extend(["--torch_num_threads", str(args.torch_num_threads)])

        result = run_command(command)
        benchmark_commands[variant_key] = command
        benchmark_summary = {
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "report_json": str(benchmark_report_path),
            "engine_path": str(engine_path),
        }
        if result.returncode != 0:
            summary["status"] = "benchmark_failed"
            summary["benchmarks"] = {variant_key: benchmark_summary}
            summary["commands"]["benchmarks"] = benchmark_commands
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            raise SystemExit(result.returncode)
        benchmark_summary["report"] = load_json(benchmark_report_path)
        benchmark_reports[variant_key] = benchmark_summary

    fp32_report = benchmark_reports["fp32"]["report"]
    mixed_report = benchmark_reports["frontend_fp32_model_fp16"]["report"]
    fp32_eer = extract_eer(fp32_report)
    mixed_eer = extract_eer(mixed_report)
    abs_eer_delta = abs(fp32_eer - mixed_eer)
    gate_passed = abs_eer_delta <= float(args.eer_tolerance)

    summary["status"] = "ok"
    summary["commands"]["benchmarks"] = benchmark_commands
    summary["benchmarks"] = benchmark_reports
    summary["comparison"] = {
        "fp32_eer": fp32_eer,
        "frontend_fp32_model_fp16_eer": mixed_eer,
        "absolute_eer_delta": abs_eer_delta,
        "gate_passed": gate_passed,
        "tolerance": float(args.eer_tolerance),
        "fp32_timings": extract_timing(fp32_report),
        "frontend_fp32_model_fp16_timings": extract_timing(mixed_report),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote comparison summary to {summary_path}")
    raise SystemExit(0 if gate_passed else 2)


if __name__ == "__main__":
    main()
