#!/usr/bin/env python3
"""Export ReDimNet2-B6 to TensorRT and validate the engine on one GPU.

Default target shape matches the requested deployment setup:
- single GPU
- batch size 32
- 6 * 16000 waveform samples

This script can:
1. export or reuse an ONNX model,
2. build a TensorRT engine for fp32 / fp16 / bf16,
3. run TensorRT inference on CUDA,
4. compare TensorRT outputs against the PyTorch reference,
5. inspect TensorRT layer placement and report whether all layers run on GPU.

Examples
--------
python benchmarking/export_redimnet2_to_tensorrt.py --gpu_id 0
python benchmarking/export_redimnet2_to_tensorrt.py --engine_precision fp16
python benchmarking/export_redimnet2_to_tensorrt.py --engine_precision bf16 --export_dtype fp32
"""

from __future__ import annotations

import argparse
import gc
import json
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.onnx_conversion.export_redimnet2_to_onnx import (
    ExportableReDimNet2,
    build_sample_waveforms,
    compare_outputs,
    default_model_config_path,
    load_training_model,
    run_torch_reference,
    validate_onnx_graph,
)
from benchmarking.checkpoint_loading import checkpoint_artifact_stem


def import_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - environment specific
        raise SystemExit(
            "TensorRT Python bindings are not installed. "
            "Install `tensorrt` in the active environment or run with `trtexec` available."
        ) from exc
    return trt


def normalize_precision(name: str) -> str:
    lowered = name.lower()
    if lowered == "fp14":
        return "fp16"
    if lowered not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision={name}")
    return lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ReDimNet2-B6 to TensorRT, validate it, and inspect GPU layer placement."
    )
    parser.add_argument("--model_name", type=str, default="b6", choices=["b6"])
    parser.add_argument(
        "--engine_precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "fp14", "bf16"],
        help="TensorRT build precision. `fp14` is accepted as an alias for `fp16`.",
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="ONNX export dtype. Keeping this at fp32 is usually the most robust.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=6 * 16000)
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device index to use.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--torch_reference_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for the PyTorch parity reference. Default `cpu` avoids stealing GPU memory from TensorRT runtime validation.",
    )
    parser.add_argument(
        "--inspect_engine",
        action="store_true",
        help="Collect TensorRT inspector layer names. Disabled by default because it can trigger extra compilation work.",
    )
    parser.add_argument(
        "--workspace_gib",
        type=float,
        default=4.0,
        help="TensorRT workspace memory pool limit in GiB.",
    )
    parser.add_argument(
        "--builder_optimization_level",
        type=int,
        default=5,
        choices=[0, 1, 2, 3, 4, 5],
        help="TensorRT builder optimization level when supported.",
    )
    parser.add_argument(
        "--max_aux_streams",
        type=int,
        default=0,
        help="TensorRT max auxiliary streams. Default 0 keeps execution single-stream and reduces memory pressure.",
    )
    parser.add_argument(
        "--onnx_path",
        type=Path,
        default=None,
        help="Optional ONNX path. Default: benchmarking/outputs/redimnet2_b6_<export_dtype>.onnx",
    )
    parser.add_argument(
        "--engine_path",
        type=Path,
        default=None,
        help="Optional TensorRT engine path. Default: benchmarking/outputs/redimnet2_b6_<precision>_bs32_s96000.plan",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional report JSON path. Default: alongside the engine with `.report.json` suffix.",
    )
    parser.add_argument(
        "--reuse_onnx",
        action="store_true",
        help="Reuse an existing ONNX file instead of exporting it again.",
    )
    parser.add_argument(
        "--dynamic_onnx",
        action="store_true",
        help="Export ONNX with dynamic batch/time axes. Disabled by default for fixed 32 x 96000 TensorRT builds.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint root, checkpoint dir, or explicit weight file to export.",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        default=None,
        help="Model YAML used when loading local checkpoints. Default: configs/model/redimnet2_<model>.yaml",
    )
    return parser.parse_args()


def require_cuda_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for TensorRT export and validation, but no GPU is available.")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise SystemExit(
            f"--gpu_id {gpu_id} is invalid; available CUDA devices: 0..{torch.cuda.device_count() - 1}"
        )
    torch.cuda.set_device(gpu_id)
    return torch.device("cuda", gpu_id)


def _supports_fp16(builder: Any) -> bool:
    return bool(
        getattr(builder, "platform_has_fp16", False)
        or getattr(builder, "platform_has_fast_fp16", False)
    )


def _supports_bf16(device: torch.device, trt: Any) -> bool:
    if not hasattr(trt.BuilderFlag, "BF16"):
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 8


def resolve_paths(args: argparse.Namespace, engine_precision: str) -> tuple[Path, Path, Path]:
    checkpoint_tag = checkpoint_artifact_stem(args.checkpoint) if args.checkpoint is not None else None
    onnx_name_parts = [f"redimnet2_{args.model_name}"]
    if checkpoint_tag is not None:
        onnx_name_parts.append(checkpoint_tag)
    onnx_name_parts.append(args.export_dtype)
    if args.onnx_path is None:
        onnx_path = (
            REPO_ROOT
            / "benchmarking"
            / "outputs"
            / f"{'_'.join(onnx_name_parts)}.onnx"
        )
    else:
        onnx_path = args.onnx_path

    if args.engine_path is None:
        engine_name_parts = [f"redimnet2_{args.model_name}_{engine_precision}"]
        if checkpoint_tag is not None:
            engine_name_parts.append(checkpoint_tag)
        if engine_precision != "fp32":
            engine_name_parts.append("frontendfp32")
        engine_stem = "_".join(engine_name_parts)
        engine_path = (
            REPO_ROOT
            / "benchmarking"
            / "outputs"
            / f"{engine_stem}_bs{args.batch_size}_s{args.num_samples}.plan"
        )
    else:
        engine_path = args.engine_path

    report_path = args.report_json or engine_path.with_suffix(".report.json")
    return onnx_path, engine_path, report_path


def maybe_export_onnx(
    model_wrapper: ExportableReDimNet2,
    args: argparse.Namespace,
    onnx_path: Path,
) -> dict[str, Any]:
    if args.reuse_onnx:
        if not onnx_path.is_file():
            raise SystemExit(f"--reuse_onnx was set but ONNX file does not exist: {onnx_path}")
    else:
        export_waveforms = build_sample_waveforms(1, args.num_samples, args.seed)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        dynamic_axes = None
        if args.dynamic_onnx:
            dynamic_axes = {
                "waveform": {0: "batch", 1: "num_samples"},
                "embeddings": {0: "batch"},
            }
        torch.onnx.export(
            model_wrapper.eval(),
            export_waveforms,
            str(onnx_path),
            input_names=["waveform"],
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        if not args.dynamic_onnx:
            specialize_onnx_io_shapes(onnx_path, batch_size=args.batch_size, num_samples=args.num_samples)
    return validate_onnx_graph(onnx_path)


def constrain_frontend_precision(
    trt: Any,
    network: Any,
    *,
    engine_precision: str,
) -> dict[str, Any]:
    if engine_precision == "fp32":
        return {
            "enabled": False,
            "frontend_layer_count": 0,
            "frontend_layers_forced_fp32": 0,
            "frontend_outputs_forced_fp32": 0,
            "boundary_cast_name": None,
            "boundary_cast_output_type": None,
            "precision_flags": [],
        }

    frontend_prefix = "/frontend/"
    boundary_cast_name = "/Cast"
    frontend_layer_count = 0
    frontend_layers_forced_fp32 = 0
    frontend_outputs_forced_fp32 = 0
    boundary_cast_output_type: str | None = None

    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        layer_name = str(getattr(layer, "name", "") or "")

        if layer_name.startswith(frontend_prefix):
            frontend_layer_count += 1
            float_output_indices: list[int] = []
            for output_idx in range(int(getattr(layer, "num_outputs", 0))):
                output = layer.get_output(output_idx)
                if output is None:
                    continue
                if str(output.dtype) == str(trt.DataType.FLOAT):
                    float_output_indices.append(output_idx)

            if float_output_indices and hasattr(layer, "precision_is_set"):
                layer.precision = trt.DataType.FLOAT
                frontend_layers_forced_fp32 += 1
            for output_idx in float_output_indices:
                layer.set_output_type(output_idx, trt.DataType.FLOAT)
                frontend_outputs_forced_fp32 += 1
            continue

        if layer_name == boundary_cast_name:
            target_dtype = trt.DataType.HALF
            if engine_precision == "bf16":
                if not hasattr(trt.DataType, "BF16"):
                    raise RuntimeError("TensorRT BF16 build requested, but trt.DataType.BF16 is unavailable.")
                target_dtype = trt.DataType.BF16
            if hasattr(layer, "precision_is_set"):
                layer.precision = trt.DataType.FLOAT
            if int(getattr(layer, "num_outputs", 0)) >= 1:
                layer.set_output_type(0, target_dtype)
                boundary_cast_output_type = str(target_dtype)

    precision_flags: list[str] = []
    if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
        precision_flags.append("OBEY_PRECISION_CONSTRAINTS")
    elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
        precision_flags.append("PREFER_PRECISION_CONSTRAINTS")

    return {
        "enabled": True,
        "frontend_layer_count": frontend_layer_count,
        "frontend_layers_forced_fp32": frontend_layers_forced_fp32,
        "frontend_outputs_forced_fp32": frontend_outputs_forced_fp32,
        "boundary_cast_name": boundary_cast_name,
        "boundary_cast_output_type": boundary_cast_output_type,
        "precision_flags": precision_flags,
    }


def specialize_onnx_io_shapes(onnx_path: Path, *, batch_size: int, num_samples: int) -> None:
    model = onnx.load(str(onnx_path))
    if len(model.graph.input) != 1:
        raise RuntimeError(f"Expected exactly 1 ONNX input, got {len(model.graph.input)}")
    if len(model.graph.output) != 1:
        raise RuntimeError(f"Expected exactly 1 ONNX output, got {len(model.graph.output)}")

    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(input_shape) != 2:
        raise RuntimeError(f"Expected waveform input rank 2, got rank {len(input_shape)}")
    input_shape[0].dim_value = int(batch_size)
    input_shape[0].ClearField("dim_param")
    input_shape[1].dim_value = int(num_samples)
    input_shape[1].ClearField("dim_param")

    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    if len(output_shape) >= 1:
        output_shape[0].dim_value = int(batch_size)
        output_shape[0].ClearField("dim_param")

    onnx.save(model, str(onnx_path))


def build_engine(
    trt: Any,
    *,
    onnx_path: Path,
    engine_path: Path,
    input_shape: tuple[int, int],
    precision: str,
    device: torch.device,
    workspace_gib: float,
    builder_optimization_level: int,
    max_aux_streams: int,
) -> tuple[Any, dict[str, Any]]:
    logger = trt.Logger(trt.Logger.INFO)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    model_bytes = onnx_path.read_bytes()
    if not parser.parse(model_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(
            "TensorRT failed to parse the ONNX model:\n" + "\n".join(errors)
        )

    if network.num_inputs != 1:
        raise RuntimeError(f"Expected exactly 1 network input, got {network.num_inputs}")

    input_tensor = network.get_input(0)
    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gib * (1024**3))
    if workspace_bytes > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(builder_optimization_level)
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = int(max_aux_streams)

    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    enabled_flags = ["TF32_DISABLED"]
    if precision == "fp16":
        if not _supports_fp16(builder):
            raise RuntimeError("This GPU / TensorRT build does not report FP16 support.")
        config.set_flag(trt.BuilderFlag.FP16)
        enabled_flags.append("FP16")
    elif precision == "bf16":
        if not _supports_bf16(device, trt):
            raise RuntimeError(
                "BF16 requires TensorRT BF16 support and an Ampere-or-newer GPU."
            )
        config.set_flag(trt.BuilderFlag.BF16)
        enabled_flags.append("BF16")
    else:
        enabled_flags.append("FP32")

    frontend_constraints = constrain_frontend_precision(
        trt,
        network,
        engine_precision=precision,
    )
    for flag_name in frontend_constraints["precision_flags"]:
        config.set_flag(getattr(trt.BuilderFlag, flag_name))
        enabled_flags.append(flag_name)

    if any(int(dim) < 0 for dim in input_tensor.shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT returned an empty serialized engine.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_bytes = bytes(serialized_engine)
    engine_path.write_bytes(engine_bytes)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize the serialized TensorRT engine.")

    build_summary = {
        "input_name": input_tensor.name,
        "network_input_shape": [int(v) for v in input_tensor.shape],
        "requested_input_shape": list(input_shape),
        "workspace_bytes": workspace_bytes,
        "builder_optimization_level": int(builder_optimization_level),
        "max_aux_streams": int(max_aux_streams),
        "enabled_flags": enabled_flags,
        "num_dla_cores": int(getattr(builder, "num_DLA_cores", 0)),
        "frontend_precision_constraints": frontend_constraints,
    }
    return engine, build_summary


def torch_dtype_from_trt_dtype(trt: Any, dtype: Any) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
    }
    if hasattr(trt.DataType, "BF16"):
        mapping[trt.DataType.BF16] = torch.bfloat16
    if dtype not in mapping:
        raise RuntimeError(f"Unsupported TensorRT tensor dtype: {dtype}")
    return mapping[dtype]


def run_tensorrt_inference(
    trt: Any,
    engine: Any,
    waveforms: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, str, list[str], list[str], Any]:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    input_names: list[str] = []
    output_names: list[str] = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly 1 TensorRT input tensor, got {input_names}")
    if not output_names:
        raise RuntimeError("TensorRT engine produced no output tensors.")

    input_name = input_names[0]
    if any(int(v) < 0 for v in engine.get_tensor_shape(input_name)):
        context.set_input_shape(input_name, tuple(int(v) for v in waveforms.shape))

    input_dtype = torch_dtype_from_trt_dtype(trt, engine.get_tensor_dtype(input_name))
    input_gpu = waveforms.to(device=device, dtype=input_dtype, non_blocking=False).contiguous()

    tensor_buffers: dict[str, torch.Tensor] = {input_name: input_gpu}
    for output_name in output_names:
        output_shape = tuple(int(v) for v in context.get_tensor_shape(output_name))
        output_dtype = torch_dtype_from_trt_dtype(trt, engine.get_tensor_dtype(output_name))
        tensor_buffers[output_name] = torch.empty(output_shape, device=device, dtype=output_dtype)

    for name, tensor in tensor_buffers.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))

    stream = torch.cuda.current_stream(device=device)
    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT execution failed.")
    stream.synchronize()

    output_name = "embeddings" if "embeddings" in output_names else output_names[0]
    output_tensor = tensor_buffers[output_name]
    return output_tensor.detach().cpu(), str(output_tensor.dtype), input_names, output_names, context


def _cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inspect_engine_layers(trt: Any, engine: Any, *, num_dla_cores: int) -> dict[str, Any]:
    inspector = engine.create_engine_inspector()
    if inspector is None:
        return {
            "status": "unavailable",
            "message": "TensorRT engine inspector is not available.",
            "all_layers_on_gpu": None,
            "layers": [],
            "non_gpu_layers": [],
        }

    try:
        raw_engine_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        engine_info = json.loads(raw_engine_info)
        layer_names = engine_info.get("Layers", [])
    except json.JSONDecodeError:
        raw_engine_info = inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
        engine_info = {"raw": raw_engine_info}
        lines = [line.strip() for line in raw_engine_info.splitlines() if line.strip()]
        layer_names = [line for line in lines if line != "Layers:"]

    layers = [{"index": idx, "name": name} for idx, name in enumerate(layer_names)]
    all_layers_on_gpu = num_dla_cores == 0
    status = "ok" if all_layers_on_gpu else "unknown"
    message = (
        "This platform reports 0 DLA cores; after successful TensorRT execution the engine is running on GPU."
        if all_layers_on_gpu
        else "Per-layer device placement was not exposed directly by the inspector."
    )

    return {
        "status": status,
        "message": message,
        "all_layers_on_gpu": all_layers_on_gpu,
        "devices_seen": ["GPU"] if all_layers_on_gpu else [],
        "layers": layers,
        "non_gpu_layers": [],
        "engine_information": engine_info,
    }


def summarize_gpu_execution_without_inspector(engine: Any, *, num_dla_cores: int) -> dict[str, Any]:
    all_layers_on_gpu = num_dla_cores == 0
    return {
        "status": "ok" if all_layers_on_gpu else "unknown",
        "message": (
            "Successful TensorRT execution on a platform with 0 DLA cores implies GPU execution."
            if all_layers_on_gpu
            else "Skipped TensorRT inspector; GPU-only execution could not be proven from hardware capabilities alone."
        ),
        "all_layers_on_gpu": all_layers_on_gpu,
        "devices_seen": ["GPU"] if all_layers_on_gpu else [],
        "layers": [{"index": idx} for idx in range(int(getattr(engine, "num_layers", 0)))],
        "non_gpu_layers": [],
        "engine_information": None,
    }


@dataclass
class ExportSummary:
    model_name: str
    export_dtype: str
    engine_precision_requested: str
    engine_precision_normalized: str
    checkpoint_path: str | None
    checkpoint_artifact_stem: str | None
    model_config_path: str | None
    onnx_path: str
    engine_path: str
    batch_size: int
    num_samples: int
    gpu_id: int
    seed: int


def main() -> None:
    args = parse_args()
    engine_precision = normalize_precision(args.engine_precision)
    onnx_path, engine_path, report_path = resolve_paths(args, engine_precision)
    device = require_cuda_device(args.gpu_id)
    torch.cuda.synchronize(device)

    resolved_model_config_path = (
        args.model_config.resolve()
        if args.model_config is not None
        else (default_model_config_path(args.model_name).resolve() if args.checkpoint is not None else None)
    )
    asv_model, train_cfg, checkpoint_info = load_training_model(
        args.model_name,
        args.checkpoint,
        model_config_path=args.model_config,
    )
    export_wrapper = ExportableReDimNet2(asv_model, dtype_name=args.export_dtype).eval()

    onnx_graph = maybe_export_onnx(export_wrapper, args, onnx_path)
    trt = import_tensorrt()
    engine, build_summary = build_engine(
        trt,
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_shape=(args.batch_size, args.num_samples),
        precision=engine_precision,
        device=device,
        workspace_gib=args.workspace_gib,
        builder_optimization_level=args.builder_optimization_level,
        max_aux_streams=args.max_aux_streams,
    )

    sample_waveforms = build_sample_waveforms(args.batch_size, args.num_samples, args.seed)
    trt_out_tensor, trt_out_dtype, input_names, output_names, context = run_tensorrt_inference(
        trt,
        engine,
        sample_waveforms,
        device,
    )
    if args.inspect_engine:
        layer_report = inspect_engine_layers(
            trt,
            engine,
            num_dla_cores=int(build_summary["num_dla_cores"]),
        )
    else:
        layer_report = summarize_gpu_execution_without_inspector(
            engine,
            num_dla_cores=int(build_summary["num_dla_cores"]),
        )

    del context
    del engine
    _cleanup_cuda_memory()

    reference_device = (
        torch.device("cuda", args.gpu_id)
        if args.torch_reference_device == "cuda"
        else torch.device("cpu")
    )
    torch_out, torch_out_dtype = run_torch_reference(export_wrapper, sample_waveforms, reference_device)
    compare_stats = asdict(compare_outputs(torch_out, trt_out_tensor))

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "export": asdict(
            ExportSummary(
                model_name=args.model_name,
                export_dtype=args.export_dtype,
                engine_precision_requested=args.engine_precision,
                engine_precision_normalized=engine_precision,
                checkpoint_path=(
                    checkpoint_info["checkpoint"]["requested_path"]
                    if checkpoint_info["checkpoint"] is not None
                    else None
                ),
                checkpoint_artifact_stem=(
                    checkpoint_info["checkpoint"]["artifact_stem"]
                    if checkpoint_info["checkpoint"] is not None
                    else None
                ),
                model_config_path=(
                    checkpoint_info["model_config_path"] or str(resolved_model_config_path)
                    if resolved_model_config_path is not None or checkpoint_info["model_config_path"] is not None
                    else None
                ),
                onnx_path=str(onnx_path.resolve()),
                engine_path=str(engine_path.resolve()),
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                gpu_id=args.gpu_id,
                seed=args.seed,
            )
        ),
        "config": {
            "train_model_name": str(train_cfg.model.name),
            "workspace_gib": args.workspace_gib,
            "builder_optimization_level": args.builder_optimization_level,
            "max_aux_streams": args.max_aux_streams,
            "reuse_onnx": bool(args.reuse_onnx),
            "dynamic_onnx": bool(args.dynamic_onnx),
            "checkpoint_kind": checkpoint_info["checkpoint_kind"],
            "checkpoint": checkpoint_info["checkpoint"],
            "model_config_source": checkpoint_info["model_config_source"],
            "inferred_model_overrides": checkpoint_info["inferred_model_overrides"],
        },
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            "tensorrt_version": getattr(trt, "__version__", "unknown"),
        },
        "onnx_graph": onnx_graph,
        "tensorrt_build": build_summary,
        "validation": {
            "torch_reference_device": str(reference_device),
            "torch_output_dtype": torch_out_dtype,
            "tensorrt_output_dtype": trt_out_dtype,
            "input_names": input_names,
            "output_names": output_names,
            "compare_stats": compare_stats,
            "checkpoint_load": {
                "ignored_missing_keys": checkpoint_info["ignored_missing_keys"],
                "ignored_unexpected_keys": checkpoint_info["ignored_unexpected_keys"],
            },
        },
        "layer_placement": layer_report,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote TensorRT engine to {engine_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
