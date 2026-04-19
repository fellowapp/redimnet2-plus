#!/usr/bin/env python3
"""Export ReDimNet2 to ONNX and validate against PyTorch.

This script currently targets the pretrained ReDimNet2-B6 speaker encoder and
exports the full waveform -> embedding path:

1. waveform
2. TFMelBanks frontend
3. ReDimNet2 encoder
4. embedding output

Notes:
- ``fp32`` export is fully validated in this repo environment.
- ``fp16`` export is supported and validated with ONNX Runtime CUDA.
- ``bf16`` export keeps the frontend in float32, then casts into bfloat16 right
  before the encoder. It exports successfully, but ONNX Runtime may still reject
  the graph because bfloat16 convolution support is incomplete for this model.
- For ONNX export we switch ReDimNet's feature-weighting blocks to sequential
  accumulation. This avoids a very large ``Concat -> Mul -> ReduceSum``
  intermediate that causes multi-GB CUDA allocations at batch 32.
- Future checkpoint loading is intentionally scaffolded via ``--checkpoint`` but
  not implemented yet.
"""

from __future__ import annotations

import argparse
import copy
import json
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import asv.train.run as train_run_module
from benchmarking.checkpoint_loading import load_model_config, load_state_dict_from_path
from asv.models.redimnet2_vendor.layers.redim_structural import weigth1d


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype={dtype_name}")


def resolve_torch_device(device: str, *, dtype_name: str) -> torch.device:
    if device == "auto":
        if dtype_name == "bf16":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_ort_providers(device: str) -> list[str]:
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    providers: list[str] = []
    if device in {"auto", "cuda"} and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers or ["CPUExecutionProvider"]


def configure_weighting_modules_for_onnx(root: nn.Module) -> None:
    """Avoid huge ``Concat`` temporaries in exported ONNX graphs."""
    for module in root.modules():
        if isinstance(module, weigth1d):
            module.sequential = True


class ExportableReDimNet2(nn.Module):
    """Waveform -> embedding wrapper used for ONNX export."""

    def __init__(self, asv_model: nn.Module, *, dtype_name: str) -> None:
        super().__init__()
        self.dtype_name = dtype_name
        self.compute_dtype = torch_dtype_from_name(dtype_name)

        encoder = asv_model.encoder
        if not hasattr(encoder, "spec_frontend"):
            raise ValueError("Training-built model encoder must expose spec_frontend.")

        # ``spec_frontend`` aliases ``encoder.model.spec``. We detach a float32
        # copy for export and remove the alias from the encoder before casting.
        self.frontend = copy.deepcopy(encoder.spec_frontend).float()
        encoder.spec_frontend = nn.Identity()
        self.model = asv_model.to(dtype=self.compute_dtype)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.frontend(waveform)
        if spec.dtype != self.compute_dtype:
            spec = spec.to(self.compute_dtype)
        embeddings = self.model(spec, return_embeddings=True)
        if embeddings.dtype != self.compute_dtype:
            embeddings = embeddings.to(self.compute_dtype)
        return embeddings


@dataclass
class CompareStats:
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    cosine_similarity_mean: float


@dataclass
class ValidationSummary:
    status: str
    backend: str
    device: str
    providers: list[str]
    message: str | None = None
    output_shape: list[int] | None = None
    output_dtype: str | None = None
    stats: CompareStats | None = None


def compose_training_cfg(model_name: str) -> Any:
    config_dir = REPO_ROOT / "configs"
    model_preset = f"redimnet2_{model_name}_pretrained"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_preset}",
                "train.torch_compile=false",
            ],
        )
    # Only the model subtree is needed for export; resolving the full training
    # config would force unrelated env vars like MUSAN_PATH / RIRS_PATH.
    OmegaConf.resolve(cfg.model)
    return cfg


def default_model_config_path(model_name: str) -> Path:
    return REPO_ROOT / "configs" / "model" / f"redimnet2_{model_name}.yaml"


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

    inferred: dict[str, Any] = {
        "out_channels": int(head_weight.shape[0]),
        "return_2d_output": bool(head_weight.ndim == 4),
    }
    return inferred


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


def _load_checkpoint_model(
    model_name: str,
    checkpoint: Path,
    *,
    model_config_path: Path | None,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    state_dict, checkpoint_kind, embedded_model_config, checkpoint_info = load_state_dict_from_path(
        checkpoint
    )
    resolved_model_config_path = None if model_config_path is None else model_config_path.resolve()
    if embedded_model_config is not None:
        model_cfg = OmegaConf.create(embedded_model_config)
    else:
        config_path = resolved_model_config_path or default_model_config_path(model_name)
        model_cfg = load_model_config(config_path)
        resolved_model_config_path = config_path.resolve()

    inferred_overrides = _infer_redimnet2_encoder_overrides(model_cfg, state_dict)
    model_cfg = _merge_encoder_overrides(model_cfg, inferred_overrides)
    cfg = OmegaConf.create({"model": model_cfg})
    OmegaConf.resolve(cfg.model)
    asv_model = train_run_module.build_model(cfg.model).eval()
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
    configure_weighting_modules_for_onnx(asv_model)
    return asv_model, cfg, {
        "checkpoint": checkpoint_info,
        "checkpoint_kind": checkpoint_kind,
        "model_config_path": (
            str(resolved_model_config_path)
            if resolved_model_config_path is not None
            else None
        ),
        "model_config_source": (
            "embedded_checkpoint_config"
            if embedded_model_config is not None
            else "model_config_path"
        ),
        "inferred_model_overrides": inferred_overrides,
        "ignored_missing_keys": missing,
        "ignored_unexpected_keys": unexpected,
    }


def load_training_model(
    model_name: str,
    checkpoint: Path | None,
    *,
    model_config_path: Path | None = None,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    if checkpoint is not None:
        return _load_checkpoint_model(
            model_name,
            checkpoint,
            model_config_path=model_config_path,
        )
    cfg = compose_training_cfg(model_name)
    # Build the model via the same path used by ``asv/train/run.py``.
    asv_model = train_run_module.build_model(cfg.model).eval()
    configure_weighting_modules_for_onnx(asv_model)
    return asv_model, cfg, {
        "checkpoint": None,
        "checkpoint_kind": "pretrained",
        "model_config_path": None,
        "model_config_source": "pretrained_preset",
        "inferred_model_overrides": {},
        "ignored_missing_keys": [],
        "ignored_unexpected_keys": [],
    }


def build_sample_waveforms(batch_size: int, num_samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(batch_size, num_samples, generator=generator, dtype=torch.float32)


def export_model(
    model: nn.Module,
    output_path: Path,
    export_waveforms: torch.Tensor,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        export_waveforms,
        str(output_path),
        input_names=["waveform"],
        output_names=["embeddings"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "num_samples"},
            "embeddings": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def validate_onnx_graph(output_path: Path) -> dict[str, Any]:
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    tensor_proto = onnx.TensorProto
    initializer_dtypes: dict[str, int] = {}
    for initializer in model.graph.initializer:
        name = tensor_proto.DataType.Name(initializer.data_type)
        initializer_dtypes[name] = initializer_dtypes.get(name, 0) + 1
    return {
        "ir_version": int(model.ir_version),
        "opset_imports": {imp.domain or "ai.onnx": int(imp.version) for imp in model.opset_import},
        "initializer_dtype_counts": initializer_dtypes,
    }


def _to_numpy_float32(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def compare_outputs(torch_out: torch.Tensor | np.ndarray, onnx_out: np.ndarray) -> CompareStats:
    torch_np = _to_numpy_float32(torch_out)
    onnx_np = np.asarray(onnx_out, dtype=np.float32)
    abs_diff = np.abs(torch_np - onnx_np)

    numerator = np.sum(torch_np * onnx_np, axis=1)
    denom = np.linalg.norm(torch_np, axis=1) * np.linalg.norm(onnx_np, axis=1)
    cosine = numerator / np.clip(denom, 1e-12, None)

    return CompareStats(
        max_abs_diff=float(abs_diff.max()),
        mean_abs_diff=float(abs_diff.mean()),
        rmse=float(np.sqrt(np.mean((torch_np - onnx_np) ** 2))),
        cosine_similarity_mean=float(cosine.mean()),
    )


def run_torch_reference(
    model: nn.Module,
    waveforms: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    model = copy.deepcopy(model).eval().to(device)
    x = waveforms.to(device)
    with torch.no_grad():
        y = model(x)
    return y.detach().float().cpu().numpy(), str(y.dtype)


def run_onnx_inference(
    output_path: Path,
    waveforms: torch.Tensor,
    ort_device: str,
    dtype_name: str,
) -> tuple[np.ndarray, ValidationSummary]:
    providers = get_ort_providers(ort_device)
    active_cuda = "CUDAExecutionProvider" in providers

    if ort_device == "cuda" and not active_cuda:
        return np.empty((0, 0), dtype=np.float32), ValidationSummary(
            status="skipped",
            backend="onnxruntime",
            device=ort_device,
            providers=providers,
            message="CUDAExecutionProvider is not available in this environment.",
        )

    if dtype_name == "bf16" and not active_cuda:
        return np.empty((0, 0), dtype=np.float32), ValidationSummary(
            status="skipped",
            backend="onnxruntime",
            device=ort_device,
            providers=providers,
            message=(
                "bf16 ONNX Runtime validation requires CUDAExecutionProvider. "
                "This environment only exposes CPUExecutionProvider."
            ),
        )

    try:
        session = ort.InferenceSession(str(output_path), providers=providers)
        outputs = session.run(None, {"waveform": waveforms.numpy()})
        onnx_out = outputs[0]
        return onnx_out, ValidationSummary(
            status="ok",
            backend="onnxruntime",
            device=ort_device,
            providers=list(session.get_providers()),
            output_shape=list(onnx_out.shape),
            output_dtype=str(onnx_out.dtype),
        )
    except Exception as exc:  # pragma: no cover - environment/provider specific
        return np.empty((0, 0), dtype=np.float32), ValidationSummary(
            status="failed",
            backend="onnxruntime",
            device=ort_device,
            providers=providers,
            message=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ReDimNet2-B6 to ONNX and validate it.")
    parser.add_argument("--model_name", type=str, default="b6", choices=["b6"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="fp32/fp16 are runnable export modes; bf16 is experimental for ONNX Runtime.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ONNX output path. Default: benchmarking/outputs/redimnet2_<model>_<dtype>.onnx",
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--export_batch_size",
        type=int,
        default=1,
        help="Example batch size used only during torch.onnx.export.",
    )
    parser.add_argument("--num_samples", type=int, default=16000 * 6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--torch_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for the PyTorch reference forward pass.",
    )
    parser.add_argument(
        "--ort_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Preferred ONNX Runtime device/provider selection.",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional report path. Default: alongside the ONNX file with .report.json suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = (
            REPO_ROOT
            / "benchmarking"
            / "outputs"
            / f"redimnet2_{args.model_name}_{args.dtype}.onnx"
        )

    report_path = args.report_json or output_path.with_suffix(".report.json")

    asv_model, train_cfg, checkpoint_info = load_training_model(
        args.model_name,
        args.checkpoint,
        model_config_path=args.model_config,
    )
    export_model_wrapper = ExportableReDimNet2(asv_model, dtype_name=args.dtype).eval()
    export_waveforms = build_sample_waveforms(args.export_batch_size, args.num_samples, args.seed)
    sample_waveforms = build_sample_waveforms(args.batch_size, args.num_samples, args.seed)

    export_model(export_model_wrapper, output_path, export_waveforms)
    graph_info = validate_onnx_graph(output_path)

    torch_device = resolve_torch_device(args.torch_device, dtype_name=args.dtype)
    torch_out, torch_out_dtype = run_torch_reference(
        export_model_wrapper, sample_waveforms, torch_device
    )
    torch_summary = ValidationSummary(
        status="ok",
        backend="torch",
        device=str(torch_device),
        providers=[],
        output_shape=list(torch_out.shape),
        output_dtype=torch_out_dtype,
    )

    onnx_out, onnx_summary = run_onnx_inference(
        output_path,
        sample_waveforms,
        args.ort_device,
        args.dtype,
    )
    if onnx_summary.status == "ok":
        onnx_summary.stats = compare_outputs(torch_out, onnx_out)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "export": {
            "model_name": args.model_name,
            "model_config_name": str(train_cfg.model.name),
            "dtype": args.dtype,
            "weighting_mode": "sequential",
            "output": str(output_path.resolve()),
            "checkpoint": checkpoint_info["checkpoint"],
            "checkpoint_kind": checkpoint_info["checkpoint_kind"],
            "model_config_path": checkpoint_info["model_config_path"],
            "model_config_source": checkpoint_info["model_config_source"],
            "inferred_model_overrides": checkpoint_info["inferred_model_overrides"],
            "batch_size": args.batch_size,
            "export_batch_size": args.export_batch_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "onnxruntime_available_providers": ort.get_available_providers(),
        },
        "onnx_graph": graph_info,
        "validation": {
            "torch": asdict(torch_summary),
            "onnxruntime": asdict(onnx_summary),
            "checkpoint_load": {
                "ignored_missing_keys": checkpoint_info["ignored_missing_keys"],
                "ignored_unexpected_keys": checkpoint_info["ignored_unexpected_keys"],
            },
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote ONNX model to {output_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
