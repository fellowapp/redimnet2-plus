"""Utilities for resolving local training checkpoints for export/inference."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

_WEIGHT_CANDIDATES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model_fsdp.bin",
    "pytorch_model_fsdp_1.bin",
    "model.pt",
)


def _strip_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip common DDP/FSDP/torch.compile wrappers from state dict keys."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            while new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def resolve_accelerate_checkpoint_dir(path: Path) -> Path | None:
    """Resolve an ``accelerator.save_state()`` directory from training outputs."""
    if not path.is_dir():
        return None
    if (path / "training_state.json").is_file():
        return path.resolve()
    latest = path / "latest.json"
    if not latest.is_file():
        return None
    meta = json.loads(latest.read_text(encoding="utf-8"))
    save_dir = str(meta.get("save_dir", "")).strip()
    if save_dir:
        candidate = path / save_dir
        if candidate.is_dir():
            return candidate.resolve()
    epoch = int(meta["epoch"])
    global_step = int(meta.get("global_step", 0))
    candidate = path / f"epoch_{epoch:04d}_step_{global_step:08d}"
    if candidate.is_dir():
        return candidate.resolve()
    legacy = path / f"epoch_{epoch:04d}"
    if legacy.is_dir():
        return legacy.resolve()
    raise FileNotFoundError(
        f"Could not resolve a checkpoint directory from {latest}. "
        f"Checked {candidate} and {legacy}."
    )


def resolve_weight_path(path: Path) -> Path:
    """Resolve a checkpoint root, checkpoint dir, or explicit weight file."""
    source = path.expanduser().resolve()
    if source.is_file():
        return source

    resolved_dir = resolve_accelerate_checkpoint_dir(source) or source
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {source}")

    for candidate_name in _WEIGHT_CANDIDATES:
        candidate = resolved_dir / candidate_name
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"No recognised weight file inside {resolved_dir}. "
        f"Expected one of: {', '.join(_WEIGHT_CANDIDATES)}."
    )


def describe_checkpoint_source(path: Path) -> dict[str, Any]:
    """Return normalized metadata for a local checkpoint source."""
    requested = path.expanduser().resolve()
    resolved_dir = resolve_accelerate_checkpoint_dir(requested) if requested.is_dir() else None
    weight_path = resolve_weight_path(requested)
    active_dir = resolved_dir or weight_path.parent
    return {
        "requested_path": str(requested),
        "resolved_dir": str(active_dir.resolve()),
        "weight_path": str(weight_path.resolve()),
        "artifact_stem": checkpoint_artifact_stem(active_dir),
    }


def load_state_dict_from_path(
    path: Path,
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any] | None, dict[str, Any]]:
    """Return `(state_dict, kind, model_config, source_info)` for local checkpoints."""
    source_info = describe_checkpoint_source(path)
    weight_path = Path(source_info["weight_path"])
    suffix = weight_path.suffix.lower()
    kind = "asv"
    model_config: dict[str, Any] | None = None
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(weight_path))
    else:
        loaded: Any = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            sd = loaded["state_dict"]
            if "model_config" in loaded:
                kind = "redimnet_release"
                model_config = loaded["model_config"]
        elif isinstance(loaded, dict):
            sd = loaded
        else:
            raise ValueError(
                f"Unsupported checkpoint payload type at {weight_path}: {type(loaded)}"
            )
    return _strip_prefix(sd), kind, model_config, source_info


def _compose_model_config(model_config_path: Path) -> Any:
    """Recursively merge local Hydra-style `defaults` entries for model configs."""
    raw = OmegaConf.load(str(model_config_path))
    merged = OmegaConf.create()
    defaults = raw.get("defaults") if OmegaConf.is_config(raw) else None
    if defaults:
        for item in defaults:
            if item == "_self_":
                continue
            if isinstance(item, str):
                default_name = item
            elif isinstance(item, dict):
                if "_self_" in item:
                    continue
                if len(item) != 1:
                    raise ValueError(
                        f"Unsupported defaults entry in {model_config_path}: {item}"
                    )
                _, value = next(iter(item.items()))
                default_name = str(value)
            else:
                raise ValueError(
                    f"Unsupported defaults entry type in {model_config_path}: {type(item)}"
                )
            default_path = model_config_path.parent / f"{default_name}.yaml"
            if not default_path.is_file():
                raise FileNotFoundError(f"Model default config not found: {default_path}")
            merged = OmegaConf.merge(merged, _compose_model_config(default_path))

    current = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
    if "defaults" in current:
        del current["defaults"]
    return OmegaConf.merge(merged, current)


def load_model_config(model_config_path: Path) -> Any:
    """Load a model yaml and resolve local defaults/interpolations."""
    raw = _compose_model_config(model_config_path)
    wrapped = OmegaConf.create({"model": raw})
    OmegaConf.resolve(wrapped)
    return wrapped.model


def checkpoint_artifact_stem(path: Path) -> str:
    """Sanitize a checkpoint path into a compact artifact stem."""
    requested = path.expanduser().resolve()
    stem = requested.stem if requested.is_file() else requested.name
    compact = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-")
    return compact or "checkpoint"
