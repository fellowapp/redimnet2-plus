"""Load model-only initialization weights for a fresh training stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

_MODEL_FILE_CANDIDATES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model_fsdp.bin",
    "pytorch_model_fsdp_1.bin",
    "model.pt",
)

_INIT_MODE_PREFIXES: dict[str, tuple[str, ...]] = {
    "full": (),
    "encoder_only": ("encoder.",),
    "encoder_bridge": ("encoder.", "bridge."),
}


def _normalize_init_weights_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    token = str(raw).strip()
    if not token or token.lower() in {"none", "null", "~"}:
        return None
    path = Path(token).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path


def _strip_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            while new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _resolve_accelerate_checkpoint_dir(path: Path) -> Path | None:
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


def _load_state_dict_from_path(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        found = next(
            (
                path / name
                for name in _MODEL_FILE_CANDIDATES
                if (path / name).is_file()
            ),
            None,
        )
        if found is None:
            raise FileNotFoundError(
                f"No recognised model file inside {path}. Expected one of: "
                + ", ".join(_MODEL_FILE_CANDIDATES)
            )
        path = found

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        loaded = load_file(str(path))
    else:
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)

    if (
        isinstance(loaded, dict)
        and "state_dict" in loaded
        and isinstance(loaded["state_dict"], dict)
    ):
        state_dict = loaded["state_dict"]
    elif isinstance(loaded, dict):
        state_dict = loaded
    else:
        raise ValueError(f"Unsupported checkpoint payload type at {path}: {type(loaded)}")

    tensor_only = {
        str(key): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    return _strip_prefix(tensor_only)


def _collect_matching_state_dict(
    model: nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
    *,
    init_mode: str,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    if init_mode not in _INIT_MODE_PREFIXES:
        raise ValueError(
            f"Unsupported train.init_mode={init_mode!r}; "
            f"expected one of {sorted(_INIT_MODE_PREFIXES)}"
        )

    prefixes = _INIT_MODE_PREFIXES[init_mode]
    model_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped_by_mode = 0
    skipped_missing = 0
    skipped_shape = 0

    for key, value in checkpoint_state_dict.items():
        if prefixes and not key.startswith(prefixes):
            skipped_by_mode += 1
            continue
        target = model_state.get(key)
        if target is None:
            skipped_missing += 1
            continue
        if target.shape != value.shape:
            skipped_shape += 1
            continue
        matched[key] = value
    return matched, skipped_by_mode, skipped_missing, skipped_shape


def initialize_model_from_weights(
    model: nn.Module,
    init_weights: str | Path | None,
    *,
    init_mode: str = "full",
) -> Path | None:
    raw_path = _normalize_init_weights_path(init_weights)
    if raw_path is None:
        return None
    resolved_path = _resolve_accelerate_checkpoint_dir(raw_path) or raw_path
    checkpoint_state_dict = _load_state_dict_from_path(resolved_path)
    (
        matched,
        skipped_by_mode,
        skipped_missing,
        skipped_shape,
    ) = _collect_matching_state_dict(
        model, checkpoint_state_dict, init_mode=init_mode
    )
    if not matched:
        raise RuntimeError(
            "No compatible weights from "
            f"{resolved_path} matched the current model "
            f"for init_mode={init_mode!r}."
        )
    load_result = model.load_state_dict(matched, strict=False)
    logger.info(
        "Initialized fresh training stage from {}: mode={}, loaded_keys={}, "
        "missing_after_load={}, skipped_by_mode={}, "
        "skipped_missing={}, skipped_shape={}",
        resolved_path,
        init_mode,
        len(matched),
        len(load_result.missing_keys),
        skipped_by_mode,
        skipped_missing,
        skipped_shape,
    )
    return resolved_path
