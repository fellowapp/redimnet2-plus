"""Load built-in codec presets from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from asv.codec.preset_schema import CodecPreset, validate_preset

_DEFAULT_PRESETS_FILE = Path(__file__).with_name("presets.yaml")


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    raise ValueError(f"expected list/tuple for parameters, got {type(value).__name__}")


def load_presets(path: Path | None = None) -> dict[str, CodecPreset]:
    """Load presets from YAML and validate them once."""
    src = _DEFAULT_PRESETS_FILE if path is None else path
    data = OmegaConf.to_container(OmegaConf.load(src), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {src}, got {type(data).__name__}")

    out: dict[str, CodecPreset] = {}
    for key, raw in data.items():
        if not isinstance(key, str):
            raise ValueError(f"preset key must be str, got {type(key).__name__}")
        if not isinstance(raw, dict):
            raise ValueError(f"preset {key!r} must be a mapping")
        payload = dict(raw)
        payload["name"] = key
        payload["export_parameters"] = _as_str_tuple(payload.get("export_parameters"))
        payload["decode_parameters"] = _as_str_tuple(payload.get("decode_parameters"))
        preset = CodecPreset(**payload)
        validate_preset(preset)
        out[key] = preset
    return out
