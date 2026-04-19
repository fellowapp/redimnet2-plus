"""Thin registry API for codec presets."""

from __future__ import annotations

from typing import Mapping

from asv.codec.preset_loader import load_presets
from asv.codec.preset_schema import CodecPreset, validate_preset

__all__ = [
    "CodecPreset",
    "CODEC_REGISTRY",
    "get_preset",
    "list_presets",
    "register_preset",
]


CODEC_REGISTRY: dict[str, CodecPreset] = load_presets()


def get_preset(name: str) -> CodecPreset:
    """Return a preset by logical name."""
    try:
        return CODEC_REGISTRY[name]
    except KeyError as e:
        known = ", ".join(sorted(CODEC_REGISTRY))
        raise KeyError(f"Unknown codec preset {name!r}. Known: {known}") from e


def list_presets() -> list[str]:
    """Sorted logical preset names."""
    return sorted(CODEC_REGISTRY)


def register_preset(preset: CodecPreset, *, overwrite: bool = False) -> None:
    """Register a custom preset at runtime (e.g. from Hydra or tests)."""
    validate_preset(preset)
    if not overwrite and preset.name in CODEC_REGISTRY:
        raise ValueError(f"preset {preset.name!r} is already registered")
    CODEC_REGISTRY[preset.name] = preset


def as_mapping() -> Mapping[str, CodecPreset]:
    """Read-only view of the registry (for config/introspection)."""
    return CODEC_REGISTRY
