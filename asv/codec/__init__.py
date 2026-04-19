"""Pydub-backed codec roundtrip registry and pipelines."""

from asv.codec.availability import (
    check_preset_available,
    clear_availability_cache,
)
from asv.codec.pipeline import CodecPipeline, CodecUnavailableError
from asv.codec.registry import (
    CODEC_REGISTRY,
    CodecPreset,
    as_mapping,
    get_preset,
    list_presets,
    register_preset,
)

__all__ = [
    "CODEC_REGISTRY",
    "CodecPipeline",
    "CodecPreset",
    "CodecUnavailableError",
    "as_mapping",
    "check_preset_available",
    "clear_availability_cache",
    "get_preset",
    "list_presets",
    "register_preset",
]
