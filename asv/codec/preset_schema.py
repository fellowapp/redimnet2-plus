"""Schema and validation for codec presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CodecPreset:
    """Encode/decode round-trip described in terms of pydub only."""

    name: str
    description: str
    export_format: str
    export_codec: str | None = None
    export_parameters: tuple[str, ...] = ()
    bitrate: str | None = None
    target_frame_rate: int | None = None

    decode_format: str | None = None
    decode_codec: str | None = None
    decode_parameters: tuple[str, ...] = ()
    normalize_frame_rate: int | None = None

    pcm_via_soundfile_wav: bool = False

    def export_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "format": self.export_format,
            "codec": self.export_codec,
            "parameters": list(self.export_parameters) if self.export_parameters else None,
        }
        if self.bitrate is not None:
            kw["bitrate"] = self.bitrate
        if kw["codec"] is None:
            del kw["codec"]
        if kw["parameters"] is None:
            del kw["parameters"]
        return kw

    def decode_from_file_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if self.decode_format is not None:
            kw["format"] = self.decode_format
        if self.decode_codec is not None:
            kw["codec"] = self.decode_codec
        if self.decode_parameters:
            kw["parameters"] = list(self.decode_parameters)
        return kw


def validate_preset(preset: CodecPreset) -> None:
    """Raise ``ValueError`` when a preset is structurally invalid."""
    if not preset.name:
        raise ValueError("preset name must be non-empty")
    if not preset.export_format:
        raise ValueError(f"preset {preset.name!r}: export_format must be non-empty")
    if preset.target_frame_rate is not None and preset.target_frame_rate <= 0:
        raise ValueError(f"preset {preset.name!r}: target_frame_rate must be > 0")
    if preset.normalize_frame_rate is not None and preset.normalize_frame_rate <= 0:
        raise ValueError(f"preset {preset.name!r}: normalize_frame_rate must be > 0")
