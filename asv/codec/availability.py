"""Whether a preset works, using pydub encode/decode probes only (cached)."""

from __future__ import annotations

import threading

from asv.codec.registry import CodecPreset

__all__ = [
    "clear_availability_cache",
    "check_preset_available",
]

class PresetAvailabilityCache:
    """Thread-safe cache around preset availability probes."""

    def __init__(self) -> None:
        self._availability_cache: dict[str, bool] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._availability_cache.clear()

    def check(self, preset: CodecPreset | str, *, use_cache: bool = True) -> bool:
        from asv.codec.registry import get_preset

        p = get_preset(preset) if isinstance(preset, str) else preset
        key = p.name
        with self._lock:
            if use_cache and key in self._availability_cache:
                return self._availability_cache[key]

        ok = _roundtrip_probe(p)
        with self._lock:
            self._availability_cache[key] = ok
        return ok


_CACHE = PresetAvailabilityCache()


def clear_availability_cache() -> None:
    """Clear cached results for :func:`check_preset_available`."""
    _CACHE.clear()


def check_preset_available(
    preset: CodecPreset | str,
    *,
    use_cache: bool = True,
) -> bool:
    """Return whether pydub can encode and decode a short synthetic clip for this preset."""
    return _CACHE.check(preset, use_cache=use_cache)


def _roundtrip_probe(preset: CodecPreset) -> bool:
    from asv.codec.pipeline import CodecPipeline

    try:
        pipe = CodecPipeline(preset, check_available=False)
    except Exception:
        return False
    try:
        import numpy as np

        sr = 16000
        t = int(0.12 * sr)
        x = np.zeros((1, t), dtype=np.float32)
        x[0, :800] = 0.01 * np.sin(2 * np.pi * 440 * np.arange(800, dtype=np.float32) / sr)
        y, out_sr = pipe.roundtrip(x, sr)
        if y.size == 0 or not np.isfinite(y).all():
            return False
        if out_sr <= 0:
            return False
        return True
    except Exception:
        return False
