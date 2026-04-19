"""Round-trip behaviour for :class:`asv.codec.CodecPipeline`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asv.codec import CodecPipeline, check_preset_available, clear_availability_cache
from asv.codec.pipeline import CodecUnavailableError
from asv.codec.registry import CODEC_REGISTRY, CodecPreset, register_preset


def setup_function() -> None:
    clear_availability_cache()


def _sine(sr: int, dur: float, freq: float = 440.0) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / sr
    w = 0.1 * np.sin(2 * np.pi * freq * t)
    return w[np.newaxis, :]


@pytest.mark.parametrize("name", ["mp3_16k", "pcm_alaw_8k", "pcm_mulaw_8k"])
def test_roundtrip_smoke_if_available(name: str) -> None:
    if not check_preset_available(name):
        pytest.skip(f"{name} not available with pydub on this system")
    pipe = CodecPipeline(name, check_available=True)
    x = _sine(16000, 0.15)
    y, sr = pipe.roundtrip(x, 16000)
    assert y.shape[0] == 1
    assert y.size > 0
    assert sr > 0
    assert np.isfinite(y).all()


def test_check_available_raises_when_missing() -> None:
    clear_availability_cache()
    fake = CodecPreset(
        name="_nonexistent_encoder_xyz",
        description="test",
        export_format="mp3",
        export_codec="libNOTREALCODEC12345",
        export_parameters=(),
        decode_format="mp3",
    )
    register_preset(fake, overwrite=False)
    try:
        assert check_preset_available(fake, use_cache=False) is False
        with pytest.raises(CodecUnavailableError):
            CodecPipeline(fake.name, check_available=True)
    finally:
        CODEC_REGISTRY.pop(fake.name, None)


def test_forward_torch_matches_numpy() -> None:
    name = "mp3_16k"
    if not check_preset_available(name):
        pytest.skip("mp3_16k not available")
    pipe = CodecPipeline(name)
    x = torch.tensor(_sine(16000, 0.12))
    y_t, sr = pipe.forward(x, 16000)
    y_np, sr2 = pipe.roundtrip(x.numpy(), 16000)
    assert sr == sr2
    assert y_t.shape == torch.Size(y_np.shape)
    assert torch.allclose(y_t, torch.from_numpy(y_np), atol=1e-6)


def test_encode_failure_raises() -> None:
    bad = CodecPreset(
        name="_bad_export_test",
        description="bad",
        export_format="mp3",
        export_codec="libNOTREALCODEC12345",
        export_parameters=(),
        decode_format="mp3",
    )
    register_preset(bad, overwrite=False)
    try:
        pipe = CodecPipeline(bad, check_available=False)
        with pytest.raises(CodecUnavailableError):
            pipe.roundtrip(_sine(16000, 0.05), 16000)
    finally:
        CODEC_REGISTRY.pop(bad.name, None)


def test_opus_roundtrip_if_available() -> None:
    if not check_preset_available("opus_16k"):
        pytest.skip("opus_16k not available")
    pipe = CodecPipeline("opus_16k")
    y, sr = pipe.roundtrip(_sine(16000, 0.12), 16000)
    assert sr == 16000
    assert y.shape[0] == 1
    assert np.isfinite(y).all()


def test_soundfile_path_matches_segment_path() -> None:
    if not check_preset_available("mp3_16k"):
        pytest.skip("mp3_16k not available")
    base = CodecPreset(
        name="_sf_cmp",
        description="tmp",
        export_format="mp3",
        bitrate="64k",
        target_frame_rate=16000,
        decode_format="mp3",
        normalize_frame_rate=16000,
        pcm_via_soundfile_wav=False,
    )
    sf = CodecPreset(
        name="_sf_cmp_sf",
        description="tmp",
        export_format="mp3",
        bitrate="64k",
        target_frame_rate=16000,
        decode_format="mp3",
        normalize_frame_rate=16000,
        pcm_via_soundfile_wav=True,
    )
    register_preset(base, overwrite=False)
    register_preset(sf, overwrite=False)
    try:
        x = _sine(16000, 0.1)
        a, sra = CodecPipeline(base, check_available=False).roundtrip(x, 16000)
        b, srb = CodecPipeline(sf, check_available=False).roundtrip(x, 16000)
        assert sra == srb == 16000
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-3)
    finally:
        CODEC_REGISTRY.pop(base.name, None)
        CODEC_REGISTRY.pop(sf.name, None)


def test_auto_selects_ffmpeg_for_mp3() -> None:
    pipe = CodecPipeline("mp3_16k")
    assert pipe.selected_backend() == "ffmpeg"


def test_auto_falls_back_to_pydub_for_mp4_container() -> None:
    preset = CodecPreset(
        name="_mp4_auto_backend",
        description="tmp",
        export_format="mp4",
        export_codec="aac",
        bitrate="64k",
        target_frame_rate=16000,
        decode_format="mp4",
        decode_codec="aac",
        normalize_frame_rate=16000,
    )
    register_preset(preset, overwrite=False)
    try:
        pipe = CodecPipeline(preset.name)
        assert pipe.selected_backend() == "pydub"
    finally:
        CODEC_REGISTRY.pop(preset.name, None)


@pytest.mark.parametrize("name", ["mp3_16k", "opus_16k"])
def test_ffmpeg_backend_roundtrip_smoke_if_available(name: str) -> None:
    if not check_preset_available(name):
        pytest.skip(f"{name} not available")
    pipe = CodecPipeline(name, backend="ffmpeg")
    y, sr = pipe.roundtrip(_sine(16000, 0.12), 16000)
    assert sr == 16000
    assert y.shape[0] == 1
    assert y.size > 0
    assert np.isfinite(y).all()
