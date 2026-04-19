from __future__ import annotations

import pytest
import torch

from asv.models.redimnet2 import ReDimNet2Encoder


def test_redimnet2_scratch_init_and_forward_with_spec() -> None:
    """Model now expects spectrogram input, not raw waveform."""
    model = ReDimNet2Encoder(model_name="b0", pretrained=False)
    # Use the exposed spec_frontend to compute spectrogram from waveform.
    waveform = torch.randn(2, 16000)
    with torch.no_grad():
        spec = model.spec_frontend(waveform)  # [B, F, T_frames]
    y = model(spec)
    assert y.shape == (2, 192)


def test_redimnet2_spec_frontend_exposed() -> None:
    model = ReDimNet2Encoder(model_name="b0", pretrained=False)
    assert model.spec_frontend is not None
    waveform = torch.randn(1, 16000)
    with torch.no_grad():
        spec = model.spec_frontend(waveform)
    assert spec.ndim == 3  # [B, F, T]
    assert spec.shape[0] == 1


def test_redimnet2_pretrained_load_and_forward() -> None:
    try:
        model = ReDimNet2Encoder(model_name="b0", pretrained=True, train_type="lm", strict_load=True)
    except Exception as exc:  # pragma: no cover - network/cache dependent
        pytest.skip(f"pretrained checkpoint unavailable: {exc}")
    model.eval()
    waveform = torch.randn(1, 16000)
    with torch.no_grad():
        spec = model.spec_frontend(waveform)
        y = model(spec)
    assert y.shape == (1, 192)
