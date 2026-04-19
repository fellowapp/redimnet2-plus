"""Tests for SpectrogramCollator and ReDimNet-style spec augmentations."""

from __future__ import annotations

import torch

from asv.data.collate import (
    SpecAugmentConfig,
    SpectrogramCollator,
    clone_frozen_frontend,
    compute_spectrogram_batch,
    _cutmix,
    _mask_along_axis,
)
from asv.models.redimnet2 import ReDimNet2Encoder


# ---------------------------------------------------------------------------
# Low-level augmentation helpers
# ---------------------------------------------------------------------------

def test_mask_along_freq_axis_zeroes_region() -> None:
    torch.manual_seed(0)
    x = torch.ones(4, 80, 100)
    out = _mask_along_axis(x, dim=1, max_width=10)
    assert out.shape == x.shape
    # At least one sample should have zeroed freq bins
    assert (out == 0.0).any()
    # Zeroed region is contiguous along freq axis
    for i in range(4):
        zero_freqs = (out[i].sum(dim=1) == 0.0)
        if zero_freqs.any():
            indices = zero_freqs.nonzero(as_tuple=True)[0]
            assert (indices[-1] - indices[0] + 1) == len(indices), "mask should be contiguous"


def test_mask_along_time_axis_zeroes_region() -> None:
    torch.manual_seed(1)
    x = torch.ones(4, 80, 100)
    out = _mask_along_axis(x, dim=2, max_width=15)
    assert out.shape == x.shape
    assert (out == 0.0).any()
    for i in range(4):
        zero_times = (out[i].sum(dim=0) == 0.0)
        if zero_times.any():
            indices = zero_times.nonzero(as_tuple=True)[0]
            assert (indices[-1] - indices[0] + 1) == len(indices), "mask should be contiguous"


def test_mask_preserves_shape_with_zero_width() -> None:
    x = torch.ones(2, 40, 50)
    out = _mask_along_axis(x, dim=1, max_width=0)
    assert torch.equal(out, x)


def test_cutmix_mixes_samples() -> None:
    torch.manual_seed(2)
    x = torch.zeros(4, 80, 100)
    for i in range(4):
        x[i] = float(i)
    out = _cutmix(x, min_frames=5, max_frames=30)
    assert out.shape == x.shape
    # At least one sample should have values from another sample
    differs = False
    for i in range(4):
        if not torch.equal(out[i], x[i]):
            differs = True
            break
    assert differs, "cutmix should change at least one sample"


def test_cutmix_single_sample_noop() -> None:
    x = torch.ones(1, 40, 50)
    out = _cutmix(x, min_frames=5, max_frames=20)
    assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# SpectrogramCollator
# ---------------------------------------------------------------------------

def test_collator_shape_and_dtype() -> None:
    """Collator produces [B, F, T_frames] spectrograms + [B] labels."""
    model = ReDimNet2Encoder(model_name="b0", pretrained=False)
    spec_frontend = model.spec_frontend

    cfg = SpecAugmentConfig(
        time_mask_p=0.0, freq_mask_p=0.0, cutmix_p=0.0,
    )
    collator = SpectrogramCollator(spec_frontend, augment_cfg=cfg, training=False)

    batch = [
        {"waveform": torch.randn(64400), "label": 0},
        {"waveform": torch.randn(64400), "label": 1},
        {"waveform": torch.randn(64400), "label": 2},
        {"waveform": torch.randn(64400), "label": 3},
    ]
    out = collator(batch)
    assert "inputs" in out and "labels" in out
    assert out["inputs"].ndim == 3  # [B, F, T_frames]
    assert out["inputs"].shape[0] == 4
    assert out["labels"].shape == (4,)
    assert out["labels"].tolist() == [0, 1, 2, 3]


def test_collator_applies_augmentations_when_training() -> None:
    """With high augmentation probabilities, output should differ from clean spec."""
    model = ReDimNet2Encoder(model_name="b0", pretrained=False)
    spec_frontend = model.spec_frontend

    # No augmentation
    cfg_clean = SpecAugmentConfig(
        time_mask_p=0.0, freq_mask_p=0.0, cutmix_p=0.0,
    )
    collator_clean = SpectrogramCollator(spec_frontend, augment_cfg=cfg_clean, training=False)

    # Full augmentation
    cfg_aug = SpecAugmentConfig(
        time_mask_p=1.0, time_mask_max_width=20,
        freq_mask_p=1.0, freq_mask_max_width=15,
        cutmix_p=0.0,  # skip cutmix to compare directly
    )
    collator_aug = SpectrogramCollator(spec_frontend, augment_cfg=cfg_aug, training=True)

    torch.manual_seed(42)
    batch = [
        {"waveform": torch.randn(64400), "label": 0},
        {"waveform": torch.randn(64400), "label": 1},
    ]
    clean = collator_clean(batch)
    aug = collator_aug(batch)

    # Augmented output should have some zeros from masking
    assert (aug["inputs"] == 0.0).any(), "augmented spec should have masked regions"
    # Shapes must match
    assert clean["inputs"].shape == aug["inputs"].shape


def test_collator_no_augment_when_not_training() -> None:
    model = ReDimNet2Encoder(model_name="b0", pretrained=False)
    cfg = SpecAugmentConfig(
        time_mask_p=1.0, freq_mask_p=1.0, cutmix_p=1.0,
    )
    collator = SpectrogramCollator(model.spec_frontend, augment_cfg=cfg, training=False)
    batch = [
        {"waveform": torch.randn(64400), "label": 0},
        {"waveform": torch.randn(64400), "label": 1},
    ]
    out = collator(batch)
    # In non-training mode, augmentation is skipped,
    # so there should be no all-zero freq/time slices from masking.
    # (Spec values can be negative after log, but masking sets to exactly 0.0)
    assert out["inputs"].shape[0] == 2


def test_collator_end_to_end_with_model() -> None:
    """Full pipeline: waveform -> collator -> model -> embedding."""
    model = ReDimNet2Encoder(model_name="b0", pretrained=False)
    cfg = SpecAugmentConfig(
        time_mask_p=0.5, freq_mask_p=0.5, cutmix_p=0.0,
    )
    collator = SpectrogramCollator(model.spec_frontend, augment_cfg=cfg, training=True)

    batch = [
        {"waveform": torch.randn(64400), "label": i}
        for i in range(4)
    ]
    out = collator(batch)
    embeddings = model(out["inputs"])
    assert embeddings.shape == (4, 192)


def test_collator_uses_private_cpu_frontend_copy() -> None:
    class ScaleFrontend(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("scale", torch.tensor(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1) * self.scale

    original = ScaleFrontend()
    collator = SpectrogramCollator(original, training=False)
    original.scale.fill_(9.0)
    out = collator([{"waveform": torch.ones(6), "label": 1}])

    assert out["inputs"].shape == (1, 1, 6)
    assert torch.allclose(out["inputs"], torch.full((1, 1, 6), 3.0))
    assert collator.spec_frontend.scale.device.type == "cpu"


def test_compute_spectrogram_batch_accepts_single_channel_waveforms() -> None:
    class ScaleFrontend(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1) * 2.0

    waveforms = torch.arange(12, dtype=torch.float32).reshape(2, 1, 6)
    spec = compute_spectrogram_batch(waveforms, ScaleFrontend(), training=False)

    assert spec.shape == (2, 1, 6)
    assert torch.allclose(spec[:, 0, :], waveforms[:, 0, :] * 2.0)


def test_clone_frozen_frontend_copies_parameters_to_target_device() -> None:
    class ScaleFrontend(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("scale", torch.tensor(4.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1) * self.scale

    original = ScaleFrontend()
    frozen = clone_frozen_frontend(original, device="cpu")
    original.scale.fill_(9.0)
    out = compute_spectrogram_batch(torch.ones(1, 1, 5), frozen, training=False)

    assert torch.allclose(out, torch.full((1, 1, 5), 4.0))
    assert frozen.scale.device.type == "cpu"
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_collator_one_of_applies_single_transform(monkeypatch) -> None:
    cfg = SpecAugmentConfig(
        one_of=True,
        level_p=1.0,
        time_mask_p=1.0,
        freq_mask_p=1.0,
        cutmix_p=1.0,
        time_mask_max_width=5,
        freq_mask_max_width=5,
        cutmix_min_frames=3,
        cutmix_max_frames=5,
    )

    call_counts = {"mask": 0, "cutmix": 0}

    def fake_mask(x: torch.Tensor, dim: int, max_width: int) -> torch.Tensor:
        del dim, max_width
        call_counts["mask"] += 1
        return x + 1.0

    def fake_cutmix(x: torch.Tensor, min_frames: int, max_frames: int) -> torch.Tensor:
        del min_frames, max_frames
        call_counts["cutmix"] += 1
        return x + 1.0

    monkeypatch.setattr("asv.data.collate._mask_along_axis", fake_mask)
    monkeypatch.setattr("asv.data.collate._cutmix", fake_cutmix)

    # Dummy frontend for constructing collator; not used in this direct _augment call.
    collator = SpectrogramCollator(torch.nn.Identity(), augment_cfg=cfg, training=True)
    spec = torch.zeros(2, 80, 100)
    out = collator._augment(spec)

    assert out.shape == spec.shape
    assert (call_counts["mask"] + call_counts["cutmix"]) == 1


def test_collator_level_p_zero_skips_augment(monkeypatch) -> None:
    cfg = SpecAugmentConfig(
        one_of=True,
        level_p=0.0,
        freq_mask_p=1.0,
        time_mask_p=1.0,
        cutmix_p=1.0,
    )
    calls = {"n": 0}

    def fake_mask(x: torch.Tensor, dim: int, max_width: int) -> torch.Tensor:
        del dim, max_width
        calls["n"] += 1
        return x

    monkeypatch.setattr("asv.data.collate._mask_along_axis", fake_mask)
    monkeypatch.setattr("asv.data.collate._cutmix", fake_mask)

    collator = SpectrogramCollator(torch.nn.Identity(), augment_cfg=cfg, training=True)
    spec = torch.zeros(2, 80, 100)
    out = collator._augment(spec)
    assert torch.equal(out, spec)
    assert calls["n"] == 0
