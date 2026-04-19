import types
import sys

import pytest
import torch

from asv.augment import (
    AudioAugmentPipeline,
    RawBoostTransform,
    SpectrogramCutMix,
    SpectrogramTimeMask,
    SpeedPerturbation,
    build_audio_augmentor,
)


def test_speed_perturbation_changes_length() -> None:
    aug = SpeedPerturbation(min_rate=0.9, max_rate=0.9, p=1.0)
    aug.train()
    x = torch.randn(2, 1, 1600)
    y = aug(x, 16000)
    assert y.ndim == 3
    assert y.shape[-1] != x.shape[-1]


def test_speed_perturbation_defaults_to_ten_uniform_factors() -> None:
    aug = SpeedPerturbation(min_rate=0.9, max_rate=1.1, p=1.0)
    assert len(aug.factors) == 10
    assert aug.factors[0] == pytest.approx(0.9)
    assert aug.factors[-1] == pytest.approx(1.1)
    for i in range(len(aug.factors) - 1):
        assert aug.factors[i] < aug.factors[i + 1]


def test_feature_mask_and_cutmix_shapes() -> None:
    x = torch.randn(4, 80, 120)
    mask = SpectrogramTimeMask(max_frames=15, p=1.0)
    mix = SpectrogramCutMix(min_mix_frames=5, max_mix_frames=20, p=1.0)
    mask.train()
    mix.train()
    y = mix(mask(x))
    assert y.shape == x.shape


def test_rawboost_wrapper_calls_target(monkeypatch) -> None:
    mod = types.ModuleType("tmp_rawboost")

    def fake(samples: torch.Tensor, sample_rate: int, scale: float) -> torch.Tensor:
        return samples * scale + float(sample_rate) * 0.0

    mod.apply = fake
    monkeypatch.setitem(sys.modules, "tmp_rawboost", mod)
    aug = RawBoostTransform(target="tmp_rawboost.apply", kwargs={"scale": 0.5}, p=1.0)
    aug.train()
    x = torch.ones(2, 1, 8)
    y = aug(x, 16000)
    assert torch.allclose(y, torch.full_like(x, 0.5))


def test_build_audio_augmentor_from_dict() -> None:
    cfg = {
        "enabled": True,
        "waveform_levels": [
            {
                "name": "mask",
                "p": 1.0,
                "one_of": True,
                "transforms": [
                    {"_target_": "asv.augment.SpeedPerturbation", "min_rate": 0.95, "max_rate": 0.95, "p": 1.0}
                ],
            }
        ],
        "feature_levels": [],
    }
    pipeline = build_audio_augmentor(cfg)
    assert isinstance(pipeline, AudioAugmentPipeline)
    pipeline.train()
    x = torch.randn(1, 1, 512)
    y = pipeline(x, 16000)
    assert y.ndim == x.ndim
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1]


def test_rawboost_builtin_runs_and_preserves_shape() -> None:
    aug = RawBoostTransform(algo=5, cfg={"N_f": 2, "nBands": 2}, p=1.0)
    aug.train()
    x = torch.randn(2, 1, 3200)
    y = aug(x, 16000)
    assert y.shape == x.shape
