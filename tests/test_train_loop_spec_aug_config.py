from __future__ import annotations

from omegaconf import OmegaConf

from asv.train.spec_augment import build_spec_augment_cfg


def test_spec_augment_cfg_reads_from_augment_feature_level() -> None:
    cfg = OmegaConf.create(
        {
            "augment": {
                "feature_levels": [
                    {
                        "name": "spec_augment",
                        "spec_augment": {
                            "freq_mask": {"p": 0.9, "max_width": 11},
                            "time_mask": {"p": 0.8, "max_width": 13},
                            "cutmix": {"p": 0.7, "min_frames": 6, "max_frames": 21},
                        },
                    }
                ]
            },
            "data": {},
        }
    )

    out = build_spec_augment_cfg(cfg)
    assert out.one_of is False
    assert out.level_p == 1.0
    assert out.freq_mask_p == 0.9
    assert out.freq_mask_max_width == 11
    assert out.time_mask_p == 0.8
    assert out.time_mask_max_width == 13
    assert out.cutmix_p == 0.7
    assert out.cutmix_min_frames == 6
    assert out.cutmix_max_frames == 21


def test_spec_augment_cfg_falls_back_to_data_node() -> None:
    cfg = OmegaConf.create(
        {
            "augment": {"feature_levels": []},
            "data": {
                "spec_augment": {
                    "level_p": 0.6,
                    "freq_mask": {"p": 0.4, "max_width": 7},
                    "time_mask": {"p": 0.3, "max_width": 9},
                    "cutmix": {"p": 0.2, "min_frames": 4, "max_frames": 17},
                }
            },
        }
    )

    out = build_spec_augment_cfg(cfg)
    assert out.one_of is False
    assert out.level_p == 0.6
    assert out.freq_mask_p == 0.4
    assert out.freq_mask_max_width == 7
    assert out.time_mask_p == 0.3
    assert out.time_mask_max_width == 9
    assert out.cutmix_p == 0.2
    assert out.cutmix_min_frames == 4
    assert out.cutmix_max_frames == 17


def test_spec_augment_cfg_reads_one_of_from_augment_feature_level() -> None:
    cfg = OmegaConf.create(
        {
            "augment": {
                "feature_levels": [
                    {
                        "name": "spec_augment",
                        "one_of": True,
                        "spec_augment": {
                            "freq_mask": {"p": 0.9, "max_width": 11},
                            "time_mask": {"p": 0.8, "max_width": 13},
                            "cutmix": {"p": 0.7, "min_frames": 6, "max_frames": 21},
                        },
                    }
                ]
            },
            "data": {},
        }
    )

    out = build_spec_augment_cfg(cfg)
    assert out.one_of is True
    assert out.level_p == 1.0


def test_spec_augment_cfg_reads_level_p_from_feature_level() -> None:
    cfg = OmegaConf.create(
        {
            "augment": {
                "feature_levels": [
                    {
                        "name": "spec_augment",
                        "p": 0.85,
                        "one_of": True,
                        "spec_augment": {
                            "freq_mask": {"p": 0.9, "max_width": 11},
                            "time_mask": {"p": 0.8, "max_width": 13},
                            "cutmix": {"p": 0.7, "min_frames": 6, "max_frames": 21},
                        },
                    }
                ]
            },
            "data": {},
        }
    )

    out = build_spec_augment_cfg(cfg)
    assert out.level_p == 0.85
