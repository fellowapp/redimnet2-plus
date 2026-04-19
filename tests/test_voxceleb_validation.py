from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from asv.validation import voxceleb as voxceleb_mod
from asv.validation.voxceleb import (
    ValidationWaveformCollator,
    collect_super_hard_variant_roots,
    collect_variant_validation_roots,
    compute_eer,
    parse_trial_setups,
    resolve_variant_validation_trial_files,
)


def test_compute_eer_perfect_separation() -> None:
    labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
    scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
    eer = compute_eer(labels, scores)
    assert eer == pytest.approx(0.0, abs=1e-9)


def test_compute_eer_balanced_overlap() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int32)
    scores = np.asarray([0.6, 0.7, 0.4, 0.3], dtype=np.float64)
    eer = compute_eer(labels, scores)
    assert eer == pytest.approx(0.5, abs=1e-9)


def test_parse_trial_setups_multiple_files(tmp_path: Path) -> None:
    root = tmp_path / "voxceleb"
    root.mkdir(parents=True, exist_ok=True)
    for rel_path in [
        "id1/v1/00001.wav",
        "id1/v2/00001.wav",
        "id2/v9/00003.wav",
        "id3/v4/00001.wav",
        "id3/v4/00002.wav",
    ]:
        wav_path = root / rel_path
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"")
    trial_a = root / "voxceleb1_o.txt"
    trial_b = root / "voxceleb_e.txt"
    trial_a.write_text(
        "1 id1/v1/00001.wav id1/v2/00001.wav\n"
        "0 id1/v1/00001.wav id2/v9/00003.wav\n",
        encoding="utf-8",
    )
    trial_b.write_text(
        "1 id3/v4/00001.wav id3/v4/00002.wav\n"
        "0 id3/v4/00001.wav id1/v2/00001.wav\n",
        encoding="utf-8",
    )

    utt_rel_paths, utt_to_id, setups = parse_trial_setups(
        trial_files=[str(trial_a), str(trial_b)],
        data_root=root,
    )

    assert len(utt_rel_paths) == 5
    assert len(utt_to_id) == 5
    assert [setup.name for setup in setups] == ["voxceleb1_o", "voxceleb_e"]
    assert setups[0].labels.tolist() == [1, 0]
    assert setups[1].labels.tolist() == [1, 0]


def test_parse_trial_setups_hard_build_glob_maps_stem(tmp_path: Path) -> None:
    root = tmp_path / "hard_root"
    utt_dir = root / "id1" / "v1"
    utt_dir.mkdir(parents=True)
    (utt_dir / "00001__hard_0000.wav").write_bytes(b"")
    (utt_dir / "00002__hard_0001.wav").write_bytes(b"")
    trial = root / "trials.txt"
    trial.write_text(
        "1 id1/v1/00001.wav id1/v1/00002.wav\n",
        encoding="utf-8",
    )
    utt_rel_paths, utt_to_id, setups = parse_trial_setups(
        trial_files=[str(trial)],
        data_root=root,
        hard_build_glob=True,
    )
    assert utt_rel_paths == ["id1/v1/00001__hard_0000.wav", "id1/v1/00002__hard_0001.wav"]
    assert len(setups) == 1
    assert setups[0].labels.tolist() == [1]


def test_resolve_variant_validation_trial_files_o_only_by_default() -> None:
    cfg = OmegaConf.create(
        {
            "validation": {
                "trial_files": [
                    "/data/voxceleb1_o.txt",
                    "/data/voxceleb_e.txt",
                    "/data/voxceleb_h.txt",
                ],
            },
            "variant_validation": {"trial_files": []},
        }
    )
    assert resolve_variant_validation_trial_files(cfg) == ["/data/voxceleb1_o.txt"]


def test_sanitize_variant_eer_suffix() -> None:
    assert voxceleb_mod._sanitize_variant_eer_suffix("aug__filters__X") == "aug__filters__X"
    assert voxceleb_mod._sanitize_variant_eer_suffix("a/b c") == "a_b_c"


def test_resolve_variant_validation_trial_files_explicit_override() -> None:
    cfg = OmegaConf.create(
        {
            "validation": {"trial_files": ["/data/voxceleb1_o.txt"]},
            "variant_validation": {"trial_files": ["/custom/all.txt"]},
        }
    )
    assert resolve_variant_validation_trial_files(cfg) == ["/custom/all.txt"]


def test_collect_variant_validation_roots_scan_and_explicit(tmp_path: Path) -> None:
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "aug__a").mkdir()
    (scan / "aug__b").mkdir()
    (scan / ".hidden").mkdir()
    extra = tmp_path / "extra_codec"
    extra.mkdir()
    cfg = OmegaConf.create(
        {
            "variant_validation": {
                "roots": [str(extra)],
                "scan_root": str(scan),
            }
        }
    )
    roots = collect_variant_validation_roots(cfg)
    names = [n for n, _ in roots]
    assert names[0] == "extra_codec"
    assert sorted(names[1:]) == ["aug__a", "aug__b"]


def test_collect_super_hard_variant_roots_accepts_scalar_path_string(tmp_path: Path) -> None:
    """CLI ``super_hard_roots=/abs/path`` is stored as str; must not iterate characters."""
    root = tmp_path / "HardOnly"
    root.mkdir()
    cfg = OmegaConf.create(
        {
            "variant_validation": {
                "super_hard_roots": str(root),
                "super_hard_scan_root": "",
            }
        }
    )
    roots = collect_super_hard_variant_roots(cfg)
    assert roots == [(root.name, str(root.resolve()))]


def test_collect_super_hard_variant_roots_scan_and_explicit(tmp_path: Path) -> None:
    scan = tmp_path / "super_scan"
    scan.mkdir()
    (scan / "codec_a").mkdir()
    (scan / "codec_b").mkdir()
    explicit = tmp_path / "super_extra"
    explicit.mkdir()
    cfg = OmegaConf.create(
        {
            "variant_validation": {
                "super_hard_roots": [str(explicit)],
                "super_hard_scan_root": str(scan),
            }
        }
    )
    roots = collect_super_hard_variant_roots(cfg)
    names = [n for n, _ in roots]
    assert names[0] == "super_extra"
    assert sorted(names[1:]) == ["codec_a", "codec_b"]


def test_parse_trial_setups_rejects_malformed_lines(tmp_path: Path) -> None:
    root = tmp_path / "voxceleb"
    root.mkdir(parents=True, exist_ok=True)
    trial = root / "broken.txt"
    trial.write_text("1 only_two_fields.wav\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_trial_setups(trial_files=[str(trial)], data_root=root)


def test_validation_collator_batches_waveforms() -> None:
    collator = ValidationWaveformCollator()
    batch = [
        {"waveform": torch.randn(64400), "utt_id": 3},
        {"waveform": torch.randn(64400), "utt_id": 8},
    ]
    out = collator(batch)

    assert out["inputs"].ndim == 3
    assert out["inputs"].shape[0] == 2
    assert out["inputs"].shape[1] == 1
    assert out["utt_ids"].tolist() == [3, 8]
