from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from asv.train.loop import _read_resume_training_state, _resolve_resume_checkpoint_dir


def test_resolve_resume_none_and_empty() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "checkpoint_dir": "/tmp/ckpt",
                "resume_from_checkpoint": None,
            }
        }
    )
    assert _resolve_resume_checkpoint_dir(cfg) is None
    cfg.train.resume_from_checkpoint = ""
    assert _resolve_resume_checkpoint_dir(cfg) is None
    cfg.train.resume_from_checkpoint = "  "
    assert _resolve_resume_checkpoint_dir(cfg) is None


def test_resolve_resume_latest(tmp_path: Path) -> None:
    ckpt_root = tmp_path / "runs"
    save_dir = "epoch_0002_step_00000099"
    epoch_dir = ckpt_root / save_dir
    epoch_dir.mkdir(parents=True)
    (ckpt_root / "latest.json").write_text(
        json.dumps(
            {
                "epoch": 2,
                "global_step": 99,
                "optimizer_step": 100,
                "next_epoch": 3,
                "save_dir": save_dir,
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "train": {
                "checkpoint_dir": str(ckpt_root),
                "resume_from_checkpoint": "latest",
            }
        }
    )
    assert _resolve_resume_checkpoint_dir(cfg) == epoch_dir.resolve()


def test_resolve_resume_latest_uses_checkpoint_dir_by_default(tmp_path: Path) -> None:
    ckpt_root = tmp_path / "same_run" / "checkpoints"
    ckpt_root.mkdir(parents=True)
    save_dir = "epoch_0001_step_00000005"
    (ckpt_root / save_dir).mkdir()
    (ckpt_root / "latest.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "global_step": 5,
                "optimizer_step": 5,
                "next_epoch": 1,
                "save_dir": save_dir,
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "train": {
                "checkpoint_dir": str(ckpt_root),
                "resume_from_checkpoint": "latest",
            }
        }
    )
    assert _resolve_resume_checkpoint_dir(cfg) == (ckpt_root / save_dir).resolve()


def test_resolve_resume_latest_legacy_epoch_only_dir(tmp_path: Path) -> None:
    """Older runs used epoch_XXXX only; latest.json without save_dir falls back to that folder."""
    ckpt_root = tmp_path / "runs"
    legacy = ckpt_root / "epoch_0002"
    legacy.mkdir(parents=True)
    (ckpt_root / "latest.json").write_text(
        json.dumps({"epoch": 2, "global_step": 99, "optimizer_step": 100, "next_epoch": 3}),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "train": {
                "checkpoint_dir": str(ckpt_root),
                "resume_from_checkpoint": "latest",
            }
        }
    )
    assert _resolve_resume_checkpoint_dir(cfg) == legacy.resolve()


def test_resolve_resume_explicit_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "my_epoch"
    target.mkdir()
    cfg = OmegaConf.create(
        {
            "train": {
                "checkpoint_dir": str(tmp_path / "other"),
                "resume_from_checkpoint": "my_epoch",
            }
        }
    )
    assert _resolve_resume_checkpoint_dir(cfg) == target.resolve()


def test_read_resume_training_state(tmp_path: Path) -> None:
    epoch_dir = tmp_path / "epoch_0001_step_00005000"
    epoch_dir.mkdir()
    (epoch_dir / "training_state.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "global_step": 5000,
                "optimizer_step": 10000,
                "next_epoch": 2,
                "save_dir": "epoch_0001_step_00005000",
            }
        ),
        encoding="utf-8",
    )
    state = _read_resume_training_state(epoch_dir)
    assert state == {"next_epoch": 2, "global_step": 5000, "optimizer_step": 10000}


def test_read_resume_training_state_missing(tmp_path: Path) -> None:
    epoch_dir = tmp_path / "epoch_0001_step_00000001"
    epoch_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="training_state"):
        _read_resume_training_state(epoch_dir)
