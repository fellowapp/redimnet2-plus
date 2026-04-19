from __future__ import annotations

from pathlib import Path

import pytest

from asv.logging.mlop import log_mlop_commit_artifact, log_mlop_metrics


class _RecordingRun:
    def __init__(self) -> None:
        self.logged_payloads: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.logged_payloads.append(payload)


def test_log_mlop_metrics_scales_eer_and_logs_other_keys() -> None:
    run = _RecordingRun()
    metrics = {
        "train/loss": 0.25,
        "val/before_epoch_1/eer/voxceleb1_o": 0.041,
        "val/epoch_1/eer/pooled": 0.112,
        "val/variants/eer_aug__filters__BandPassFilter": 0.083,
        "step": 1,
        "epoch": 2,
        "artifact/git_commit": "abc",
        "val/before/epoch_1/num_unique/utts": 11,
        "val/before_epoch_1/num_unique_utts": 11,
    }

    log_mlop_metrics(run, metrics)

    assert len(run.logged_payloads) == 1
    payload = run.logged_payloads[0]
    assert payload["train/loss"] == 0.25
    assert payload["val/before_epoch_1/eer/voxceleb1_o"] == pytest.approx(4.1)
    assert payload["val/epoch_1/eer/pooled"] == pytest.approx(11.2)
    assert payload["val/variants/eer_aug__filters__BandPassFilter"] == pytest.approx(8.3)
    assert payload["step"] == 1
    assert payload["epoch"] == 2
    assert payload["artifact/git_commit"] == "abc"
    assert payload["val/before/epoch_1/num_unique/utts"] == 11
    assert payload["val/before_epoch_1/num_unique_utts"] == 11


def test_log_mlop_commit_artifact_keeps_commit_hash_only(tmp_path: Path) -> None:
    run = _RecordingRun()
    log_mlop_commit_artifact(run, commit_hash="abc123", output_dir=tmp_path)

    assert len(run.logged_payloads) == 1
    assert run.logged_payloads[0] == {"git/commit_hash": "abc123"}
    assert (tmp_path / "git_commit.txt").read_text(encoding="utf-8").startswith("abc123")
