from __future__ import annotations

import os
from pathlib import Path

import torch

from submission.run_submission_unified import (
    _autodetect_weights_path,
    _clip_to_max_length,
    _resolve_neighbor_params,
)


def test_clip_to_max_length_truncates_long_waveform_from_start() -> None:
    waveform = torch.arange(10, dtype=torch.float32)

    out = _clip_to_max_length(waveform, 4)

    assert torch.equal(out, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_clip_to_max_length_keeps_short_waveform_unchanged() -> None:
    waveform = torch.tensor([1.0, 2.0, 3.0])

    out = _clip_to_max_length(waveform, 8)

    assert torch.equal(out, waveform)


def test_resolve_neighbor_params_caps_to_available_neighbors() -> None:
    effective_k, effective_pool_k = _resolve_neighbor_params(
        num_items=3,
        requested_k=10,
        requested_candidate_pool_k=200,
    )

    assert effective_k == 2
    assert effective_pool_k == 2


def test_autodetect_weights_path_picks_latest_checkpoint_artifact(tmp_path: Path) -> None:
    older = tmp_path / "older" / "model.pt"
    older.parent.mkdir(parents=True)
    older.write_bytes(b"older")

    newer = tmp_path / "newer" / "model.safetensors"
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b"newer")

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert _autodetect_weights_path(tmp_path) == newer.resolve()
