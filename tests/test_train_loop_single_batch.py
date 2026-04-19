from __future__ import annotations

import json

import pytest
import torch

from asv.train.loop import (
    _append_validation_metrics,
    _compute_total_margin_schedule_steps,
    _compute_single_batch_eer_metrics,
)


def test_compute_single_batch_eer_metrics_is_zero_for_separable_embeddings(
) -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    metrics = _compute_single_batch_eer_metrics(embeddings, labels)

    assert metrics["val/single_batch/eer"] == 0.0
    assert metrics["val/single_batch/positive_score_mean"] == pytest.approx(
        1.0
    )
    assert metrics["val/single_batch/negative_score_mean"] == pytest.approx(
        0.0
    )
    assert metrics["val/single_batch/num_unique_labels"] == 2.0
    assert metrics["val/single_batch/max_label_count"] == 2.0


def test_compute_single_batch_eer_metrics_requires_multiple_labels() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="distinct speaker labels"):
        _compute_single_batch_eer_metrics(embeddings, labels)


def test_append_validation_metrics_writes_jsonl(tmp_path) -> None:
    metrics = {
        "val/eer/pooled": 0.123,
        "val/eer/voxceleb1_o": 0.111,
    }

    _append_validation_metrics(
        output_dir=tmp_path,
        tag="epoch_001_end",
        metrics=metrics,
    )

    payload = json.loads(
        (tmp_path / "validation_metrics.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert payload["tag"] == "epoch_001_end"
    assert payload["metrics"] == metrics


def test_compute_total_margin_schedule_steps_uses_effective_optimizer_steps(
) -> None:
    train_loader = [0] * 10

    total_steps = _compute_total_margin_schedule_steps(
        train_loader,
        grad_accum_steps=4,
        max_epochs=3,
    )

    assert total_steps == 9
