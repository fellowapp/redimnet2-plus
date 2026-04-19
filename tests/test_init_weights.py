from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from asv.train.init_weights import (
    _resolve_accelerate_checkpoint_dir,
    initialize_model_from_weights,
)


class TinyAsvModel(nn.Module):
    def __init__(self, classifier_out: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.bridge = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, classifier_out)


def test_resolve_accelerate_checkpoint_dir_from_latest_json(
    tmp_path: Path,
) -> None:
    ckpt_root = tmp_path / "run" / "checkpoints"
    save_dir = "epoch_0002_step_00000099"
    (ckpt_root / save_dir).mkdir(parents=True)
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
    assert _resolve_accelerate_checkpoint_dir(ckpt_root) == (
        ckpt_root / save_dir
    ).resolve()


def test_initialize_model_from_weights_skips_mismatched_classifier(
    tmp_path: Path,
) -> None:
    source_model = TinyAsvModel(classifier_out=5)
    target_model = TinyAsvModel(classifier_out=4)

    source_model.encoder.weight.data.fill_(1.25)
    source_model.encoder.bias.data.fill_(0.25)
    source_model.bridge.weight.data.fill_(2.5)
    source_model.bridge.bias.data.fill_(0.5)
    source_model.classifier.weight.data.fill_(7.0)
    source_model.classifier.bias.data.fill_(8.0)

    target_model.encoder.weight.data.zero_()
    target_model.encoder.bias.data.zero_()
    target_model.bridge.weight.data.zero_()
    target_model.bridge.bias.data.zero_()
    original_classifier_weight = (
        target_model.classifier.weight.detach().clone()
    )
    original_classifier_bias = target_model.classifier.bias.detach().clone()

    ckpt_dir = tmp_path / "epoch_0001_step_00000001"
    ckpt_dir.mkdir()
    torch.save(source_model.state_dict(), ckpt_dir / "model.pt")
    (ckpt_dir / "training_state.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "global_step": 1,
                "optimizer_step": 1,
                "next_epoch": 2,
                "save_dir": ckpt_dir.name,
            }
        ),
        encoding="utf-8",
    )

    resolved = initialize_model_from_weights(
        target_model,
        ckpt_dir,
        init_mode="full",
    )

    assert resolved == ckpt_dir.resolve()
    assert torch.allclose(target_model.encoder.weight, source_model.encoder.weight)
    assert torch.allclose(target_model.encoder.bias, source_model.encoder.bias)
    assert torch.allclose(
        target_model.bridge.weight,
        source_model.bridge.weight,
    )
    assert torch.allclose(target_model.bridge.bias, source_model.bridge.bias)
    assert torch.allclose(
        target_model.classifier.weight,
        original_classifier_weight,
    )
    assert torch.allclose(
        target_model.classifier.bias,
        original_classifier_bias,
    )


def test_initialize_model_from_weights_encoder_only(tmp_path: Path) -> None:
    source_model = TinyAsvModel(classifier_out=4)
    target_model = TinyAsvModel(classifier_out=4)

    source_model.encoder.weight.data.fill_(3.0)
    source_model.encoder.bias.data.fill_(4.0)
    source_model.bridge.weight.data.fill_(5.0)
    source_model.bridge.bias.data.fill_(6.0)

    original_bridge_weight = target_model.bridge.weight.detach().clone()
    original_bridge_bias = target_model.bridge.bias.detach().clone()

    ckpt_file = tmp_path / "model.pt"
    torch.save(source_model.state_dict(), ckpt_file)

    initialize_model_from_weights(
        target_model,
        ckpt_file,
        init_mode="encoder_only",
    )

    assert torch.allclose(target_model.encoder.weight, source_model.encoder.weight)
    assert torch.allclose(target_model.encoder.bias, source_model.encoder.bias)
    assert torch.allclose(
        target_model.bridge.weight,
        original_bridge_weight,
    )
    assert torch.allclose(target_model.bridge.bias, original_bridge_bias)
