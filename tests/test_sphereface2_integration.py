from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from asv.losses.integrated import LossComputedInModel
from asv.models.classifier import SphereFace2Head


def test_sphereface2_head_train_returns_logits_and_loss() -> None:
    head = SphereFace2Head(32, 8, margin=0.2, margin_type="C")
    z = torch.randn(4, 32)
    labels = torch.randint(0, 8, (4,))
    out = head(z, labels)
    assert isinstance(out, tuple)
    logits, loss = out
    assert logits.shape == (4, 8)
    assert loss.shape == ()


def test_sphereface2_head_eval_returns_logits() -> None:
    head = SphereFace2Head(32, 8)
    z = torch.randn(4, 32)
    logits = head(z, None)
    assert logits.shape == (4, 8)


def test_sphereface2_head_margin_schedule_uses_cosine_decay() -> None:
    head = SphereFace2Head(32, 8, first_margin=0.3, last_margin=0.1)

    assert head.has_margin_schedule()
    assert head.set_margin_for_step(
        step=0, total_steps=5
    ) == pytest.approx(0.3)
    assert head.set_margin_for_step(
        step=2, total_steps=5
    ) == pytest.approx(0.2)
    assert head.set_margin_for_step(
        step=4, total_steps=5
    ) == pytest.approx(0.1)


def test_sphereface2_head_requires_complete_margin_schedule() -> None:
    with pytest.raises(ValueError, match="both first_margin and last_margin"):
        SphereFace2Head(32, 8, first_margin=0.3)


def test_loss_computed_in_model_is_not_callable_at_runtime() -> None:
    loss_mod = LossComputedInModel()
    with pytest.raises(RuntimeError, match="must not be called"):
        loss_mod(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))


def test_compose_sphereface2_with_matching_model_preset() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "loss=sphereface2",
                "model=redimnet2_b6_sphereface2",
                "model.num_classes=16",
            ],
        )
    assert str(cfg.model.classifier._target_).endswith("SphereFace2Head")
    assert str(cfg.loss._target_).endswith("LossComputedInModel")
    assert cfg.model.classifier.in_features == 192


def test_compose_sphereface2_margin_schedule_overrides_instantiate_head(
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "loss=sphereface2",
                "model=redimnet2_b6_sphereface2",
                "model.num_classes=16",
                "model.classifier.first_margin=0.3",
                "model.classifier.last_margin=0.1",
            ],
        )
    head = instantiate(cfg.model.classifier)
    assert isinstance(head, SphereFace2Head)
    assert head.has_margin_schedule()
    assert head.current_margin == pytest.approx(0.3)
