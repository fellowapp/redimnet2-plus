from __future__ import annotations

import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from asv.train.loop import _compute_total_optimizer_steps


def test_total_optimizer_steps_account_for_grad_accumulation() -> None:
    loader = DataLoader(list(range(10)), batch_size=1)

    total_steps = _compute_total_optimizer_steps(
        loader,
        grad_accum_steps=4,
        max_epochs=3,
    )

    assert total_steps == 9


def test_wrapped_scheduler_steps_only_on_real_optimizer_updates() -> None:
    dataset = TensorDataset(
        torch.arange(4, dtype=torch.float32).unsqueeze(1),
        torch.arange(4, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=2,
        eta_min=0.0,
    )
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    scheduler = accelerator.prepare_scheduler(scheduler)

    lrs_after_step: list[float] = []
    for inputs, targets in loader:
        with accelerator.accumulate(model):
            loss = ((model(inputs) - targets) ** 2).mean()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            lrs_after_step.append(float(optimizer.param_groups[0]["lr"]))

    assert lrs_after_step == pytest.approx([1.0, 0.5, 0.5, 0.0])
