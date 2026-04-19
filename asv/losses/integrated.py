"""Плейсхолдер-лосс: нужен, когда сам тренировочный критерий считается внутри
модели (например, в ``SphereFace2Head``, который возвращает ``(logits, loss)``).

Сам модуль никогда не вызывается; он существует только чтобы Hydra/Accelerate
могли ``prepare`` какой-то loss-модуль без падения. Реальный лосс цикл трейна
берёт из выхода модели."""

from __future__ import annotations

from torch import nn


class LossComputedInModel(nn.Module):
    """Not used at runtime when the classifier returns ``(logits, loss)``.

    Exists so Hydra/Accelerate can still ``prepare`` a loss module; the training loop
    must take the scalar loss from the model output instead of calling this module.
    """

    def forward(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        raise RuntimeError(
            "LossComputedInModel must not be called: use a classifier head that returns "
            "(logits, loss) and the training loop integrated-loss branch."
        )
