#!/usr/bin/env python3
"""Hydra-точка входа для трейна ASV.

Читает корневой конфиг ``configs/config.yaml`` (model/data/augment/loss/logging
+ корневые поля ``seed``, ``optim``, ``train``, ``validation``, ``variant_validation``)
и передаёт его в :func:`asv.train.run.run`, где строится модель, лосс и запускается
Accelerate-цикл. Все поля переопределяются из CLI (например, ``data.batch_size=64``)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from asv.train.run import run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
