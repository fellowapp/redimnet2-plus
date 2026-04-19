"""Драйвер трейна, который вызывается из Hydra-entrypoint (``train.py``).

Здесь собирается модель через :func:`asv.models.assemble.build_model`, лосс через
:func:`asv.losses.registry.build_loss`, опционально подгружаются веса (``train.init_weights``)
и запускается основной цикл :func:`asv.train.loop.train` под Accelerate/FSDP."""

from __future__ import annotations

import os
import warnings
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\. Please install nvidia-ml-py",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Transforms now expect an `output_type` argument.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Online softmax is disabled on the fly",
    category=UserWarning,
    module=r"torch\._inductor\.lowering",
)

import torch

import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from asv.losses import build_loss
from asv.models import build_model
from asv.train.init_weights import initialize_model_from_weights
from asv.train.loop import train


def _is_main_process() -> bool:
    return os.getenv("LOCAL_RANK", "0") == "0"


def run(cfg: DictConfig) -> None:
    """Разобрать конфиг, собрать модель/лосс, запустить Accelerate-трейн.

    Поведение:
    - Резолвит интерполяции конфига, проверяет что ``data.manifest_path`` задан.
    - Запрещает одновременно ``train.init_weights`` и ``train.resume_from_checkpoint``.
    - Если ``train.auto_num_classes=true`` — сам высчитывает число спикеров из манифеста
      (медленно на больших датасетах, обычно удобнее передать через ``ASV_NUM_CLASSES``
      в ``run.sh`` и прокинуть в ``model.num_classes``).
    - Опционально компилит модель через ``torch.compile``.
    - Вызывает :func:`asv.train.loop.train` для основного цикла."""
    OmegaConf.resolve(cfg)
    if not cfg.data.manifest_path:
        raise ValueError("data.manifest_path must be set to a parquet manifest path")
    if cfg.train.get("init_weights") and cfg.train.get("resume_from_checkpoint"):
        raise ValueError(
            "train.init_weights and train.resume_from_checkpoint are mutually exclusive. "
            "Use init_weights for a fresh finetuning stage and resume_from_checkpoint for full resume."
        )

    if bool(cfg.train.auto_num_classes):
        parquet_file = pq.ParquetFile(str(cfg.data.manifest_path), memory_map=True)
        speakers: set[str] = set()
        for rg in range(parquet_file.num_row_groups):
            batch = parquet_file.read_row_group(rg, columns=["speaker_id"])
            speakers.update(str(s) for s in batch.column("speaker_id").to_pylist())
        cfg.model.num_classes = len(speakers)
        if _is_main_process():
            logger.info("Auto-set model.num_classes from manifest: {}", cfg.model.num_classes)

    if _is_main_process():
        logger.info("Model config node: keys={}", list(cfg.model.keys()))
    if _is_main_process():
        logger.info("Building model and loss from Hydra config")
    model = build_model(cfg.model)
    initialized_from = initialize_model_from_weights(
        model,
        cfg.train.get("init_weights"),
        init_mode=str(cfg.train.get("init_mode", "full")),
    )
    if _is_main_process():
        logger.info("Model initialized: {}", type(model).__name__)
        if initialized_from is not None:
            logger.info(
                "Fresh-stage finetuning init loaded from {} with mode={}",
                initialized_from,
                cfg.train.init_mode,
            )
    if bool(cfg.train.torch_compile):
        mode = str(cfg.train.torch_compile_mode)
        if _is_main_process():
            logger.info("Compiling model with torch.compile(mode='{}')", mode)
        model = torch.compile(model, mode=mode)
    if _is_main_process():
        logger.info("Model ready after optional compile step")
    if _is_main_process():
        logger.info("Initializing loss from Hydra config")
    loss_fn = build_loss(cfg.loss)
    if _is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model parameters: {:,} total / {:,} trainable", n_params, n_trainable)
        logger.info("Loss initialized: {}", type(loss_fn).__name__)
    train(cfg, model=model, loss_fn=loss_fn)


def run_with_cfg_dict(cfg: dict[str, Any]) -> None:
    """Хелпер для тестов и вызовов без Hydra: принимает dict и оборачивает в DictConfig."""
    run(DictConfig(cfg))
