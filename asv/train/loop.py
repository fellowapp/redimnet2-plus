"""Основной цикл обучения на базе Hugging Face Accelerate (FSDP bf16).

Сюда заходит уже готовая модель + лосс. Функция :func:`train` собирает:
- DataLoader на основе :class:`asv.data.ParquetAudioDataset` + ``collate_waveform_batch``;
- AdamW + cosine scheduler с ``eta_min = optim.final_lr``;
- Accelerator с ``mixed_precision`` и ``gradient_accumulation_steps``;
- (опционально) возобновляет трейн из Accelerate-чекпоинта (``resume_from_checkpoint``);
- запускает валидацию (VoxCeleb1 EER) и сохраняет чекпоинты по расписанию
  (``train.save_every_epochs`` и/или ``validation.every_n_steps``).

Форма батча: ``{"inputs": waveform [B,1,T], "labels": [B]}``. В цикле waveform
проходит через spec-frontend (mel) и ``SpecAugment`` в :func:`_preprocess_training_batch`,
и только потом идёт в модель (энкодер ожидает mel-спектрограммы).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from asv.data import CodecAugmentProbabilities, ParquetAudioDataset, SpecAugmentConfig, collate_waveform_batch
from asv.data.collate import clone_frozen_frontend, compute_spectrogram_batch
from asv.logging.mlop import finish_mlop_run, init_mlop_logger, log_mlop_commit_artifact, log_mlop_metrics
from asv.train.spec_augment import build_spec_augment_cfg
from asv.validation import (
    VoxCelebEERValidator,
    VoxCelebVariantEERValidator,
    collect_variant_validation_roots,
    resolve_variant_validation_trial_files,
)
from asv.validation.voxceleb import compute_eer


def _build_train_loader(
    cfg: DictConfig,
) -> tuple[ParquetAudioDataset, DataLoader]:
    """Собирает датасет из Parquet-манифеста и DataLoader для трейна.

    Особенности:
    - ``shuffle=True``, ``drop_last=True`` — стандарт для multi-GPU трейна.
    - ``persistent_workers=True`` при ``num_workers > 0`` — чтобы не переинициализировать
      воркеры каждую эпоху (на больших датасетах это заметно).
    - ``collate_fn = collate_waveform_batch`` — форма ``[B,1,T]`` + ``[B]``.
    - Mel-фронтенд в цикле применяется отдельно (``compute_spectrogram_batch``)
      уже после prefetch/HtoD, поэтому в коллаторе спектрограммы не считаем.
    """
    num_workers = int(cfg.data.num_workers)
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg.data.batch_size),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.data.pin_memory),
        "drop_last": True,
        "persistent_workers": num_workers > 0,
        "collate_fn": None,
    }
    if num_workers > 0 and cfg.data.get("prefetch_factor") is not None:
        loader_kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)

    probabilities = CodecAugmentProbabilities(**OmegaConf.to_container(cfg.data.transform_probabilities, resolve=True))
    dataset = ParquetAudioDataset(
        manifest_path=cfg.data.manifest_path,
        sample_rate=int(cfg.data.sample_rate),
        trim_num_samples=cfg.data.trim_num_samples,
        codec_preset=cfg.data.codec_preset,
        augment_cfg=cfg.augment,
        probabilities=probabilities,
        fallback_on_decode_error=bool(cfg.train.fallback_on_decode_error),
    )
    loader_kwargs["collate_fn"] = collate_waveform_batch
    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, loader


def _preprocess_training_batch(
    batch: dict[str, torch.Tensor],
    spec_frontend: torch.nn.Module,
    spec_augment_cfg: SpecAugmentConfig,
) -> dict[str, torch.Tensor]:
    return {
        "inputs": compute_spectrogram_batch(
            batch["inputs"],
            spec_frontend,
            augment_cfg=spec_augment_cfg,
            training=True,
        ),
        "labels": batch["labels"],
    }


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if str(cfg.optim.name).lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
            betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
            eps=float(cfg.optim.eps),
        )
    raise ValueError(f"Unsupported optimizer '{cfg.optim.name}'")


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _unwrap_train_core(model: torch.nn.Module) -> torch.nn.Module:
    core = model
    if hasattr(core, "_orig_mod"):
        core = core._orig_mod
    return core


def _param_fingerprint(accelerator: Accelerator, model: torch.nn.Module) -> float:
    """Cheap scalar signature of weights for checkpoint load verification.

    FSDP/DDP: only rank 0 sees full (or meaningful) tensors via unwrap_model; other ranks
    often have empty parameter placeholders — skip fingerprint there entirely.
    """
    if not accelerator.is_main_process:
        return 0.0
    core = accelerator.unwrap_model(model)
    core = _unwrap_train_core(core)
    parts: list[torch.Tensor] = []
    for p in core.parameters():
        flat = p.detach().float().flatten()
        take = min(256, flat.numel())
        if take <= 0:
            continue
        parts.append(flat[:take])
        if len(parts) >= 3:
            break
    if not parts:
        raise RuntimeError("Checkpoint fingerprint: model has no non-empty parameters (unexpected for training).")
    cat = torch.cat(parts)
    return float(cat.sum().item() + cat.norm().item())


def _write_training_state(
    *,
    ckpt_dir: Path,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    next_epoch: int,
    save_dir: str,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "next_epoch": next_epoch,
        "save_dir": save_dir,
    }
    (ckpt_dir / "training_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _checkpoint_subdir_name(epoch: int, global_step: int) -> str:
    """Unique folder per save (epoch + global_step); avoids collisions for mid-epoch checkpoints."""
    return f"epoch_{epoch:04d}_step_{global_step:08d}"


def _save_checkpoint(
    accelerator: Accelerator,
    cfg: DictConfig,
    *,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    next_epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    subdir = _checkpoint_subdir_name(epoch, global_step)
    ckpt_dir = Path(cfg.train.checkpoint_dir) / subdir
    accelerator.save_state(str(ckpt_dir))
    if accelerator.is_main_process:
        latest = Path(cfg.train.checkpoint_dir) / "latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "next_epoch": next_epoch,
            "save_dir": subdir,
        }
        latest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _write_training_state(
            ckpt_dir=ckpt_dir,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            next_epoch=next_epoch,
            save_dir=subdir,
        )
        logger.info("Saved checkpoint to {}", ckpt_dir)


def _resolve_resume_checkpoint_dir(cfg: DictConfig) -> Path | None:
    raw = cfg.train.get("resume_from_checkpoint")
    if raw is None:
        return None
    token = str(raw).strip()
    if not token or token.lower() in {"none", "null", "~"}:
        return None
    if token.lower() == "latest":
        ckpt_root = Path(cfg.train.checkpoint_dir).expanduser().resolve()
        latest = ckpt_root / "latest.json"
        if not latest.is_file():
            raise FileNotFoundError(f"resume_from_checkpoint=latest but missing {latest}")
        meta = json.loads(latest.read_text(encoding="utf-8"))
        save_dir = meta.get("save_dir")
        if isinstance(save_dir, str) and save_dir.strip():
            candidate = ckpt_root / save_dir.strip()
        else:
            epoch = int(meta["epoch"])
            gs = int(meta.get("global_step", 0))
            candidate = ckpt_root / _checkpoint_subdir_name(epoch, gs)
        if candidate.is_dir():
            return candidate
        legacy = ckpt_root / f"epoch_{int(meta['epoch']):04d}"
        if legacy.is_dir():
            return legacy
        raise FileNotFoundError(
            f"resume_from_checkpoint=latest: expected checkpoint at {candidate} "
            f"(or legacy {legacy}), but neither directory exists."
        )
    path = Path(token).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path


def _checkpoint_has_scheduler_state(ckpt_dir: Path) -> bool:
    return (ckpt_dir / "scheduler.bin").is_file()


def _read_resume_training_state(ckpt_dir: Path) -> dict[str, int]:
    state_path = ckpt_dir / "training_state.json"
    if state_path.is_file():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return {
            "next_epoch": int(data["next_epoch"]),
            "global_step": int(data["global_step"]),
            "optimizer_step": int(data["optimizer_step"]),
        }
    raise FileNotFoundError(
        f"Missing {state_path.name} under {ckpt_dir}. "
        "Re-save checkpoints with the current training code or pass a directory created by it."
    )


def _maybe_resume_from_checkpoint(
    accelerator: Accelerator,
    cfg: DictConfig,
    model: torch.nn.Module,
    *,
    ckpt_dir: Path | None = None,
) -> tuple[int, int, int]:
    """Returns (start_epoch, global_step, optimizer_step) for the training loop."""
    if ckpt_dir is None:
        ckpt_dir = _resolve_resume_checkpoint_dir(cfg)
    if ckpt_dir is None:
        return 1, 0, 0
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"resume_from_checkpoint path is not a directory: {ckpt_dir}")
    input_dir = str(ckpt_dir)
    if accelerator.is_main_process:
        logger.info("Resuming from checkpoint directory: {}", ckpt_dir)
    state = _read_resume_training_state(ckpt_dir)
    fp_before = _param_fingerprint(accelerator, model)
    accelerator.load_state(input_dir)
    accelerator.wait_for_everyone()
    fp_after = _param_fingerprint(accelerator, model)
    if bool(cfg.train.verify_checkpoint_load) and accelerator.is_main_process:
        if fp_before == fp_after or math.isclose(fp_before, fp_after, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                "Checkpoint load verification failed: model parameters unchanged after load_state. "
                f"Fingerprint before={fp_before!r} after={fp_after!r}. "
                "Check that the checkpoint directory contains a full Accelerate save and matches this run's model."
            )
        logger.info(
            "Checkpoint load verified: parameter fingerprint changed (before={:.6g}, after={:.6g})",
            fp_before,
            fp_after,
        )
    if accelerator.is_main_process:
        logger.info(
            "Resumed training state: next_epoch={}, global_step={}, optimizer_step={}",
            state["next_epoch"],
            state["global_step"],
            state["optimizer_step"],
        )
    accelerator.wait_for_everyone()
    return state["next_epoch"], state["global_step"], state["optimizer_step"]


def _grad_norm(parameters: Any) -> float:
    norms = []
    for p in parameters:
        if p.grad is not None:
            norms.append(p.grad.detach().data.norm(2))
    if not norms:
        return 0.0
    stacked = torch.stack(
        [n if isinstance(n, torch.Tensor) else torch.tensor(float(n)) for n in norms]
    )
    return float(torch.norm(stacked, 2).item())


def _resolve_margin_controller(model: torch.nn.Module) -> Any | None:
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        return None
    if not hasattr(classifier, "set_margin_for_step") or not hasattr(
        classifier, "current_margin"
    ):
        return None
    return classifier


def _compute_total_margin_schedule_steps(
    train_loader: DataLoader,
    *,
    grad_accum_steps: int,
    max_epochs: int,
) -> int:
    steps_per_epoch = max(
        1, math.ceil(len(train_loader) / max(1, grad_accum_steps))
    )
    return steps_per_epoch * max(1, max_epochs)


def _compute_total_optimizer_steps(
    train_loader: DataLoader,
    *,
    grad_accum_steps: int,
    max_epochs: int,
) -> int:
    steps_per_epoch = max(
        1, math.ceil(len(train_loader) / max(1, grad_accum_steps))
    )
    return steps_per_epoch * max(1, max_epochs)


def _build_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    *,
    grad_accum_steps: int,
    max_epochs: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    start_lr = float(cfg.optim.lr)
    final_lr = float(cfg.optim.get("final_lr", start_lr))
    if final_lr < 0.0:
        raise ValueError(f"optim.final_lr must be >= 0, got {final_lr}")
    total_steps = _compute_total_optimizer_steps(
        train_loader,
        grad_accum_steps=grad_accum_steps,
        max_epochs=max_epochs,
    )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=final_lr,
    )


def _sync_scheduler_to_optimizer_step(
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    optimizer_step: int,
) -> None:
    if optimizer_step > 0:
        scheduler.step(optimizer_step)


def _maybe_update_margin(
    margin_controller: Any | None,
    *,
    step: int,
    total_steps: int,
) -> float | None:
    if margin_controller is None:
        return None
    return float(
        margin_controller.set_margin_for_step(step=step, total_steps=total_steps)
    )


def _init_variant_validator(
    cfg: DictConfig,
    accelerator: Accelerator,
    spec_frontend: torch.nn.Module,
) -> VoxCelebVariantEERValidator | None:
    vv = cfg.get("variant_validation")
    if vv is None or not bool(vv.enabled):
        return None
    roots = collect_variant_validation_roots(cfg)
    if not roots:
        if accelerator.is_main_process:
            logger.warning(
                "variant_validation.enabled=true but scan_root and roots are empty; skipping variant validation",
            )
        return None
    trial_files = resolve_variant_validation_trial_files(cfg)
    if not trial_files:
        if accelerator.is_main_process:
            logger.warning(
                "variant_validation.enabled=true but no trial files after resolving "
                "(set variant_validation.trial_files or include a stem ending in _o in validation.trial_files); skipping",
            )
        return None
    if accelerator.is_main_process:
        logger.info(
            "Variant validation enabled: {} dataset roots, trial_files={}",
            len(roots),
            trial_files,
        )
    return VoxCelebVariantEERValidator(
        cfg=cfg,
        accelerator=accelerator,
        spec_frontend=spec_frontend,
        roots=roots,
        trial_files=trial_files,
    )


def _init_validator(
    cfg: DictConfig,
    accelerator: Accelerator,
    spec_frontend: torch.nn.Module,
) -> VoxCelebEERValidator | None:
    validation_cfg = cfg.get("validation")
    if validation_cfg is None or not bool(validation_cfg.enabled):
        return None
    validator = VoxCelebEERValidator(cfg=cfg, accelerator=accelerator, spec_frontend=spec_frontend)
    if accelerator.is_main_process:
        logger.info(
            "Validation enabled: data_root='{}', trial_files={}, batch={}, workers={}, every_n_steps={}, before_first_epoch={}",
            validation_cfg.data_root,
            list(validation_cfg.trial_files),
            validation_cfg.batch_size,
            validation_cfg.num_workers,
            validation_cfg.every_n_steps,
            validation_cfg.run_before_first_epoch,
        )
    return validator


def _run_validation_coupled(
    variant_validator: VoxCelebVariantEERValidator | None,
    main_validator: VoxCelebEERValidator | None,
    model: torch.nn.Module,
    mlop_run: Any,
    *,
    tag: str,
    local_metrics_output_dir: str | Path | None = None,
) -> None:
    """Run variant validation first, then primary validation; log each stage separately."""
    if variant_validator is None and main_validator is None:
        return
    if variant_validator is not None:
        variant_metrics = variant_validator.run(model, tag=tag)
        if variant_metrics:
            log_mlop_metrics(mlop_run, variant_metrics)
            if local_metrics_output_dir is not None:
                _append_validation_metrics(
                    output_dir=local_metrics_output_dir,
                    tag=f"{tag}__variant_val",
                    metrics=variant_metrics,
                )
    if main_validator is not None:
        primary_metrics = main_validator.run(model, tag=tag)
        if primary_metrics:
            log_mlop_metrics(mlop_run, primary_metrics)
            if local_metrics_output_dir is not None:
                _append_validation_metrics(
                    output_dir=local_metrics_output_dir,
                    tag=f"{tag}__primary_val",
                    metrics=primary_metrics,
                )


def _append_validation_metrics(
    *,
    output_dir: str | Path,
    tag: str,
    metrics: dict[str, Any],
) -> None:
    path = Path(output_dir) / "validation_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _forward_and_loss(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    spec_frontend: torch.nn.Module | None = None,
    spec_augment_cfg: SpecAugmentConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec_frontend is not None:
        if spec_augment_cfg is None:
            raise ValueError("spec_augment_cfg is required when spec_frontend is provided")
        batch = _preprocess_training_batch(batch, spec_frontend, spec_augment_cfg)
    out = model(batch["inputs"], batch["labels"])
    if isinstance(out, tuple):
        logits, loss = out
    else:
        logits = out
        loss = loss_fn(logits, batch["labels"])
    return logits, loss


def _build_single_batch_negative_pairs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.size < 2:
        raise ValueError("Single-batch test requires at least two gathered samples")
    left_indices: list[int] = []
    right_indices: list[int] = []
    for idx, label in enumerate(labels.tolist()):
        for shift in range(1, labels.size):
            other_idx = (idx + shift) % labels.size
            if labels[other_idx] != label:
                left_indices.append(idx)
                right_indices.append(other_idx)
                break
    if not left_indices:
        raise ValueError("Single-batch test requires at least two distinct speaker labels in the gathered batch")
    return np.asarray(left_indices, dtype=np.int32), np.asarray(right_indices, dtype=np.int32)


def _compute_single_batch_eer_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    normalized = F.normalize(embeddings.detach().float(), dim=1).cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(np.int32, copy=False)
    neg_left, neg_right = _build_single_batch_negative_pairs(labels_np)

    positive_scores = np.sum(normalized * normalized, axis=1)
    negative_scores = np.sum(normalized[neg_left] * normalized[neg_right], axis=1)
    trial_labels = np.concatenate(
        [
            np.ones_like(positive_scores, dtype=np.int32),
            np.zeros_like(negative_scores, dtype=np.int32),
        ]
    )
    trial_scores = np.concatenate([positive_scores, negative_scores])
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    return {
        "val/single_batch/eer": compute_eer(trial_labels, trial_scores),
        "val/single_batch/positive_score_mean": float(positive_scores.mean()),
        "val/single_batch/negative_score_mean": float(negative_scores.mean()),
        "val/single_batch/num_unique_labels": float(unique_labels.size),
        "val/single_batch/max_label_count": float(counts.max()),
    }


def _build_single_batch_signature(
    accelerator: Accelerator,
    batch: dict[str, torch.Tensor],
) -> str:
    local_hasher = hashlib.sha1()
    local_hasher.update(batch["inputs"].detach().float().cpu().contiguous().numpy().tobytes())
    local_hasher.update(batch["labels"].detach().cpu().contiguous().numpy().tobytes())
    local_hash_value = int.from_bytes(local_hasher.digest()[:8], "big", signed=False) % (2**63 - 1)
    local_hash_tensor = torch.tensor([local_hash_value], device=batch["labels"].device, dtype=torch.long)
    gathered_hashes = accelerator.gather_for_metrics(local_hash_tensor).cpu().tolist()
    signature_payload = json.dumps(
        {
            "global_batch_size": int(accelerator.gather_for_metrics(batch["labels"]).shape[0]),
            "local_input_shape": [int(dim) for dim in batch["inputs"].shape],
            "rank_hashes": gathered_hashes,
        },
        sort_keys=True,
    )
    return hashlib.sha1(signature_payload.encode("utf-8")).hexdigest()[:16]


def _run_single_batch_test(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    train_loader: DataLoader,
    loss_fn: torch.nn.Module,
    spec_frontend: torch.nn.Module,
    spec_augment_cfg: SpecAugmentConfig,
    mlop_run: Any,
    max_grad_norm: float | None,
    margin_controller: Any | None,
    global_step: int,
    total_margin_steps: int,
    local_metrics_output_dir: str | Path | None,
) -> None:
    batch = next(iter(train_loader), None)
    if batch is None:
        raise RuntimeError(
            "Single-batch test could not fetch a training batch"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    current_margin = _maybe_update_margin(
        margin_controller,
        step=global_step,
        total_steps=total_margin_steps,
    )
    with accelerator.autocast():
        _, loss = _forward_and_loss(
            model,
            loss_fn,
            batch,
            spec_frontend=spec_frontend,
            spec_augment_cfg=spec_augment_cfg,
        )
    accelerator.backward(loss)
    grad_norm_value = _grad_norm(model.parameters())
    if max_grad_norm is not None:
        clipped_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        if clipped_norm is not None:
            grad_norm_value = float(clipped_norm.item() if hasattr(clipped_norm, "item") else clipped_norm)
    lr = _current_lr(optimizer)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    reduced_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
    processed_batch = _preprocess_training_batch(batch, spec_frontend, spec_augment_cfg)
    batch_size_total = int(
        accelerator.gather_for_metrics(processed_batch["labels"]).shape[0]
    )
    batch_signature = _build_single_batch_signature(accelerator, processed_batch)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        with accelerator.autocast():
            embeddings = model(processed_batch["inputs"], return_embeddings=True)
    if was_training:
        model.train()

    gathered_embeddings = accelerator.gather_for_metrics(embeddings.detach().float()).cpu()
    gathered_labels = accelerator.gather_for_metrics(processed_batch["labels"].detach()).cpu()
    validation_metrics = _compute_single_batch_eer_metrics(
        gathered_embeddings, gathered_labels
    )
    validation_metrics["val/single_batch/batch_index"] = 0
    validation_metrics["val/single_batch/global_batch_size"] = batch_size_total
    validation_metrics["val/single_batch/batch_signature"] = batch_signature

    if accelerator.is_main_process:
        if current_margin is None:
            logger.info(
                "Single-batch test: train/loss={:.6f} grad_norm={:.6f} lr={:.6g} val/single_batch/eer={:.6f} signature={}",
                reduced_loss,
                grad_norm_value,
                lr,
                validation_metrics["val/single_batch/eer"],
                batch_signature,
            )
        else:
            logger.info(
                "Single-batch test: train/loss={:.6f} grad_norm={:.6f} margin={:.6f} lr={:.6g} val/single_batch/eer={:.6f} signature={}",
                reduced_loss,
                grad_norm_value,
                current_margin,
                lr,
                validation_metrics["val/single_batch/eer"],
                batch_signature,
            )
        metrics = {
            "optimizer_step": 1,
            "train/single_batch/loss_before": reduced_loss,
            "train/single_batch/grad_norm": grad_norm_value,
            "train/single_batch/lr": lr,
            **validation_metrics,
        }
        if current_margin is not None:
            metrics["train/single_batch/margin"] = current_margin
        log_mlop_metrics(mlop_run, metrics)
        if local_metrics_output_dir is not None:
            _append_validation_metrics(
                output_dir=local_metrics_output_dir,
                tag="single_batch_test",
                metrics=validation_metrics,
            )

    tolerance = 1e-6
    if float(validation_metrics["val/single_batch/eer"]) > tolerance:
        raise RuntimeError(
            f"Single-batch test failed: expected val/single_batch/eer <= {tolerance}, "
            f"got {validation_metrics['val/single_batch/eer']:.6f}"
        )

    accelerator.wait_for_everyone()


def train(cfg: DictConfig, model: torch.nn.Module, loss_fn: torch.nn.Module) -> None:
    """Основной цикл трейна.

    Шаги:
    1. Поднимает :class:`Accelerator` с FSDP/DDP и mixed precision.
    2. Извлекает из модели mel-фронтенд (``model.encoder.spec_frontend``) — его клон
       крутится на каждом шаге ``compute_spectrogram_batch`` (eval + no_grad).
    3. Строит loader через :func:`_build_train_loader`, оптимизатор (AdamW) и
       cosine-scheduler по числу optimizer-шагов за всё обучение.
    4. Если задан ``train.resume_from_checkpoint`` — ресюмит Accelerate-стейт
       и проверяет, что веса реально изменились (``verify_checkpoint_load``).
    5. Прогоняет эпохи: forward → loss → backward → grad_clip → optimizer.step → scheduler.step.
    6. Запускает валидацию (VoxCeleb1 EER) по расписанию (``validation.every_n_steps``
       и в конце каждой эпохи); variant-валидация — через ``variant_validation``.
    7. Сохраняет чекпоинт на последней эпохе и на каждом валидационном шаге по шагам.
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.train.grad_accum_steps),
        mixed_precision=str(cfg.train.mixed_precision),
    )
    if accelerator.is_main_process:
        logger.info(
            "Accelerator created: processes={}, grad_accum_steps={}, mixed_precision={}",
            accelerator.num_processes,
            cfg.train.grad_accum_steps,
            cfg.train.mixed_precision,
        )
    set_seed(int(cfg.seed))
    if accelerator.is_main_process:
        logger.info(
            "Preparing train dataloader from manifest: {}",
            cfg.data.manifest_path,
        )
        logger.info("Training seed set to {}", cfg.seed)
    # Extract the mel-spectrogram frontend from the encoder for the collator.
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap torch.compile
    spec_frontend = raw_model.encoder.spec_frontend
    if accelerator.is_main_process:
        logger.info(
            "Model frontend extracted for collator: {}",
            type(spec_frontend).__name__,
        )
    dataset, train_loader = _build_train_loader(cfg)
    if accelerator.is_main_process:
        logger.info(
            "Dataset ready: {} samples, {} speakers, batch/gpu={}, workers/gpu={}",
            len(dataset),
            dataset.num_speakers,
            cfg.data.batch_size,
            cfg.data.num_workers,
        )

    max_epochs = int(cfg.train.epochs)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(
        cfg,
        optimizer,
        train_loader,
        grad_accum_steps=int(cfg.train.grad_accum_steps),
        max_epochs=max_epochs,
    )
    if accelerator.is_main_process:
        logger.info(
            "Optimizer initialized: name={}, lr={}, final_lr={}, weight_decay={}, betas={}",
            cfg.optim.name,
            cfg.optim.lr,
            cfg.optim.final_lr,
            cfg.optim.weight_decay,
            f"({cfg.optim.beta1}, {cfg.optim.beta2})",
        )
    model, optimizer, train_loader, loss_fn = accelerator.prepare(
        model, optimizer, train_loader, loss_fn
    )
    if accelerator.is_main_process:
        logger.info("Distributed objects prepared for training")
    resume_ckpt_dir = _resolve_resume_checkpoint_dir(cfg)
    resume_has_scheduler_state = (
        resume_ckpt_dir is not None
        and _checkpoint_has_scheduler_state(resume_ckpt_dir)
    )
    if resume_ckpt_dir is None or resume_has_scheduler_state:
        scheduler = accelerator.prepare_scheduler(scheduler)
    start_epoch, global_step, optimizer_step = _maybe_resume_from_checkpoint(
        accelerator,
        cfg,
        model,
        ckpt_dir=resume_ckpt_dir,
    )
    if resume_ckpt_dir is not None and not resume_has_scheduler_state:
        scheduler = accelerator.prepare_scheduler(scheduler)
        _sync_scheduler_to_optimizer_step(scheduler, optimizer_step)
    if accelerator.is_main_process:
        logger.info(
            "Cosine scheduler initialized: total_steps={}, start_lr={}, final_lr={}, current_lr={}, optimizer_step={}",
            _compute_total_optimizer_steps(
                train_loader,
                grad_accum_steps=int(cfg.train.grad_accum_steps),
                max_epochs=max_epochs,
            ),
            cfg.optim.lr,
            cfg.optim.final_lr,
            _current_lr(optimizer),
            optimizer_step,
        )
    train_spec_frontend = clone_frozen_frontend(spec_frontend, device=accelerator.device)
    spec_augment_cfg = build_spec_augment_cfg(cfg)
    mlop_run = None
    if accelerator.is_main_process:
        config_for_log = OmegaConf.to_container(cfg, resolve=True)
        mlop_run = init_mlop_logger(cfg, config_for_log=config_for_log)
        log_mlop_commit_artifact(
            mlop_run,
            commit_hash=str(cfg.train.git_commit_hash),
            output_dir=str(cfg.output_dir),
        )
    local_validation_metrics_dir = str(cfg.output_dir) if mlop_run is None else None

    save_every_epochs = int(cfg.train.save_every_epochs)
    max_grad_norm = (
        float(cfg.train.max_grad_norm)
        if cfg.train.max_grad_norm is not None
        else None
    )
    train_core = _unwrap_train_core(accelerator.unwrap_model(model))
    margin_controller = _resolve_margin_controller(train_core)
    total_margin_steps = _compute_total_margin_schedule_steps(
        train_loader,
        grad_accum_steps=int(cfg.train.grad_accum_steps),
        max_epochs=max_epochs,
    )
    if accelerator.is_main_process:
        logger.info(
            "Training started: epochs={}, grad_accum={}, mixed_precision={}, fsdp_expected={}",
            max_epochs,
            cfg.train.grad_accum_steps,
            cfg.train.mixed_precision,
            cfg.train.expect_fsdp,
        )

    log_every_steps = int(cfg.train.log_every_steps)
    val_every_n_steps = (
        int(cfg.validation.every_n_steps)
        if cfg.get("validation") is not None
        else 0
    )
    single_batch_test = _env_flag("SINGLE_BATCH_TEST")
    if single_batch_test:
        if accelerator.is_main_process:
            logger.info("Single-batch test mode enabled via SINGLE_BATCH_TEST")
        try:
            _run_single_batch_test(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                loss_fn=loss_fn,
                spec_frontend=train_spec_frontend,
                spec_augment_cfg=spec_augment_cfg,
                mlop_run=mlop_run,
                max_grad_norm=max_grad_norm,
                margin_controller=margin_controller,
                global_step=global_step,
                total_margin_steps=total_margin_steps,
                local_metrics_output_dir=local_validation_metrics_dir,
            )
        finally:
            if accelerator.is_main_process:
                finish_mlop_run(mlop_run)
        return
    validator = _init_validator(cfg, accelerator, spec_frontend)
    variant_validator = _init_variant_validator(cfg, accelerator, spec_frontend)
    if (
        (variant_validator is not None or validator is not None)
        and bool(cfg.validation.run_before_first_epoch)
    ):
        _run_validation_coupled(
            variant_validator,
            validator,
            model,
            mlop_run,
            tag="before_epoch_1",
            local_metrics_output_dir=local_validation_metrics_dir,
        )
    # raise RuntimeError("Stop here")
    if start_epoch > max_epochs and accelerator.is_main_process:
        logger.warning(
            "start_epoch={} is past train.epochs={}; training loop will not run",
            start_epoch,
            max_epochs,
        )
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running_loss = 0.0
        num_seen = 0
        ema_loss: float | None = None
        prog = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{max_epochs}",
            disable=not accelerator.is_local_main_process,
            leave=False,
        )
        for batch in prog:
            current_margin = _maybe_update_margin(
                margin_controller,
                step=global_step,
                total_steps=total_margin_steps,
            )
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    _, loss = _forward_and_loss(
                        model,
                        loss_fn,
                        batch,
                        spec_frontend=train_spec_frontend,
                        spec_augment_cfg=spec_augment_cfg,
                    )
                accelerator.backward(loss)
                grad_norm_value = _grad_norm(model.parameters())
                if max_grad_norm is not None and accelerator.sync_gradients:
                    clipped_norm = accelerator.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    if clipped_norm is not None:
                        grad_norm_value = float(
                            clipped_norm.item()
                            if hasattr(clipped_norm, "item")
                            else clipped_norm
                        )
                lr = _current_lr(optimizer)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    optimizer_step += 1

            reduced_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
            batch_size_total = int(batch["labels"].shape[0]) * accelerator.num_processes
            running_loss += reduced_loss * batch_size_total
            num_seen += batch_size_total
            ema_loss = (
                reduced_loss
                if ema_loss is None
                else (0.98 * ema_loss + 0.02 * reduced_loss)
            )

            if accelerator.is_local_main_process:
                prog.set_postfix(loss_ema=f"{ema_loss:.4f}")

            if accelerator.sync_gradients:
                global_step += 1

            if accelerator.is_main_process and accelerator.sync_gradients:
                if (
                    log_every_steps <= 0
                    or global_step == 1
                    or global_step % log_every_steps == 0
                ):
                    if current_margin is None:
                        logger.info(
                            "epoch={} opt_step={} step={} train/loss={:.6f} grad_norm={:.6f} lr={:.6g}",
                            epoch,
                            optimizer_step,
                            global_step,
                            reduced_loss,
                            grad_norm_value,
                            lr,
                        )
                    else:
                        logger.info(
                            "epoch={} opt_step={} step={} train/loss={:.6f} grad_norm={:.6f} margin={:.6f} lr={:.6g}",
                            epoch,
                            optimizer_step,
                            global_step,
                            reduced_loss,
                            grad_norm_value,
                            current_margin,
                            lr,
                        )
                    metrics = {
                        "optimizer_step": optimizer_step,
                        "train/loss": reduced_loss,
                        "train/grad_norm": grad_norm_value,
                        "train/lr": lr,
                    }
                    if current_margin is not None:
                        metrics["train/margin"] = current_margin
                    log_mlop_metrics(mlop_run, metrics)
            if (
                (variant_validator is not None or validator is not None)
                and accelerator.sync_gradients
                and val_every_n_steps > 0
                and global_step % val_every_n_steps == 0
            ):
                _run_validation_coupled(
                    variant_validator,
                    validator,
                    model,
                    mlop_run,
                    tag=f"epoch_{epoch:03d}_step_{global_step:08d}",
                    local_metrics_output_dir=local_validation_metrics_dir,
                )
                _save_checkpoint(
                    accelerator,
                    cfg,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    next_epoch=epoch,
                )

        epoch_loss = running_loss / max(1, num_seen)
        if accelerator.is_main_process:
            logger.info("Epoch {} finished: train/loss={:.6f}", epoch, epoch_loss)
            log_mlop_metrics(mlop_run, {"train/epoch_loss": epoch_loss})

        _run_validation_coupled(
            variant_validator,
            validator,
            model,
            mlop_run,
            tag=f"epoch_{epoch:03d}_end",
            local_metrics_output_dir=local_validation_metrics_dir,
        )

        if save_every_epochs > 0 and epoch % save_every_epochs == 0:
            _save_checkpoint(
                accelerator,
                cfg,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                next_epoch=epoch + 1,
            )

    if accelerator.is_main_process:
        logger.info("Training completed at step {}", global_step)
        finish_mlop_run(mlop_run)