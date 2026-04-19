"""Валидация на VoxCeleb (multi-GPU расчёт эмбеддингов + EER).

:class:`VoxCelebEERValidator` гоняет все уникальные utterance'ы из trial-файлов
через модель, собирает эмбеддинги на main-процессе (через ``gather_for_metrics``),
считает cosine-similarity по парам и выдаёт Equal Error Rate.

:class:`VoxCelebVariantEERValidator` делает то же самое, но на наборе «вариантных»
директорий (например, аугментированных версий VoxCeleb) — позволяет оценить
устойчивость модели к codec/шуму/reverb. Плюс считает pooled_hard EER — общий EER
по всем вариантам."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from asv.data.collate import clone_frozen_frontend, compute_spectrogram_batch, pad_random
from asv.data.decode import load_with_torchcodec

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrialSetup:
    """One VoxCeleb trial setup file with parsed labels and utterance IDs."""

    name: str
    labels: np.ndarray
    utt1_ids: np.ndarray
    utt2_ids: np.ndarray


class VoxCelebUtteranceDataset(Dataset[dict[str, Any]]):
    """Decode unique utterances referenced by trial files."""

    def __init__(self, utt_rel_paths: list[str], data_root: str | Path, sample_rate: int, trim_num_samples: int | None):
        self._utt_rel_paths = utt_rel_paths
        self._data_root = Path(data_root).expanduser().resolve()
        self._sample_rate = int(sample_rate)
        self._trim_num_samples = int(trim_num_samples) if trim_num_samples is not None else None

    def __len__(self) -> int:
        return len(self._utt_rel_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rel_path = self._utt_rel_paths[index]
        wav_path = self._data_root / rel_path
        waveform, _ = load_with_torchcodec(str(wav_path), sample_rate=self._sample_rate)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected waveform with shape [T] or [C,T], got {tuple(waveform.shape)} for {wav_path}")
        if self._trim_num_samples is not None:
            waveform = pad_random(waveform, self._trim_num_samples)
        return {"waveform": waveform, "utt_id": index}


class ValidationWaveformCollator:
    """Batch validation waveforms and utterance IDs for on-device preprocessing."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        waveforms = torch.stack([item["waveform"] for item in batch], dim=0).unsqueeze(1)
        utt_ids = torch.tensor([int(item["utt_id"]) for item in batch], dtype=torch.long)
        return {"inputs": waveforms, "utt_ids": utt_ids}


def _sanitize_variant_eer_suffix(name: str) -> str:
    """Make variant directory names safe as a single metric key segment."""
    cleaned = name.replace("/", "_").replace(" ", "_").replace(".", "_")
    return cleaned or "unknown"


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """EER (Equal Error Rate) по бинарным меткам и скорам косинусной близости.

    EER — точка, где FAR (false accept rate) равен FRR (false reject rate).
    Реализация: сортируем скоры по убыванию, считаем кумулятивные TP/FP, ищем индекс,
    где ``|FAR - FRR|`` минимален, возвращаем среднее между FAR и FRR в этой точке."""
    if labels.size == 0:
        raise ValueError("Cannot compute EER on empty labels")
    labels = labels.astype(np.int32, copy=False)
    scores = scores.astype(np.float64, copy=False)
    num_pos = int(labels.sum())
    num_neg = int(labels.size - num_pos)
    if num_pos == 0 or num_neg == 0:
        raise ValueError("EER requires both positive and negative trials")

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, dtype=np.int64)
    fp = np.cumsum(1 - sorted_labels, dtype=np.int64)

    fn = num_pos - tp
    far = fp / float(num_neg)
    frr = fn / float(num_pos)
    best_idx = int(np.argmin(np.abs(far - frr)))
    return float((far[best_idx] + frr[best_idx]) * 0.5)


def _resolve_utterance_path(root: Path, utt_rel_path: str) -> tuple[Path, bool]:
    rel = Path(utt_rel_path)
    direct_path = root / rel
    if direct_path.exists():
        return direct_path, True
    if rel.parts and rel.parts[0] == "wav":
        nested_path = root / Path(*rel.parts[1:])
        if nested_path.exists():
            return nested_path, True
    return direct_path, False


def parse_trial_setups(trial_files: list[str], data_root: str | Path) -> tuple[list[str], dict[str, int], list[TrialSetup]]:
    """Parse multiple VoxCeleb trial files.

    Returns:
      - Sorted unique utterance relative paths
      - Mapping from relative path -> utterance ID
      - Parsed trial setup structures
    """
    root = Path(data_root).expanduser().resolve()
    if not trial_files:
        raise ValueError("validation.trial_files must not be empty when validation is enabled")

    raw_rows: dict[str, list[tuple[int, str, str]]] = {}
    all_paths: set[str] = set()
    for trial_file in trial_files:
        path = Path(trial_file)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Trial file not found: {path}")
        rows: list[tuple[int, str, str]] = []
        skipped_rows = 0
        missing_examples: list[str] = []
        for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed trial line in {path}:{line_num} -> '{line}'")
            label = int(parts[0])
            if label not in (0, 1):
                raise ValueError(f"Trial label must be 0/1 in {path}:{line_num}, got {label}")
            utt1, utt2 = parts[1], parts[2]
            path1, has_utt1 = _resolve_utterance_path(root, utt1)
            path2, has_utt2 = _resolve_utterance_path(root, utt2)
            if not has_utt1 or not has_utt2:
                skipped_rows += 1
                if len(missing_examples) < 5:
                    if not has_utt1:
                        missing_examples.append(f"{utt1}->{path1}")
                    if not has_utt2:
                        missing_examples.append(f"{utt2}->{path2}")
                continue
            all_paths.add(utt1)
            all_paths.add(utt2)
            rows.append((label, utt1, utt2))
        if skipped_rows:
            LOG.warning(
                "Skipping %d trial rows in %s due to missing utterance files (examples: %s)",
                skipped_rows,
                path,
                ", ".join(missing_examples) if missing_examples else "none",
            )
        if not rows:
            LOG.warning("Skipping trial file with no usable rows: %s", path)
            continue
        raw_rows[path.stem] = rows

    if not raw_rows:
        raise ValueError("No usable trial rows found after filtering missing utterances in validation files")

    utt_rel_paths = sorted(all_paths)
    utt_to_id = {path: idx for idx, path in enumerate(utt_rel_paths)}

    setups: list[TrialSetup] = []
    for setup_name, rows in raw_rows.items():
        labels = np.asarray([row[0] for row in rows], dtype=np.int32)
        utt1_ids = np.asarray([utt_to_id[row[1]] for row in rows], dtype=np.int32)
        utt2_ids = np.asarray([utt_to_id[row[2]] for row in rows], dtype=np.int32)
        setups.append(TrialSetup(name=setup_name, labels=labels, utt1_ids=utt1_ids, utt2_ids=utt2_ids))
    return utt_rel_paths, utt_to_id, setups


class VoxCelebEERValidator:
    """Считает EER для одного или нескольких VoxCeleb trial-сетапов.

    Один проход по всем уникальным utterance'ам → L2-нормализованные эмбеддинги →
    dot-product по парам из триалов → EER на каждый trial-файл + pooled."""

    def __init__(self, cfg: Any, accelerator: Accelerator, spec_frontend: nn.Module):
        self._cfg = cfg
        self._accelerator = accelerator
        self._sample_rate = int(cfg.data.sample_rate)
        self._trim_num_samples = cfg.validation.trim_num_samples
        self._spec_frontend = clone_frozen_frontend(spec_frontend, device=self._accelerator.device)
        trial_files = [str(item) for item in cfg.validation.trial_files]
        self._utt_rel_paths, _, self._trial_setups = parse_trial_setups(trial_files=trial_files, data_root=str(cfg.validation.data_root))
        self._dataset = VoxCelebUtteranceDataset(
            utt_rel_paths=self._utt_rel_paths,
            data_root=str(cfg.validation.data_root),
            sample_rate=self._sample_rate,
            trim_num_samples=self._trim_num_samples,
        )
        num_workers = int(self._cfg.validation.num_workers)
        loader_kwargs: dict[str, Any] = {
            "batch_size": int(self._cfg.validation.batch_size),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": bool(self._cfg.validation.pin_memory),
            "drop_last": False,
            "persistent_workers": False,
            "collate_fn": ValidationWaveformCollator(),
        }
        if num_workers > 0 and self._cfg.validation.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(self._cfg.validation.prefetch_factor)
        loader = DataLoader(self._dataset, **loader_kwargs)
        self._loader = self._accelerator.prepare(loader)

    def _compute_embeddings(self, model: torch.nn.Module) -> torch.Tensor | None:
        embeddings_by_id: dict[int, torch.Tensor] = {}
        progress = tqdm(
            self._loader,
            desc="validation/embeddings",
            disable=not self._accelerator.is_local_main_process,
            leave=False,
        )
        with torch.no_grad():
            for batch in progress:
                with self._accelerator.autocast():
                    inputs = compute_spectrogram_batch(batch["inputs"], self._spec_frontend, training=False)
                    # Call the wrapped model's forward path so FSDP/DDP hooks run
                    # before encoder weights are touched.
                    embeds = model(inputs, return_embeddings=True)
                embeds = F.normalize(embeds.detach().float(), dim=1)
                gathered_embeds = self._accelerator.gather_for_metrics(embeds).cpu()
                gathered_ids = self._accelerator.gather_for_metrics(batch["utt_ids"]).cpu()
                if self._accelerator.is_main_process:
                    for idx in range(gathered_ids.shape[0]):
                        utt_id = int(gathered_ids[idx].item())
                        if utt_id not in embeddings_by_id:
                            embeddings_by_id[utt_id] = gathered_embeds[idx]

        if not self._accelerator.is_main_process:
            return None
        ordered = [embeddings_by_id[i] for i in range(len(self._utt_rel_paths))]
        return torch.stack(ordered, dim=0)

    def _pooled_labels_and_scores(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pooled_labels: list[np.ndarray] = []
        pooled_scores: list[np.ndarray] = []
        for setup in self._trial_setups:
            score_vec = np.sum(emb[setup.utt1_ids] * emb[setup.utt2_ids], axis=1)
            pooled_labels.append(setup.labels)
            pooled_scores.append(score_vec)
        return np.concatenate(pooled_labels), np.concatenate(pooled_scores)

    def _pooled_eer_from_embedding_matrix(self, emb: np.ndarray) -> float:
        labels, scores = self._pooled_labels_and_scores(emb)
        return compute_eer(labels, scores)

    def _eer_metrics_dict(self, emb: np.ndarray, prefix: str) -> tuple[dict[str, float], float]:
        metrics: dict[str, float] = {}
        for setup in self._trial_setups:
            score_vec = np.sum(emb[setup.utt1_ids] * emb[setup.utt2_ids], axis=1)
            metrics[f"{prefix}/{setup.name}"] = compute_eer(setup.labels, score_vec)
        pooled_eer = self._pooled_eer_from_embedding_matrix(emb)
        metrics[f"{prefix}/pooled"] = pooled_eer
        return metrics, pooled_eer

    def run_pooled_labels_and_scores(self, model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray] | None:
        """Embedding pass + pooled trial labels and scores (main process only)."""
        model.eval()
        embedding_matrix = self._compute_embeddings(model)
        self._accelerator.wait_for_everyone()
        if not self._accelerator.is_main_process:
            model.train()
            return None
        if embedding_matrix is None:
            raise RuntimeError("Main process did not receive validation embeddings")
        labels, scores = self._pooled_labels_and_scores(embedding_matrix.numpy())
        model.train()
        return labels, scores

    def run_pooled_eer_only(self, model: torch.nn.Module) -> float | None:
        """Embedding pass + pooled EER only (no per-trial-file metrics). Main process only."""
        result = self.run_pooled_labels_and_scores(model)
        if result is None:
            return None
        labels, scores = result
        pooled_eer = compute_eer(labels, scores)
        return pooled_eer

    def run(self, model: torch.nn.Module, *, tag: str, metrics_key_prefix: str = "val/eer") -> dict[str, float]:
        model.eval()
        embedding_matrix = self._compute_embeddings(model)
        self._accelerator.wait_for_everyone()

        if not self._accelerator.is_main_process:
            model.train()
            return {}

        if embedding_matrix is None:
            raise RuntimeError("Main process did not receive validation embeddings")
        prefix = metrics_key_prefix.rstrip("/")
        metrics, pooled_eer = self._eer_metrics_dict(embedding_matrix.numpy(), prefix)

        model.train()
        LOG.info("Validation %s complete (%s): pooled EER=%.6f", tag, prefix, pooled_eer)
        return metrics


def resolve_variant_validation_trial_files(cfg: Any) -> list[str]:
    """Trial paths used for augmentation/codec variant validation.

    If ``variant_validation.trial_files`` is non-empty, returns those paths. Otherwise
    returns only entries from ``validation.trial_files`` whose basename stem ends with
    ``_o`` (e.g. ``voxceleb1_o.txt``), matching datasets built from the original trial
    list only. Main validation still uses the full ``validation.trial_files`` list.
    """
    vv = cfg.get("variant_validation")
    if vv is None:
        return []
    if not OmegaConf.is_config(vv):
        vv = OmegaConf.create(vv)
    explicit: Sequence[Any] = vv.get("trial_files") or []
    if explicit:
        return [str(p) for p in explicit]
    val = cfg.get("validation")
    if val is None:
        return []
    trial_files = val.get("trial_files") if OmegaConf.is_config(val) else (val or {}).get("trial_files")
    if not trial_files:
        return []
    return [str(p) for p in trial_files if Path(str(p)).stem.endswith("_o")]


def collect_variant_validation_roots(cfg: Any) -> list[tuple[str, str]]:
    """Return ``(dir_name, resolved_path)`` for each variant dataset root.

    ``variant_validation.roots`` entries are listed first (deduped), then
    non-hidden subdirectories of ``variant_validation.scan_root`` (sorted).
    """
    vv = cfg.get("variant_validation")
    if vv is None:
        return []
    if not OmegaConf.is_config(vv):
        vv = OmegaConf.create(vv)
    roots_out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in list(vv.get("roots", []) or []):
        p = Path(str(r)).expanduser().resolve()
        if not p.is_dir():
            LOG.warning("variant_validation.roots entry is not a directory: %s", p)
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        roots_out.append((p.name, key))
    scan = str(vv.get("scan_root", "") or "").strip()
    if scan:
        scan_p = Path(scan).expanduser().resolve()
        if not scan_p.is_dir():
            LOG.warning("variant_validation.scan_root is not a directory: %s", scan_p)
        else:
            for child in sorted(scan_p.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                key = str(child.resolve())
                if key in seen:
                    continue
                seen.add(key)
                roots_out.append((child.name, key))
    return roots_out


def _cfg_with_validation_overrides(
    cfg: DictConfig | dict[str, Any],
    *,
    data_root: str,
    trial_files: list[str],
) -> DictConfig:
    """Deep-copy ``cfg`` with ``validation.data_root`` and ``trial_files`` replaced."""
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    val = dict(resolved.get("validation") or {})
    val["data_root"] = data_root
    val["trial_files"] = list(trial_files)
    resolved["validation"] = val
    return OmegaConf.create(resolved)


class VoxCelebVariantEERValidator:
    """Runs :class:`VoxCelebEERValidator` on multiple trial-audio variant trees.

    Each root must mirror the layout under ``validation.data_root`` (relative trial
    paths resolve the same way as for clean VoxCeleb wav).
    """

    def __init__(
        self,
        cfg: Any,
        accelerator: Accelerator,
        spec_frontend: nn.Module,
        *,
        roots: list[tuple[str, str]],
        trial_files: list[str],
    ) -> None:
        if not roots:
            raise ValueError("VoxCelebVariantEERValidator requires non-empty roots")
        if not trial_files:
            raise ValueError("VoxCelebVariantEERValidator requires non-empty trial_files")
        self._accelerator = accelerator
        self._validators: list[tuple[str, VoxCelebEERValidator]] = []
        for name, root in roots:
            sub_cfg = _cfg_with_validation_overrides(cfg, data_root=root, trial_files=trial_files)
            self._validators.append((name, VoxCelebEERValidator(sub_cfg, accelerator, spec_frontend)))

    def run(self, model: torch.nn.Module, *, tag: str) -> dict[str, float]:
        """One metric per variant root + global hard-pool EER."""
        metrics: dict[str, float] = {}
        pooled_labels: list[np.ndarray] = []
        pooled_scores: list[np.ndarray] = []
        for name, validator in self._validators:
            result = validator.run_pooled_labels_and_scores(model)
            if result is not None:
                labels, scores = result
                pooled = compute_eer(labels, scores)
                suffix = _sanitize_variant_eer_suffix(name)
                metrics[f"val/variants/eer_{suffix}"] = pooled
                LOG.info("Variant validation %s [%s]: pooled EER=%.6f", tag, name, pooled)
                pooled_labels.append(labels)
                pooled_scores.append(scores)
            else:
                LOG.debug("Variant validation %s [%s]: skipped on non-main process", tag, name)
        if pooled_labels:
            all_labels = np.concatenate(pooled_labels)
            all_scores = np.concatenate(pooled_scores)
            metrics["val/eer/pooled_hard"] = compute_eer(all_labels, all_scores)
            LOG.info("Variant validation %s: pooled hard EER=%.6f", tag, metrics["val/eer/pooled_hard"])
        return metrics