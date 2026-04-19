"""Parquet-датасет с mmap-чтением + политики применения кодеков и аугментаций.

Основной класс — :class:`ParquetAudioDataset`. На каждый запрос ``__getitem__`` он:
1. По индексу находит нужный ``(row_group, row_index)`` и читает одну строку манифеста.
2. Декодирует аудио через :func:`asv.data.decode.load_with_torchcodec` (со случайным
   окном длины ``trim_num_samples`` — на длинных файлах не читаем весь файл).
3. Выбирает режим обработки через :class:`CodecAugmentProbabilities`:
   ``keep`` / ``codec`` / ``augment`` / ``codec_augment``.
4. При необходимости прогоняет кодек-roundtrip (см. :mod:`asv.codec.pipeline`) и/или
   аугментации waveform-уровня (см. :mod:`asv.augment.pipeline`).
5. Приводит к моно и pad/trim'ит к ``trim_num_samples``.

Если декод упал — ищем ближайший валидный индекс (при ``fallback_on_decode_error=True``),
чтобы один битый файл не клал воркер."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import Dataset

from asv.augment import AudioAugmentPipeline, build_audio_augmentor
from asv.codec import CodecPipeline, check_preset_available, list_presets
from asv.data.collate import pad_random
from asv.data.decode import load_with_torchcodec

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodecAugmentProbabilities:
    """Вероятности режимов обработки одного сэмпла в ``__getitem__``.

    Сумма четырёх полей обязана быть равна 1.0 (с точностью до 1e-6), иначе
    конструктор упадёт. Режимы:
    - ``keep_original`` — ничего не делаем, аудио идёт как прочитано;
    - ``codec_only`` — только кодек-roundtrip (lossy compression);
    - ``augment_only`` — только waveform-аугментации (фильтры/шум/RIR/perturb);
    - ``codec_then_augment`` — сначала кодек, потом аугментации.

    В дефолтных конфигах эти вероятности промаркированы как эксперимент-лок —
    трогать только руками (см. ``AGENTS.md``)."""

    keep_original: float
    codec_only: float
    augment_only: float
    codec_then_augment: float

    def __post_init__(self) -> None:
        total = self.keep_original + self.codec_only + self.augment_only + self.codec_then_augment
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Probabilities must sum to 1.0; "
                f"got {total:.6f} from "
                f"{self.keep_original}, {self.codec_only}, {self.augment_only}, {self.codec_then_augment}"
            )
        for value in (
            self.keep_original,
            self.codec_only,
            self.augment_only,
            self.codec_then_augment,
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Probability must be in [0,1], got {value}")

    def choose_mode(self) -> str:
        r = random.random()
        b0 = self.keep_original
        b1 = b0 + self.codec_only
        b2 = b1 + self.augment_only
        if r < b0:
            return "keep"
        if r < b1:
            return "codec"
        if r < b2:
            return "augment"
        return "codec_augment"


class ParquetAudioDataset(Dataset[dict[str, Any]]):
    """Датасет из Parquet-манифеста с ленивым mmap-чтением по row-группам.

    При инициализации строит индекс ``(row_group, row_index)`` для O(1) доступа по idx
    и маппинг ``speaker_id -> label`` (сортировкой по строкам, чтобы был детерминизм).
    На каждом ``__getitem__`` кэширует целиком один row-group — это снижает I/O,
    если шаффл попадает в одну группу подряд.

    Параметры:
    - ``sample_rate`` — целевая частота (torchcodec ресемплит при необходимости).
    - ``trim_num_samples`` — жёсткая длина: короче — repeat-pad, длиннее — случайный crop.
    - ``codec_preset`` — если задан, все кодек-ветки используют этот preset; иначе
      выбирается случайно из доступных в системе (`list_presets()` + `check_preset_available`).
    - ``codec_min_duration_sec`` — не применять кодек к слишком коротким файлам.
    - ``augment_cfg`` — конфиг аугментаций (см. :mod:`asv.augment.pipeline`).
    - ``probabilities`` — вероятности режимов (см. :class:`CodecAugmentProbabilities`).
    - ``fallback_on_decode_error`` — при ошибке декода пробовать соседние индексы."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16000,
        trim_num_samples: int | None = None,
        codec_preset: str | None = None,
        codec_backend: str = "auto",
        codec_min_duration_sec: float | None = None,
        augment_cfg: DictConfig | dict[str, Any] | None = None,
        probabilities: CodecAugmentProbabilities | None = None,
        fallback_on_decode_error: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.trim_num_samples = trim_num_samples
        self.codec_min_duration_sec = codec_min_duration_sec
        self.fallback_on_decode_error = fallback_on_decode_error
        self._codec_backend = codec_backend
        self.probabilities = probabilities or CodecAugmentProbabilities(
            keep_original=1.0,
            codec_only=0.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        )

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.parquet_file = pq.ParquetFile(self.manifest_path, memory_map=True)
        self.index: list[tuple[int, int]] = []
        for rg in range(self.parquet_file.num_row_groups):
            n = self.parquet_file.metadata.row_group(rg).num_rows
            self.index.extend((rg, i) for i in range(n))

        self._speaker_to_label = self._build_speaker_index()
        self._cached_row_group_idx: int | None = None
        self._cached_row_group: Any = None

        self.codec = (
            CodecPipeline(codec_preset, backend=codec_backend)
            if codec_preset
            else None
        )
        self._codec_pipelines: dict[str, CodecPipeline] = {}
        self._random_codec_preset_names = self._build_random_codec_preset_names(
            codec_preset
        )
        self.augmentor = (
            build_audio_augmentor(augment_cfg)
            if augment_cfg is not None
            else AudioAugmentPipeline(enabled=False)
        )

    def __len__(self) -> int:
        return len(self.index)

    @property
    def num_speakers(self) -> int:
        return len(self._speaker_to_label)

    def _build_speaker_index(self) -> dict[str, int]:
        speakers: set[str] = set()
        for rg in range(self.parquet_file.num_row_groups):
            batch = self.parquet_file.read_row_group(rg, columns=["speaker_id"])
            speakers.update(str(s) for s in batch.column("speaker_id").to_pylist())
        return {speaker: idx for idx, speaker in enumerate(sorted(speakers))}

    def _read_row(self, row_group: int, row_index: int) -> dict[str, Any]:
        if self._cached_row_group_idx != row_group:
            self._cached_row_group = self.parquet_file.read_row_group(row_group)
            self._cached_row_group_idx = row_group
        table = self._cached_row_group
        row: dict[str, Any] = {}
        for name in table.column_names:
            row[name] = table.column(name)[row_index].as_py()
        return row

    def _decode(self, path: str) -> tuple[torch.Tensor, int]:
        try:
            return load_with_torchcodec(
                path,
                sample_rate=self.sample_rate,
                window_num_samples=self.trim_num_samples,
            )
        except Exception:
            LOG.exception("Failed to decode audio with torchcodec: %s", path)
            raise

    def _fallback_sample_index(
        self,
        idx: int,
        attempted_indices: set[int],
    ) -> int | None:
        for candidate in range(idx - 1, -1, -1):
            if candidate not in attempted_indices:
                return candidate
        for candidate in range(idx + 1, len(self)):
            if candidate not in attempted_indices:
                return candidate
        return None

    def _codec_modes_enabled(self) -> bool:
        return (
            self.probabilities.codec_only > 0.0
            or self.probabilities.codec_then_augment > 0.0
        )

    def _build_random_codec_preset_names(
        self, codec_preset: str | None
    ) -> list[str]:
        if codec_preset is not None or not self._codec_modes_enabled():
            return []
        available = [
            name for name in list_presets() if check_preset_available(name)
        ]
        if not available:
            LOG.warning(
                "Codec probabilities are enabled, but no available codec presets were found; "
                "codec transforms will be skipped."
            )
        return available

    def _get_codec_pipeline(self) -> CodecPipeline | None:
        if self.codec is not None:
            return self.codec
        if not self._random_codec_preset_names:
            return None
        preset_name = random.choice(self._random_codec_preset_names)
        pipeline = self._codec_pipelines.get(preset_name)
        if pipeline is None:
            pipeline = CodecPipeline(
                preset_name,
                backend=self._codec_backend,
                check_available=False,
            )
            self._codec_pipelines[preset_name] = pipeline
        return pipeline

    def _apply_codec(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        pipeline = self._get_codec_pipeline()
        if pipeline is None:
            return waveform, sample_rate
        # Some corpora (for example 3D-Speaker array recordings) carry more
        # than two channels, while the codec round-trip only supports mono/stereo.
        if waveform.ndim == 2 and waveform.shape[0] > 2:
            waveform = self._to_mono(waveform)
        arr = waveform.detach().to(
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).numpy()
        if arr.ndim == 1:
            arr = arr[None, :]
        out_np, out_sr = pipeline.roundtrip(arr, sample_rate)
        out = torch.from_numpy(out_np)
        if out.shape[0] == 1:
            out = out.squeeze(0)
        out, out_sr = self._ensure_codec_output_16k(out, out_sr)
        return out, out_sr

    @staticmethod
    def _ensure_codec_output_16k(
        waveform: torch.Tensor, sample_rate: int, target_sample_rate: int = 16_000
    ) -> tuple[torch.Tensor, int]:
        if sample_rate == target_sample_rate:
            return waveform, sample_rate

        to_resample = waveform
        squeezed = False
        if waveform.ndim == 1:
            to_resample = waveform.unsqueeze(0)
            squeezed = True
        resampled = torchaudio.functional.resample(
            to_resample,
            sample_rate,
            target_sample_rate,
        )
        if squeezed:
            resampled = resampled.squeeze(0)
        return resampled, target_sample_rate

    def _should_apply_codec(self, row: dict[str, Any]) -> bool:
        if self.codec is None and not self._random_codec_preset_names:
            return False
        if self.codec_min_duration_sec is None:
            return True
        duration = row.get("duration_sec")
        if duration is None:
            return True
        try:
            return float(duration) >= float(self.codec_min_duration_sec)
        except (TypeError, ValueError):
            return True

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.trim_num_samples is None:
            return waveform
        return pad_random(waveform, self.trim_num_samples)

    def _trim_if_long(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.trim_num_samples is None:
            return waveform
        if waveform.shape[-1] <= self.trim_num_samples:
            return waveform
        start = random.randint(0, waveform.shape[-1] - self.trim_num_samples)
        return waveform[..., start : start + self.trim_num_samples]

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            return waveform
        if waveform.ndim == 2:
            return waveform.mean(dim=0)
        raise ValueError(f"expected waveform [T] or [C,T], got {tuple(waveform.shape)}")

    def _waveform_for_index(
        self,
        idx: int,
        *,
        attempted_indices: set[int] | None = None,
    ) -> tuple[torch.Tensor, int, str]:
        """Декод + codec/augment пайплайн для одной строки; возвращает waveform, label, режим.

        ``mode`` — одно из ``keep``, ``codec``, ``augment``, ``codec_augment`` (внутренние
        имена из :meth:`CodecAugmentProbabilities.choose_mode`). Если декод падает и
        ``fallback_on_decode_error=True`` — рекурсивно пробуем соседние индексы,
        защищаясь от зацикливания через ``attempted_indices``."""
        attempted = attempted_indices or set()
        if idx in attempted:
            raise RuntimeError(f"dataset fallback loop detected at index {idx}")
        attempted.add(idx)
        row_group, row_index = self.index[idx]
        row = self._read_row(row_group, row_index)
        path = str(row["audio_path"])
        speaker_id = str(row["speaker_id"])
        label = self._speaker_to_label[speaker_id]

        try:
            waveform, sample_rate = self._decode(path)
        except Exception:
            if not self.fallback_on_decode_error:
                raise
            fallback_idx = self._fallback_sample_index(idx, attempted)
            if fallback_idx is None:
                raise RuntimeError(
                    "Failed to decode audio and no fallback dataset item was available"
                ) from None
            LOG.warning(
                "Falling back from dataset index %s (%s) to index %s after decode failure",
                idx,
                path,
                fallback_idx,
            )
            return self._waveform_for_index(
                fallback_idx,
                attempted_indices=attempted,
            )
        waveform = self._trim_if_long(waveform)
        mode = self.probabilities.choose_mode()
        if mode in {"codec", "codec_augment"} and self._should_apply_codec(row):
            waveform, sample_rate = self._apply_codec(waveform, sample_rate)
        if mode in {"augment", "codec_augment"}:
            waveform = self.augmentor.apply_to_waveform(waveform, sample_rate)
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)
        waveform = self._to_mono(waveform)
        waveform = self._pad_or_trim(waveform)
        return waveform, label, mode

    def sample_with_mode(self, idx: int) -> tuple[dict[str, Any], str]:
        """Same as ``__getitem__`` but also returns the sampled codec/augment mode key."""
        waveform, label, mode = self._waveform_for_index(idx)
        return {"waveform": waveform, "label": label}, mode

    def __getitem__(self, idx: int) -> dict[str, Any]:
        waveform, label, _ = self._waveform_for_index(idx)
        return {
            "waveform": waveform,
            "label": label,
        }
