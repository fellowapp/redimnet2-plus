#!/usr/bin/env python3
"""
Benchmark speaker-embedding inference speed on a list of utterances.

- ``--backend onnx``: ONNX Runtime, waveform in → embeddings (e.g. ECAPA ONNX export).
- ``--backend trt``: TensorRT engine, waveform in → embeddings.
- ``--backend torch``: Hydra ``encoder`` from a model YAML (e.g. ReDimNet2-B6 pretrained):
  mel via ``encoder.spec_frontend`` + ``encoder(...)``. Use ``--compute_dtype bfloat16`` for
  CUDA ``torch.autocast`` bf16 matmul/conv (weights stay float32).

Requires: ``torch``, ``torchaudio``, ``tqdm``; for ONNX also ``onnxruntime`` (or GPU build),
for TensorRT also ``tensorrt``.

Waveform IO uses built-in ``load_audio`` / ``get_chunk`` by default, or
``--preprocessor model_dir`` to import from ``model_dir/src/dataset.py`` (may need pandas).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _try_cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_onnx_providers(device: str = "auto") -> list[str]:
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    providers: list[str] = []
    if device in ("auto", "cuda"):
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers if providers else ["CPUExecutionProvider"]


def import_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise SystemExit(
            "TensorRT backend requested but `tensorrt` is not installed in the active environment."
        ) from exc
    return trt


def builtin_load_audio(filepath: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)


def builtin_get_chunk(
    waveform: Union[torch.Tensor, np.ndarray],
    chunk_len: Union[tuple[int, int], int],
    random_chunk: bool = True,
) -> torch.Tensor | np.ndarray:
    """Aligned with baseline ``src.dataset.get_chunk`` (random crop / tile / pad)."""
    ndim = len(waveform.shape)
    is_torch = False
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().numpy()
        is_torch = True
    if ndim == 2:
        waveform = waveform[0]
    data_len = len(waveform)
    if isinstance(chunk_len, tuple):
        chunk_len = int(np.random.uniform(*chunk_len))
    if data_len >= chunk_len:
        chunk_start = 0
        if random_chunk:
            chunk_start = random.randint(0, data_len - chunk_len)
        waveform = waveform[chunk_start : chunk_start + chunk_len]
    elif data_len > 0:
        repeat_factor = chunk_len // data_len + 1
        waveform = np.tile(waveform, repeat_factor)
        waveform = waveform[:chunk_len]
    else:
        waveform = np.zeros(chunk_len, dtype=np.float32)
    if ndim == 2:
        waveform = waveform[None, :]
    if is_torch:
        waveform = torch.tensor(waveform)
    return waveform


def import_preprocessors(model_dir: Path) -> tuple[Callable, Callable]:
    """Load ``load_audio`` and ``get_chunk`` from ``model_dir/src/dataset.py``."""
    root = model_dir.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.dataset import get_chunk, load_audio  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            f"Could not import src.dataset from model_dir={root}. "
            "Ensure the package exposes load_audio(path, sr) and get_chunk(...)."
        ) from e
    return load_audio, get_chunk


def parse_voxceleb_trial_rows(path: Path) -> list[tuple[int, str, str]]:
    """Parse ``label wav1 wav2`` lines from a VoxCeleb trial file."""
    rows: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed trial line in {path}:{line_num} -> {line!r}")
            label = int(parts[0])
            if label not in (0, 1):
                raise ValueError(f"Trial label must be 0/1 in {path}:{line_num}, got {label}")
            rows.append((label, parts[1], parts[2]))
    return rows


@dataclass
class TrialSetup:
    name: str
    path: str
    audio_root: str
    hard_build_glob: bool
    include_in_pooled: bool
    rows: list[tuple[int, str, str]]


def discover_voxceleb_trial_files(path: Path) -> list[Path]:
    """Resolve the standard VoxCeleb ``_o``, ``_e``, ``_h`` trial files."""
    source = path.expanduser().resolve()
    root = source if source.is_dir() else source.parent
    names = ["voxceleb1_o.txt", "voxceleb_e.txt", "voxceleb_h.txt"]
    files = [root / name for name in names]
    missing = [str(item) for item in files if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "Could not resolve all pooled VoxCeleb trial files under "
            f"{root}: missing {missing}"
        )
    return files


def resolve_trial_utterance_path(
    audio_root: Path,
    utt_path: str,
    *,
    hard_build_glob: bool,
) -> str | None:
    rel = Path(utt_path)
    direct = rel if rel.is_absolute() else (audio_root / rel)
    if direct.is_file():
        return str(direct.resolve())
    if not hard_build_glob:
        return None
    parent_dir = direct.parent
    stem = direct.stem
    if not parent_dir.is_dir():
        return None
    matches = sorted(parent_dir.glob(f"{stem}__hard_*.wav"))
    if not matches:
        return None
    return str(matches[0].resolve())


def load_trial_setups(
    path: Path,
    *,
    pooled: bool,
    audio_root: Path,
    extra_voxceleb_o_audio_root: Path | None,
) -> list[TrialSetup]:
    trial_files = discover_voxceleb_trial_files(path) if pooled else [path]
    setups: list[TrialSetup] = []
    for trial_file in trial_files:
        setups.append(
            TrialSetup(
                name=trial_file.stem,
                path=str(trial_file.resolve()),
                audio_root=str(audio_root.resolve()),
                hard_build_glob=False,
                include_in_pooled=True,
                rows=parse_voxceleb_trial_rows(trial_file),
            )
        )
    if extra_voxceleb_o_audio_root is not None:
        voxceleb1_o_file = next(
            (trial_file for trial_file in trial_files if trial_file.stem == "voxceleb1_o"),
            None,
        )
        if voxceleb1_o_file is None:
            discovered = discover_voxceleb_trial_files(path)
            voxceleb1_o_file = next(
                (trial_file for trial_file in discovered if trial_file.stem == "voxceleb1_o"),
                None,
            )
        if voxceleb1_o_file is None:
            raise FileNotFoundError("Could not resolve voxceleb1_o.txt for extra hard benchmark setup.")
        setups.append(
            TrialSetup(
                name="voxceleb1_o_hard",
                path=str(voxceleb1_o_file.resolve()),
                audio_root=str(extra_voxceleb_o_audio_root.resolve()),
                hard_build_glob=True,
                include_in_pooled=False,
                rows=parse_voxceleb_trial_rows(voxceleb1_o_file),
            )
        )
    return setups


def parse_voxceleb_trial_setups(
    setups: Sequence[TrialSetup],
    *,
    dedupe: bool,
    max_utterances: int | None,
) -> list[str]:
    """Parse one or more trial files; return relative wav paths."""
    rel_paths: list[str] = []
    seen: set[str] = set()
    for setup in setups:
        setup_audio_root = Path(setup.audio_root)
        for _label, utt1, utt2 in setup.rows:
            for p in (utt1, utt2):
                resolved = resolve_trial_utterance_path(
                    setup_audio_root,
                    p,
                    hard_build_glob=setup.hard_build_glob,
                )
                if resolved is None:
                    continue
                if dedupe:
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                rel_paths.append(resolved)
                if max_utterances is not None and len(rel_paths) >= max_utterances:
                    return rel_paths
    return rel_paths


def parse_voxceleb_trials(
    path: Path,
    *,
    dedupe: bool,
    max_utterances: int | None,
) -> list[str]:
    """Parse ``label wav1 wav2`` lines; return relative wav paths."""
    rel_paths: list[str] = []
    seen: set[str] = set()
    for _label, utt1, utt2 in parse_voxceleb_trial_rows(path):
        for p in (utt1, utt2):
            if dedupe:
                if p in seen:
                    continue
                seen.add(p)
            rel_paths.append(p)
            if max_utterances is not None and len(rel_paths) >= max_utterances:
                return rel_paths
    return rel_paths


def parse_plain_list(path: Path, *, max_utterances: int | None) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
            if max_utterances is not None and len(out) >= max_utterances:
                break
    return out


def resolve_paths(rel_or_abs: Sequence[str], audio_root: Path) -> list[str]:
    resolved: list[str] = []
    for p in rel_or_abs:
        pp = Path(p)
        if pp.is_absolute():
            resolved.append(str(pp))
        else:
            resolved.append(str((audio_root / pp).resolve()))
    return resolved


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Equal Error Rate from binary labels and similarity scores."""
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


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def compute_trial_scores(
    normalized_embeddings: np.ndarray,
    path_to_idx: dict[str, int],
    setup: TrialSetup,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    labels: list[int] = []
    scores: list[float] = []
    skipped_trials = 0
    setup_audio_root = Path(setup.audio_root)
    for label, utt1, utt2 in setup.rows:
        resolved1 = resolve_trial_utterance_path(
            setup_audio_root,
            utt1,
            hard_build_glob=setup.hard_build_glob,
        )
        resolved2 = resolve_trial_utterance_path(
            setup_audio_root,
            utt2,
            hard_build_glob=setup.hard_build_glob,
        )
        if resolved1 is None or resolved2 is None:
            skipped_trials += 1
            continue
        idx1 = path_to_idx.get(resolved1)
        idx2 = path_to_idx.get(resolved2)
        if idx1 is None or idx2 is None:
            skipped_trials += 1
            continue
        labels.append(label)
        scores.append(float(np.dot(normalized_embeddings[idx1], normalized_embeddings[idx2])))

    if not labels:
        return None, None, skipped_trials
    return (
        np.asarray(labels, dtype=np.int32),
        np.asarray(scores, dtype=np.float64),
        skipped_trials,
    )


def compute_trial_eer(
    embeddings: np.ndarray,
    embedding_paths: Sequence[str],
    trial_rows: Sequence[tuple[int, str, str]] | None,
    audio_root: Path,
) -> tuple[float | None, int | None, int]:
    if not trial_rows:
        return None, None, 0

    path_to_idx: dict[str, int] = {}
    for idx, path in enumerate(embedding_paths):
        path_to_idx.setdefault(path, idx)

    normalized = l2_normalize_embeddings(embeddings)
    temp_setup = TrialSetup(
        name="trial",
        path="",
        audio_root=str(audio_root.resolve()),
        hard_build_glob=False,
        include_in_pooled=True,
        rows=list(trial_rows),
    )
    labels_np, scores_np, skipped_trials = compute_trial_scores(normalized, path_to_idx, temp_setup)
    if labels_np is None or scores_np is None:
        return None, 0, skipped_trials
    return compute_eer(labels_np, scores_np), int(labels_np.size), skipped_trials


@dataclass
class TrialVerificationMetrics:
    eer: float | None
    num_trials: int | None
    skipped_trials: int


def compute_verification_metrics(
    embeddings: np.ndarray,
    embedding_paths: Sequence[str],
    trial_setups: Sequence[TrialSetup] | None,
    audio_root: Path,
) -> "VerificationMetrics":
    if not trial_setups:
        return VerificationMetrics(
            eer=None,
            num_trials=None,
            skipped_trials=0,
            pooled_eer=None,
            pooled_num_trials=None,
            pooled_skipped_trials=0,
            setups={},
        )

    path_to_idx: dict[str, int] = {}
    for idx, path in enumerate(embedding_paths):
        path_to_idx.setdefault(path, idx)

    normalized = l2_normalize_embeddings(embeddings)
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    setup_metrics: dict[str, TrialVerificationMetrics] = {}
    pooled_skipped_trials = 0

    for setup in trial_setups:
        labels_np, scores_np, skipped_trials = compute_trial_scores(normalized, path_to_idx, setup)
        if setup.include_in_pooled:
            pooled_skipped_trials += skipped_trials
        if labels_np is None or scores_np is None:
            setup_metrics[setup.name] = TrialVerificationMetrics(
                eer=None,
                num_trials=0,
                skipped_trials=skipped_trials,
            )
            continue
        if setup.include_in_pooled:
            pooled_labels.append(labels_np)
            pooled_scores.append(scores_np)
        setup_metrics[setup.name] = TrialVerificationMetrics(
            eer=compute_eer(labels_np, scores_np),
            num_trials=int(labels_np.size),
            skipped_trials=skipped_trials,
        )

    if pooled_labels and pooled_scores:
        pooled_labels_np = np.concatenate(pooled_labels)
        pooled_scores_np = np.concatenate(pooled_scores)
        pooled_eer = compute_eer(pooled_labels_np, pooled_scores_np)
        pooled_num_trials = int(pooled_labels_np.size)
    else:
        pooled_eer = None
        pooled_num_trials = 0

    return VerificationMetrics(
        eer=pooled_eer,
        num_trials=pooled_num_trials,
        skipped_trials=pooled_skipped_trials,
        pooled_eer=pooled_eer,
        pooled_num_trials=pooled_num_trials,
        pooled_skipped_trials=pooled_skipped_trials,
        setups=setup_metrics,
    )


class WaveformListDataset(Dataset):
    """Fixed-length waveform chunks for a list of absolute audio paths."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        sample_rate: int,
        chunk_seconds: float,
        load_audio_fn: Callable,
        get_chunk_fn: Callable,
        random_chunk: bool,
    ) -> None:
        self.paths = list(paths)
        self.sample_rate = int(sample_rate)
        self.num_samples = int(self.sample_rate * float(chunk_seconds))
        self._load_audio = load_audio_fn
        self._get_chunk = get_chunk_fn
        self.random_chunk = bool(random_chunk)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        waveform = self._load_audio(path, self.sample_rate)
        waveform = self._get_chunk(waveform, self.num_samples, self.random_chunk)
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        return waveform.float()


def pick_embedding_output(session: ort.InferenceSession, outputs: list[np.ndarray]) -> np.ndarray:
    names = [o.name for o in session.get_outputs()]
    if "embeddings" in names:
        return outputs[names.index("embeddings")]
    return outputs[0]


def run_session(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    inputs = session.get_inputs()
    input_name = inputs[0].name
    outs = session.run(None, {input_name: x.astype(np.float32, copy=False)})
    return pick_embedding_output(session, outs)


def torch_dtype_from_trt_dtype(trt, dtype) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
    }
    if hasattr(trt.DataType, "BF16"):
        mapping[trt.DataType.BF16] = torch.bfloat16
    if dtype not in mapping:
        raise RuntimeError(f"Unsupported TensorRT tensor dtype: {dtype}")
    return mapping[dtype]


def load_trt_engine(engine_path: Path):
    trt = import_tensorrt()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    return trt, runtime, engine


def get_trt_io_names(trt, engine) -> tuple[str, list[str]]:
    input_names: list[str] = []
    output_names: list[str] = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly 1 TensorRT input tensor, got {input_names}")
    if not output_names:
        raise RuntimeError("TensorRT engine produced no output tensors.")
    return input_names[0], output_names


class TrtRunner:
    """Reuse TensorRT buffers for fixed-shape TensorRT inference."""

    def __init__(self, trt, engine, context, device: torch.device) -> None:
        self.trt = trt
        self.engine = engine
        self.context = context
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.input_name, self.output_names = get_trt_io_names(trt, engine)
        self.output_name = (
            "embeddings" if "embeddings" in self.output_names else self.output_names[0]
        )
        self.input_dtype = torch_dtype_from_trt_dtype(
            trt, engine.get_tensor_dtype(self.input_name)
        )
        self._expected_input_shape = tuple(
            int(v) for v in engine.get_tensor_shape(self.input_name)
        )
        self._is_dynamic = any(dim < 0 for dim in self._expected_input_shape)
        self._buffer_shape: tuple[int, ...] | None = None
        self._tensor_buffers: dict[str, torch.Tensor] = {}
        self._pending_trim_to: int | None = None
        self._pending_start_event: torch.cuda.Event | None = None
        self._pending_end_event: torch.cuda.Event | None = None

    def _ensure_buffers(self, batch_shape: tuple[int, ...]) -> None:
        need_reallocate = False
        if self._is_dynamic:
            if self._buffer_shape != batch_shape:
                self.context.set_input_shape(self.input_name, batch_shape)
                self._buffer_shape = batch_shape
                need_reallocate = True
        else:
            expected_shape = self._expected_input_shape
            if len(expected_shape) != len(batch_shape):
                raise RuntimeError(
                    f"TensorRT input rank mismatch: engine={expected_shape}, batch={batch_shape}"
                )
            if batch_shape[0] > expected_shape[0]:
                raise RuntimeError(
                    f"Batch size {batch_shape[0]} exceeds fixed TensorRT engine batch {expected_shape[0]}"
                )
            if tuple(batch_shape[1:]) != tuple(expected_shape[1:]):
                raise RuntimeError(
                    "TensorRT input shape mismatch beyond batch: "
                    f"engine={expected_shape}, batch={batch_shape}"
                )
            if self._buffer_shape is None:
                self._buffer_shape = expected_shape

        if self._tensor_buffers and not need_reallocate:
            return

        assert self._buffer_shape is not None
        self._tensor_buffers = {}
        self._tensor_buffers[self.input_name] = torch.empty(
            self._buffer_shape,
            device=self.device,
            dtype=self.input_dtype,
        )
        for output_name in self.output_names:
            output_shape = tuple(int(v) for v in self.context.get_tensor_shape(output_name))
            output_dtype = torch_dtype_from_trt_dtype(
                self.trt, self.engine.get_tensor_dtype(output_name)
            )
            self._tensor_buffers[output_name] = torch.empty(
                output_shape,
                device=self.device,
                dtype=output_dtype,
            )

        for name, tensor in self._tensor_buffers.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

    def submit(self, batch: torch.Tensor) -> None:
        x = batch.contiguous()
        trim_to = int(x.shape[0])
        self._ensure_buffers(tuple(int(v) for v in x.shape))
        self._pending_trim_to = trim_to
        self._pending_start_event = torch.cuda.Event(enable_timing=True)
        self._pending_end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(self.stream):
            # Caller's HtoD copy of `x` lives on another stream; wait for it so the TRT stream
            # doesn't race against a half-copied input (manifested as per-batch corruption past
            # the first prefetched batch).
            if x.is_cuda:
                self.stream.wait_stream(torch.cuda.default_stream(x.device))
            input_buffer = self._tensor_buffers[self.input_name]
            input_buffer[:trim_to].copy_(x, non_blocking=True)
            if trim_to < input_buffer.shape[0]:
                input_buffer[trim_to:].zero_()

            self._pending_start_event.record(self.stream)
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execution failed.")
            self._pending_end_event.record(self.stream)

    def wait(self) -> tuple[np.ndarray, float]:
        if self._pending_trim_to is None:
            raise RuntimeError("TensorRT runner wait() called without a pending submission.")

        self.stream.synchronize()
        output = (
            self._tensor_buffers[self.output_name][: self._pending_trim_to]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        assert self._pending_start_event is not None
        assert self._pending_end_event is not None
        elapsed_s = self._pending_start_event.elapsed_time(self._pending_end_event) / 1000.0
        self._pending_trim_to = None
        self._pending_start_event = None
        self._pending_end_event = None
        return output, elapsed_s

    def run(self, batch: torch.Tensor) -> tuple[np.ndarray, float]:
        self.submit(batch)
        return self.wait()


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_encoder_from_hydra_config(config_path: Path) -> torch.nn.Module:
    """Instantiate ``encoder`` from a model YAML (e.g. ``configs/model/redimnet2_b6_pretrained.yaml``)."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if "encoder" not in cfg:
        raise ValueError(f"Config {config_path} must define top-level 'encoder'")
    enc = instantiate(cfg.encoder)
    return enc.float().eval()


def benchmark_torch_loop(
    loader: DataLoader,
    encoder: torch.nn.Module,
    device: torch.device,
    warmup_batches: int,
    *,
    compute_dtype: str,
) -> tuple[np.ndarray, float, float, int]:
    """Time mel frontend + encoder forward. ``compute_dtype`` is float32 or bfloat16 (CUDA autocast)."""
    from asv.data.collate import compute_spectrogram_batch

    use_bf16 = compute_dtype == "bfloat16"
    if use_bf16 and device.type != "cuda":
        raise ValueError("bfloat16 inference requires CUDA (torch.autocast)")

    encoder = encoder.to(device)
    amp_ctx = torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=use_bf16,
    )

    it = iter(loader)
    for _ in range(max(0, warmup_batches)):
        try:
            batch = next(it)
        except StopIteration:
            break
        w = batch.to(device, non_blocking=True)
        _try_cuda_sync()
        with torch.no_grad(), amp_ctx:
            spec = compute_spectrogram_batch(
                w, encoder.spec_frontend, augment_cfg=None, training=False
            )
            _ = encoder(spec)
        _try_cuda_sync()

    all_emb: list[torch.Tensor] = []
    compute_time = 0.0
    n_batches = 0
    t_wall0 = time.perf_counter()
    for batch in tqdm(loader, desc="torch", total=len(loader)):
        w = batch.to(device, non_blocking=True)
        _try_cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad(), amp_ctx:
            spec = compute_spectrogram_batch(
                w, encoder.spec_frontend, augment_cfg=None, training=False
            )
            emb = encoder(spec)
        _try_cuda_sync()
        compute_time += time.perf_counter() - t0
        all_emb.append(emb.detach().float().cpu())
        n_batches += 1
    wall = time.perf_counter() - t_wall0
    emb_np = torch.cat(all_emb, dim=0).numpy()
    return emb_np, wall, compute_time, n_batches


def benchmark_trt_loop(
    loader: DataLoader,
    engine_path: Path,
    device: torch.device,
    warmup_batches: int,
    trt_inflight_batches: int,
) -> tuple[np.ndarray, float, float, int, str, list[str]]:
    trt, runtime, engine = load_trt_engine(engine_path)
    _ = runtime  # Keep runtime alive for the engine lifetime.
    runners: list[TrtRunner] = []
    for _ in range(max(1, trt_inflight_batches)):
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        runners.append(TrtRunner(trt, engine, context, device))

    it = iter(loader)
    for _ in range(max(0, warmup_batches)):
        try:
            batch = next(it)
        except StopIteration:
            break
        _try_cuda_sync()
        _ = runners[0].run(batch)
        _try_cuda_sync()

    all_emb: list[np.ndarray] = []
    compute_time = 0.0
    n_batches = 0
    pending: deque[TrtRunner] = deque()
    t_wall0 = time.perf_counter()
    for batch in tqdm(loader, desc="trt", total=len(loader)):
        runner = runners[n_batches % len(runners)]
        if len(pending) == len(runners):
            finished_runner = pending.popleft()
            emb, elapsed_s = finished_runner.wait()
            compute_time += elapsed_s
            all_emb.append(emb)
        runner.submit(batch)
        pending.append(runner)
        n_batches += 1

    while pending:
        finished_runner = pending.popleft()
        emb, elapsed_s = finished_runner.wait()
        compute_time += elapsed_s
        all_emb.append(emb)

    wall = time.perf_counter() - t_wall0
    emb_np = np.concatenate(all_emb, axis=0)
    ref_runner = runners[0]
    return emb_np, wall, compute_time, n_batches, ref_runner.input_name, ref_runner.output_names


@dataclass
class BenchmarkParams:
    backend: str
    model_dir: str
    preprocessor: str
    onnx_path: str
    trt_engine_path: str | None
    torch_encoder_config: str | None
    compute_dtype: str
    utterance_source: str
    list_format: str
    audio_root: str
    extra_voxceleb_o_audio_root: str | None
    batch_size: int
    num_workers: int
    trt_inflight_batches: int
    sample_rate: int
    chunk_seconds: float
    random_chunk: bool
    dedupe: bool
    max_utterances: int | None
    device: str
    warmup_batches: int
    onnx_providers_requested: str
    torch_num_threads: int | None
    hostname: str
    started_at_utc: str
    num_utterances: int
    onnx_input_name: str
    onnx_output_names: list[str]
    onnx_active_providers: list[str]
    trt_input_name: str
    trt_output_names: list[str]


@dataclass
class BenchmarkTimings:
    wall_time_s: float
    compute_time_s: float
    batches: int
    utterances_per_s_wall: float
    seconds_audio: float
    rtf: float
    embedding_dim: int


@dataclass
class VerificationMetrics:
    eer: float | None
    num_trials: int | None
    skipped_trials: int
    pooled_eer: float | None
    pooled_num_trials: int | None
    pooled_skipped_trials: int
    setups: dict[str, TrialVerificationMetrics]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark embedding inference (ONNX, TensorRT, or PyTorch encoder) on utterance lists."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("onnx", "trt", "torch"),
        default="onnx",
        help="onnx: ONNX Runtime; trt: TensorRT engine; torch: Hydra encoder (e.g. ReDimNet2) + mel (see --compute_dtype).",
    )
    parser.add_argument(
        "--torch_encoder_config",
        type=Path,
        default=REPO_ROOT / "configs/model/redimnet2_b6_pretrained.yaml",
        help="Model YAML with top-level ``encoder`` (used when --backend torch).",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("/home/kirill/ASV/benchmarking/outputs/speakerid-challenge-baseline-main"),
        help="Reference package dir (provenance). With --preprocessor model_dir, imports src.dataset.",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=("builtin", "model_dir"),
        default="builtin",
        help="builtin: self-contained torchaudio loader; model_dir: import from model_dir/src/dataset.py.",
    )
    parser.add_argument(
        "--onnx_path",
        type=Path,
        default=Path("/home/kirill/ASV/benchmarking/outputs/baseline.onnx"),
        help="Path to ONNX weights.",
    )
    parser.add_argument(
        "--trt_engine_path",
        type=Path,
        default=Path(
            "/home/kirill/ASV/benchmarking/outputs/redimnet2_b6_fp16_frontendfp32_bs32_s96000.plan"
        ),
        help="Path to TensorRT engine (.plan). Used when --backend trt.",
    )
    parser.add_argument(
        "--utterance_list",
        type=Path,
        default=Path("/home/kirill/mnt/ssd_1tb_1/VoxCeleb/voxceleb1_o.txt"),
        help="Trial/list file or plain path list.",
    )
    parser.add_argument(
        "--audio_root",
        type=Path,
        default=Path("/home/kirill/mnt/ssd_1tb_1/VoxCeleb/wav"),
        help="Root directory prepended to relative wav paths (VoxCeleb: .../wav).",
    )
    parser.add_argument(
        "--list_format",
        type=str,
        choices=("voxceleb_trials", "voxceleb_trials_pooled", "plain"),
        default="voxceleb_trials",
        help=(
            "voxceleb_trials: one 'label wav1 wav2' file; "
            "voxceleb_trials_pooled: discover and pool voxceleb1_o/e/h trial files from "
            "--utterance_list (file sibling directory or explicit VoxCeleb root); "
            "plain: one path per line."
        ),
    )
    parser.add_argument(
        "--extra_voxceleb_o_audio_root",
        type=Path,
        default=None,
        help=(
            "Optional extra verification root scored against voxceleb1_o trials only. "
            "Intended for datasets like VoxCelebHard that store augmented files as "
            "<stem>__hard_*.wav."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers. Default is auto-tuned per backend: 0 for TRT, 4 otherwise.",
    )
    parser.add_argument(
        "--trt_inflight_batches",
        type=int,
        default=None,
        help="TensorRT only: execution contexts kept in flight on separate CUDA streams. Default: 2 for TRT, 1 otherwise.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunk_seconds", type=float, default=6.0)
    parser.add_argument(
        "--random_chunk",
        action="store_true",
        help=(
            "Use random temporal crops for long clips (same idea as ``asv.validation.voxceleb`` "
            "``pad_random`` on ``validation.trim_num_samples``). Default is OFF, which always "
            "takes the first ``chunk_seconds`` of each file and will NOT match Hydra VoxCeleb EER."
        ),
    )
    parser.add_argument(
        "--no_dedupe",
        action="store_true",
        help="For voxceleb_trials: do not dedupe paths (more work if files repeat).",
    )
    parser.add_argument(
        "--max_utterances",
        type=int,
        default=None,
        help="Cap utterances after parsing (for quick dry runs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument(
        "--compute_dtype",
        type=str,
        choices=("float32", "bfloat16"),
        default="float32",
        help="Torch backend only: float32, or bfloat16 via CUDA autocast (requires --device cuda/auto with GPU).",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Write full params + timings JSON (default: benchmarking/outputs/speed_bench_<timestamp>.json).",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=None,
        help="If set, call torch.set_num_threads(n) for CPU fairness.",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Parse data and write report only (no ONNX run).",
    )
    args = parser.parse_args()

    resolved_num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else (0 if args.backend == "trt" else 4)
    )
    resolved_trt_inflight_batches = (
        int(args.trt_inflight_batches)
        if args.trt_inflight_batches is not None
        else (2 if args.backend == "trt" else 1)
    )

    if args.backend in {"onnx", "trt"} and args.compute_dtype != "float32":
        raise SystemExit("--compute_dtype applies only to --backend torch.")

    if args.torch_num_threads is not None:
        torch.set_num_threads(int(args.torch_num_threads))

    if args.preprocessor == "model_dir":
        load_audio_fn, get_chunk_fn = import_preprocessors(args.model_dir)
    else:
        load_audio_fn, get_chunk_fn = builtin_load_audio, builtin_get_chunk

    dedupe = not args.no_dedupe
    trial_setups: list[TrialSetup] | None = None
    if args.list_format == "voxceleb_trials":
        trial_setups = load_trial_setups(
            args.utterance_list,
            pooled=False,
            audio_root=args.audio_root,
            extra_voxceleb_o_audio_root=args.extra_voxceleb_o_audio_root,
        )
        rel = parse_voxceleb_trial_setups(
            trial_setups, dedupe=dedupe, max_utterances=args.max_utterances
        )
    elif args.list_format == "voxceleb_trials_pooled":
        trial_setups = load_trial_setups(
            args.utterance_list,
            pooled=True,
            audio_root=args.audio_root,
            extra_voxceleb_o_audio_root=args.extra_voxceleb_o_audio_root,
        )
        rel = parse_voxceleb_trial_setups(
            trial_setups, dedupe=dedupe, max_utterances=args.max_utterances
        )
    else:
        rel = parse_plain_list(args.utterance_list, max_utterances=args.max_utterances)

    paths = resolve_paths(rel, args.audio_root)
    if not paths:
        raise SystemExit("No utterances parsed; check --utterance_list and --list_format.")

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        sample = missing[:5]
        raise SystemExit(
            f"{len(missing)} audio files missing (showing up to 5): {sample}. "
            "Fix --audio_root or paths."
        )

    ds = WaveformListDataset(
        paths,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        load_audio_fn=load_audio_fn,
        get_chunk_fn=get_chunk_fn,
        random_chunk=args.random_chunk,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=resolved_num_workers,
        pin_memory=torch.cuda.is_available() and args.device != "cpu",
    )

    providers = get_onnx_providers(args.device)
    started_at = datetime.now(timezone.utc).isoformat()

    torch_cfg = (
        str(args.torch_encoder_config.resolve())
        if args.backend == "torch"
        else None
    )

    if args.skip_inference:
        report = {
            "params": asdict(
                BenchmarkParams(
                    backend=args.backend,
                    model_dir=str(args.model_dir.resolve()),
                    preprocessor=args.preprocessor,
                    onnx_path=str(args.onnx_path.resolve()),
                    trt_engine_path=str(args.trt_engine_path.resolve()),
                    torch_encoder_config=torch_cfg,
                    compute_dtype=args.compute_dtype,
                    utterance_source=str(args.utterance_list.resolve()),
                    list_format=args.list_format,
                    audio_root=str(args.audio_root.resolve()),
                    extra_voxceleb_o_audio_root=(
                        str(args.extra_voxceleb_o_audio_root.resolve())
                        if args.extra_voxceleb_o_audio_root is not None
                        else None
                    ),
                    batch_size=args.batch_size,
                    num_workers=resolved_num_workers,
                    trt_inflight_batches=resolved_trt_inflight_batches,
                    sample_rate=args.sample_rate,
                    chunk_seconds=args.chunk_seconds,
                    random_chunk=args.random_chunk,
                    dedupe=dedupe,
                    max_utterances=args.max_utterances,
                    device=args.device,
                    warmup_batches=args.warmup_batches,
                    onnx_providers_requested=args.device,
                    torch_num_threads=args.torch_num_threads,
                    hostname=socket.gethostname(),
                    started_at_utc=started_at,
                    num_utterances=len(paths),
                    onnx_input_name="",
                    onnx_output_names=[],
                    onnx_active_providers=[],
                    trt_input_name="",
                    trt_output_names=[],
                )
            ),
            "timings": None,
            "verification": asdict(
                VerificationMetrics(
                    eer=None,
                    num_trials=None,
                    skipped_trials=0,
                    pooled_eer=None,
                    pooled_num_trials=None,
                    pooled_skipped_trials=0,
                    setups={},
                )
            ),
            "note": "skip_inference=True",
        }
        out = args.report_json or (
            Path(__file__).resolve().parent
            / "outputs"
            / f"speed_bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["params"], indent=2))
        print(f"Wrote {out}")
        return

    if args.backend == "torch":
        if not args.torch_encoder_config.is_file():
            raise SystemExit(f"Torch encoder config not found: {args.torch_encoder_config}")
        enc = load_encoder_from_hydra_config(args.torch_encoder_config)
        if not hasattr(enc, "spec_frontend"):
            raise SystemExit(
                "Encoder must expose ``spec_frontend`` for mel computation (e.g. ReDimNet2Encoder)."
            )
        device = resolve_torch_device(args.device)
        if args.compute_dtype == "bfloat16" and device.type != "cuda":
            raise SystemExit("--compute_dtype bfloat16 requires CUDA (use --device cuda or auto with a GPU).")
        emb_np, wall, compute_time, n_batches = benchmark_torch_loop(
            loader,
            enc,
            device,
            args.warmup_batches,
            compute_dtype=args.compute_dtype,
        )
        n_utts = int(emb_np.shape[0])
        seconds_audio = n_utts * float(args.chunk_seconds)
        utt_per_s = n_utts / wall if wall > 0 else float("nan")
        rtf = wall / seconds_audio if seconds_audio > 0 else float("nan")
        timings = BenchmarkTimings(
            wall_time_s=wall,
            compute_time_s=compute_time,
            batches=n_batches,
            utterances_per_s_wall=utt_per_s,
            seconds_audio=seconds_audio,
            rtf=rtf,
            embedding_dim=int(emb_np.shape[1]),
        )
        params = BenchmarkParams(
            backend="torch",
            model_dir=str(args.model_dir.resolve()),
            preprocessor=args.preprocessor,
            onnx_path=str(args.onnx_path.resolve()),
            trt_engine_path=str(args.trt_engine_path.resolve()),
            torch_encoder_config=torch_cfg,
            compute_dtype=args.compute_dtype,
            utterance_source=str(args.utterance_list.resolve()),
            list_format=args.list_format,
            audio_root=str(args.audio_root.resolve()),
            extra_voxceleb_o_audio_root=(
                str(args.extra_voxceleb_o_audio_root.resolve())
                if args.extra_voxceleb_o_audio_root is not None
                else None
            ),
            batch_size=args.batch_size,
            num_workers=resolved_num_workers,
            trt_inflight_batches=resolved_trt_inflight_batches,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            random_chunk=args.random_chunk,
            dedupe=dedupe,
            max_utterances=args.max_utterances,
            device=str(device),
            warmup_batches=args.warmup_batches,
            onnx_providers_requested=args.device,
            torch_num_threads=args.torch_num_threads,
            hostname=socket.gethostname(),
            started_at_utc=started_at,
            num_utterances=n_utts,
            onnx_input_name="",
            onnx_output_names=[],
            onnx_active_providers=[],
            trt_input_name="",
            trt_output_names=[],
        )
        report = {
            "params": asdict(params),
            "timings": asdict(timings),
            "verification": asdict(
                compute_verification_metrics(emb_np, paths, trial_setups, args.audio_root)
            ),
            "environment": {
                "torch_version": torch.__version__,
                "torch_device": str(device),
                "torch_autocast_bfloat16": args.compute_dtype == "bfloat16",
            },
        }
    elif args.backend == "onnx":
        sess = ort.InferenceSession(str(args.onnx_path), providers=providers)
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        it = iter(loader)
        for _ in range(max(0, args.warmup_batches)):
            try:
                batch = next(it)
            except StopIteration:
                break
            x = batch.numpy() if hasattr(batch, "numpy") else np.asarray(batch)
            _try_cuda_sync()
            run_session(sess, x)
            _try_cuda_sync()

        all_emb: list[np.ndarray] = []
        compute_time = 0.0
        n_batches = 0
        t_wall0 = time.perf_counter()
        for batch in tqdm(loader, desc="onnx", total=len(loader)):
            x = batch.numpy() if hasattr(batch, "numpy") else np.asarray(batch)
            _try_cuda_sync()
            t0 = time.perf_counter()
            emb = run_session(sess, x)
            _try_cuda_sync()
            compute_time += time.perf_counter() - t0
            all_emb.append(emb)
            n_batches += 1
        wall = time.perf_counter() - t_wall0

        emb_np = np.concatenate(all_emb, axis=0)
        n_utts = int(emb_np.shape[0])
        seconds_audio = n_utts * float(args.chunk_seconds)
        utt_per_s = n_utts / wall if wall > 0 else float("nan")
        rtf = wall / seconds_audio if seconds_audio > 0 else float("nan")

        timings = BenchmarkTimings(
            wall_time_s=wall,
            compute_time_s=compute_time,
            batches=n_batches,
            utterances_per_s_wall=utt_per_s,
            seconds_audio=seconds_audio,
            rtf=rtf,
            embedding_dim=int(emb_np.shape[1]),
        )

        params = BenchmarkParams(
            backend="onnx",
            model_dir=str(args.model_dir.resolve()),
            preprocessor=args.preprocessor,
            onnx_path=str(args.onnx_path.resolve()),
            trt_engine_path=str(args.trt_engine_path.resolve()),
            torch_encoder_config=None,
            compute_dtype="float32",
            utterance_source=str(args.utterance_list.resolve()),
            list_format=args.list_format,
            audio_root=str(args.audio_root.resolve()),
            extra_voxceleb_o_audio_root=(
                str(args.extra_voxceleb_o_audio_root.resolve())
                if args.extra_voxceleb_o_audio_root is not None
                else None
            ),
            batch_size=args.batch_size,
            num_workers=resolved_num_workers,
            trt_inflight_batches=resolved_trt_inflight_batches,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            random_chunk=args.random_chunk,
            dedupe=dedupe,
            max_utterances=args.max_utterances,
            device=args.device,
            warmup_batches=args.warmup_batches,
            onnx_providers_requested=args.device,
            torch_num_threads=args.torch_num_threads,
            hostname=socket.gethostname(),
            started_at_utc=started_at,
            num_utterances=n_utts,
            onnx_input_name=in_name,
            onnx_output_names=out_names,
            onnx_active_providers=list(sess.get_providers()),
            trt_input_name="",
            trt_output_names=[],
        )

        report = {
            "params": asdict(params),
            "timings": asdict(timings),
            "verification": asdict(
                compute_verification_metrics(emb_np, paths, trial_setups, args.audio_root)
            ),
            "environment": {
                "onnxruntime_available_providers": ort.get_available_providers(),
                "torch_version": torch.__version__,
            },
        }
    else:
        device = resolve_torch_device(args.device)
        if device.type != "cuda":
            raise SystemExit("TensorRT benchmarking requires CUDA (use --device cuda or auto with a GPU).")
        if not args.trt_engine_path.is_file():
            raise SystemExit(f"TensorRT engine not found: {args.trt_engine_path}")

        emb_np, wall, compute_time, n_batches, in_name, out_names = benchmark_trt_loop(
            loader,
            args.trt_engine_path,
            device,
            args.warmup_batches,
            resolved_trt_inflight_batches,
        )
        n_utts = int(emb_np.shape[0])
        seconds_audio = n_utts * float(args.chunk_seconds)
        utt_per_s = n_utts / wall if wall > 0 else float("nan")
        rtf = wall / seconds_audio if seconds_audio > 0 else float("nan")

        timings = BenchmarkTimings(
            wall_time_s=wall,
            compute_time_s=compute_time,
            batches=n_batches,
            utterances_per_s_wall=utt_per_s,
            seconds_audio=seconds_audio,
            rtf=rtf,
            embedding_dim=int(emb_np.shape[1]),
        )

        params = BenchmarkParams(
            backend="trt",
            model_dir=str(args.model_dir.resolve()),
            preprocessor=args.preprocessor,
            onnx_path=str(args.onnx_path.resolve()),
            trt_engine_path=str(args.trt_engine_path.resolve()),
            torch_encoder_config=None,
            compute_dtype="float32",
            utterance_source=str(args.utterance_list.resolve()),
            list_format=args.list_format,
            audio_root=str(args.audio_root.resolve()),
            extra_voxceleb_o_audio_root=(
                str(args.extra_voxceleb_o_audio_root.resolve())
                if args.extra_voxceleb_o_audio_root is not None
                else None
            ),
            batch_size=args.batch_size,
            num_workers=resolved_num_workers,
            trt_inflight_batches=resolved_trt_inflight_batches,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            random_chunk=args.random_chunk,
            dedupe=dedupe,
            max_utterances=args.max_utterances,
            device=str(device),
            warmup_batches=args.warmup_batches,
            onnx_providers_requested=args.device,
            torch_num_threads=args.torch_num_threads,
            hostname=socket.gethostname(),
            started_at_utc=started_at,
            num_utterances=n_utts,
            onnx_input_name="",
            onnx_output_names=[],
            onnx_active_providers=[],
            trt_input_name=in_name,
            trt_output_names=out_names,
        )

        report = {
            "params": asdict(params),
            "timings": asdict(timings),
            "verification": asdict(
                compute_verification_metrics(emb_np, paths, trial_setups, args.audio_root)
            ),
            "environment": {
                "torch_version": torch.__version__,
                "tensorrt_version": getattr(import_tensorrt(), "__version__", "unknown"),
                "torch_device": str(device),
            },
        }

    out = args.report_json
    if out is None:
        out = (
            Path(__file__).resolve().parent
            / "outputs"
            / f"speed_bench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote report to {out}")


if __name__ == "__main__":
    main()
