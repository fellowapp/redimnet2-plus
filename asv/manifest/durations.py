"""Duration parquet loading helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def audio_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``audio/<speaker>/<video>/<file>.ogg``."""
    p = path_str.replace("\\", "/")
    key = "/audio/"
    i = p.rfind(key)
    if i != -1:
        return p[i + 1:]
    if p.lower().startswith("audio/"):
        return p
    return None


def voxceleb2_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``dev/aac/...`` keys."""
    p = path_str.replace("\\", "/")
    for marker in ("/dev/aac/", "/dev/wav/", "/test/aac/", "/test/wav/"):
        i = p.rfind(marker)
        if i != -1:
            return p[i + 1:]
    for prefix in ("dev/aac/", "dev/wav/", "test/aac/", "test/wav/"):
        if p.startswith(prefix):
            return p
    return None


def voxceleb1_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``wav/...`` style rel keys."""
    p = path_str.replace("\\", "/")
    marker = "/wav/"
    i = p.rfind(marker)
    if i != -1:
        return p[i + 1:]
    if p.startswith("wav/"):
        return p
    return None


def cnceleb2_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``data/<speaker>/<file>.flac``."""
    p = path_str.replace("\\", "/")
    marker = "/data/"
    i = p.rfind(marker)
    if i != -1:
        return p[i + 1:]
    if p.startswith("data/"):
        return p
    return None


def cnceleb_all_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``data/...`` or ``eval/...`` keys."""
    p = path_str.replace("\\", "/")
    best_index = -1
    best_marker: str | None = None
    for marker in ("/data/", "/eval/"):
        i = p.rfind(marker)
        if i > best_index:
            best_index = i
            best_marker = marker
    if best_marker is not None and best_index != -1:
        return p[best_index + 1:]
    for prefix in ("data/", "eval/"):
        if p.startswith(prefix):
            return p
    return None


def speaker3d_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``3D_SPK_xxxxx/file.wav``."""
    p = path_str.replace("\\", "/")
    parts = [part for part in p.split("/") if part]
    for idx, part in enumerate(parts):
        if part.startswith("3D_SPK_") and len(part.split("_")) == 3:
            return "/".join(parts[idx:])
    return None


def kryp_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``train/<speaker>/<file>.flac``."""
    p = path_str.replace("\\", "/")
    marker = "/train/"
    i = p.rfind(marker)
    if i != -1:
        return p[i + 1:]
    if p.startswith("train/"):
        return p
    return None


def tidy_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``idXXXXX/<lang>/<file>.wav``."""
    p = path_str.replace("\\", "/")
    parts = [part for part in p.split("/") if part]
    for idx, part in enumerate(parts):
        if part.startswith("id") and len(part) > 2 and part[2:].isdigit():
            tail = parts[idx:]
            if len(tail) >= 3:
                return "/".join(tail)
    return None


def kespeech_rel_from_any_path(path_str: str) -> str | None:
    """Map a full or partial path to ``Audio/<speaker>/<phase>/<file>.wav``."""
    p = path_str.replace("\\", "/")
    marker = "/Audio/"
    i = p.rfind(marker)
    if i != -1:
        return p[i + 1:]
    if p.startswith("Audio/"):
        return p
    return None


def load_duration_map(
    durations_path: Path,
    *,
    rel_key_fn: Callable[[str], str | None] | None = None,
) -> dict[str, float]:
    """Read durations parquet and return a map keyed by normalized rel path.

    ``rel_key_fn`` maps each ``path`` cell to the same rel string as
    manifest ``rel_path``.
    Default: :func:`audio_rel_from_any_path`.
    """
    if rel_key_fn is None:
        rel_key_fn = audio_rel_from_any_path

    table = pq.read_table(durations_path, columns=["path", "duration"])
    paths = table.column("path").to_pylist()
    durs = table.column("duration").to_pylist()
    out: dict[str, float] = {}
    for p, d in tqdm(
        zip(paths, durs),
        total=len(paths),
        desc="Loading durations",
        unit="row",
    ):
        if p is None:
            continue
        rel = rel_key_fn(str(p))
        if rel is None:
            continue
        out[rel] = float(d)
    return out
