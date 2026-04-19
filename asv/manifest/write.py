"""Запись ASV-манифестов в Parquet.

Главная функция — :func:`write_manifest`: принимает корень датасета, тип
(``voxceleb1``/``voxceleb2``/``cnceleb*``/``3dspeaker``/``kryp``/``tidy``/``kespeech``),
обходит файлы через соответствующий сканер из :mod:`asv.manifest.scan`,
опционально джойнит длительности и пишет Parquet со сжатием ``zstd``.

Формат Parquet (колонки и типы) — в :mod:`asv.manifest.schema`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from asv.manifest.durations import (
    cnceleb_all_rel_from_any_path,
    cnceleb2_rel_from_any_path,
    kespeech_rel_from_any_path,
    kryp_rel_from_any_path,
    load_duration_map,
    speaker3d_rel_from_any_path,
    tidy_rel_from_any_path,
    voxceleb1_rel_from_any_path,
    voxceleb2_rel_from_any_path,
)
from asv.manifest.scan import (
    iter_3d_speaker_rows,
    iter_cnceleb_all_rows,
    iter_cnceleb2_rows,
    iter_kespeech_rows,
    iter_kryp_train_rows,
    iter_ogg_rows,
    iter_tidy_rows,
    iter_voxceleb1_rows,
    iter_voxceleb2_rows,
)
from asv.manifest.schema import BATCH_ROWS, SCHEMA


def _voxceleb_bare_rel(path_str: str, ds: str) -> str:
    """Strip leading split/prefix so trial utterances match manifest ``rel_path``."""
    p = path_str.replace("\\", "/").strip()
    if ds == "voxceleb1":
        if p.startswith("wav/"):
            return p[4:]
        return p
    if ds == "voxceleb2":
        for prefix in ("dev/aac/", "dev/wav/", "test/aac/", "test/wav/"):
            if p.startswith(prefix):
                return p[len(prefix) :]
        return p
    return p


def _voxceleb_trial_bare_utt_set(trial_paths: Sequence[Path], ds: str) -> set[str]:
    out: set[str] = set()
    for trial_path in trial_paths:
        path = trial_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VoxCeleb trial list not found: {path}")
        for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed trial line in {path}:{line_num} -> {line!r}")
            out.add(_voxceleb_bare_rel(parts[1], ds))
            out.add(_voxceleb_bare_rel(parts[2], ds))
    return out


def write_manifest(
    input_dir: Path,
    output_dir: Path,
    durations_path: Path | None,
    output_name: str,
    *,
    dataset: str = "voxceleb2",
    kryp_probe_durations: bool = False,
    exclude_voxceleb_trial_txt: Path | Sequence[Path] | None = None,
) -> tuple[Path, int]:
    """Построить и записать Parquet-манифест, опционально приджойнив длительности.

    ``exclude_voxceleb_trial_txt`` — только для ``voxceleb1``/``voxceleb2``: исключить
    utterance'ы, встречающиеся в указанных trial-файлах (чтобы валидационные
    сэмплы случайно не попали в трейн)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name

    ds = dataset.lower().strip()
    exclude_bare: set[str] | None = None
    if exclude_voxceleb_trial_txt is not None:
        if ds not in ("voxceleb1", "voxceleb2"):
            raise ValueError(
                "exclude_voxceleb_trial_txt is only supported for dataset voxceleb1 or voxceleb2"
            )
        paths = (
            [exclude_voxceleb_trial_txt]
            if isinstance(exclude_voxceleb_trial_txt, Path)
            else list(exclude_voxceleb_trial_txt)
        )
        if not paths:
            exclude_bare = None
        else:
            exclude_bare = _voxceleb_trial_bare_utt_set(paths, ds)
    duration_map = None
    if durations_path is not None and durations_path.is_file():
        if ds == "voxceleb2":
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=voxceleb2_rel_from_any_path,
            )
        elif ds == "voxceleb1":
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=voxceleb1_rel_from_any_path,
            )
        elif ds in {"cnceleb", "cnceleb2"}:
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=cnceleb2_rel_from_any_path,
            )
        elif ds == "cnceleb_all":
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=cnceleb_all_rel_from_any_path,
            )
        elif ds == "3dspeaker":
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=speaker3d_rel_from_any_path,
            )
        elif ds == "kryp":
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=kryp_rel_from_any_path,
            )
        elif ds == "tidy":
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=tidy_rel_from_any_path,
            )
        elif ds == "kespeech":
            duration_map = load_duration_map(
                durations_path,
                rel_key_fn=kespeech_rel_from_any_path,
            )
        else:
            raise ValueError(f"Unknown dataset={dataset!r} for duration join")

    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
    batch_audio: list[str] = []
    batch_rel: list[str] = []
    batch_spk: list[str] = []
    batch_vid: list[str] = []
    batch_seg: list[str] = []
    batch_aid: list[str] = []
    batch_dur: list[float | None] = []

    def flush() -> None:
        if not batch_audio:
            return
        table = pa.Table.from_pydict(
            {
                "audio_path": batch_audio,
                "rel_path": batch_rel,
                "speaker_id": batch_spk,
                "video_id": batch_vid,
                "segment_id": batch_seg,
                "audio_id": batch_aid,
                "duration_sec": batch_dur,
            },
            schema=SCHEMA,
        )
        writer.write_table(table)
        batch_audio.clear()
        batch_rel.clear()
        batch_spk.clear()
        batch_vid.clear()
        batch_seg.clear()
        batch_aid.clear()
        batch_dur.clear()


    if ds == "voxceleb1":
        row_iter = iter_voxceleb1_rows(input_dir, duration_map)
        desc = "Indexing audio (VoxCeleb1)"
    elif ds == "voxceleb2":
        row_iter = iter_voxceleb2_rows(input_dir, duration_map)
        desc = "Indexing audio (VoxCeleb2)"
    elif ds in {"cnceleb", "cnceleb2"}:
        row_iter = iter_cnceleb2_rows(input_dir, duration_map)
        desc = "Indexing audio (CN-Celeb)"
    elif ds == "cnceleb_all":
        row_iter = iter_cnceleb_all_rows(input_dir, duration_map)
        desc = "Indexing audio (CN-Celeb all splits)"
    elif ds == "3dspeaker":
        row_iter = iter_3d_speaker_rows(input_dir, duration_map)
        desc = "Indexing audio (3D-Speaker)"
    elif ds == "kryp":
        row_iter = iter_kryp_train_rows(
            input_dir,
            duration_map,
            probe_durations=kryp_probe_durations,
        )
        desc = "Indexing audio (Kryp train)"
    elif ds == "tidy":
        row_iter = iter_tidy_rows(input_dir, duration_map)
        desc = "Indexing audio (tidy)"
    elif ds == "kespeech":
        row_iter = iter_kespeech_rows(input_dir, duration_map)
        desc = "Indexing audio (KeSpeech)"
    else:
        raise ValueError(
            "Unknown dataset="
            f"{dataset!r}; expected 'voxceleb1', "
            "'voxceleb2', 'cnceleb', 'cnceleb2', "
            "'cnceleb_all', '3dspeaker', 'kryp', 'tidy', or 'kespeech'"
        )

    n = 0
    for row in tqdm(row_iter, desc=desc, unit="utt"):
        if exclude_bare is not None and _voxceleb_bare_rel(row[1], ds) in exclude_bare:
            continue
        batch_audio.append(row[0])
        batch_rel.append(row[1])
        batch_spk.append(row[2])
        batch_vid.append(row[3])
        batch_seg.append(row[4])
        batch_aid.append(row[5])
        batch_dur.append(row[6])
        n += 1
        if len(batch_audio) >= BATCH_ROWS:
            flush()

    flush()
    writer.close()
    return out_path, n
