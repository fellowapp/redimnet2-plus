"""Сканеры файловой системы для разных типов датасетов.

Каждый датасет имеет свою раскладку файлов. Для каждого — пара функций:
- ``iter_<dataset>_rows(input_dir, duration_by_rel)`` — итератор по строкам манифеста
  ``(audio_path, rel_path, speaker_id, video_id, segment_id, audio_id, duration_sec)``;
- ``count_<dataset>_speakers(input_dir)`` — число уникальных спикеров (для
  ``num_classes``).

Поддерживаемые датасеты: VoxCeleb1/2, CN-Celeb/CN-Celeb2, CN-Celeb all splits,
3D-Speaker, Kryp (через ``train.csv``), tidy, KeSpeech. Для OGG-корпусов с
``audio/<speaker>/<video>/<file>.ogg`` раскладкой — :func:`iter_ogg_rows`.

Нюансы нормализации ``rel_path`` (``wav/``, ``dev/aac/``, ``data/``, ``Audio/``)
нужны для совместимости с duration-parquet и триалами, где пути могут быть
представлены в разных формах."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterator

import soundfile as sf

from asv.manifest.schema import AUDIO_SUBDIR


ManifestRow = tuple[str, str, str, str, str, str, float | None]

_VOXCELEB1_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".m4a", ".mp4")
_VOXCELEB2_AUDIO_EXTS = (".m4a", ".wav", ".mp4", ".flac", ".ogg")
_CNCELEB2_AUDIO_EXTS = (".flac", ".wav", ".ogg", ".m4a", ".mp4")
_3D_SPEAKER_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".m4a", ".mp4")
_TIDY_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".m4a", ".mp4")
_KESPEECH_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".m4a", ".mp4")


def _voxceleb1_audio_root(input_dir: Path) -> Path:
    """Return the ``wav`` root under ``input_dir`` or ``input_dir`` itself when already there."""
    d = input_dir.resolve()
    wav_root = d / "wav"
    if wav_root.is_dir():
        return wav_root
    if d.name == "wav" and d.is_dir():
        return d
    raise FileNotFoundError(f"VoxCeleb1: expected wav/ under {input_dir} (or pass the wav directory itself)")


def _voxceleb2_audio_roots(input_dir: Path) -> list[Path]:
    """Return ``dev/aac``, ``dev/wav``, ``test/aac``, ``test/wav`` roots that exist under ``input_dir``."""
    d = input_dir.resolve()
    roots: list[Path] = []
    for sub in ("dev/aac", "dev/wav", "test/aac", "test/wav"):
        p = d / sub
        if p.is_dir():
            roots.append(p)
    if roots:
        return roots
    # Dataset root might be ``.../VoxCeleb2/dev`` with ``aac`` / ``wav`` inside.
    if d.name == "dev" or d.name == "test":
        for name in ("aac", "wav"):
            p = d / name
            if p.is_dir():
                roots.append(p)
    if not roots:
        raise FileNotFoundError(
            f"VoxCeleb2: expected dev/aac or dev/wav under {input_dir} "
            f"(or pass a root that contains those directories)"
        )
    return roots


def _is_voxceleb_speaker_dir(name: str) -> bool:
    return name.startswith("id") and len(name) > 2 and name[2:].isdigit()


def _is_3d_speaker_dir(name: str) -> bool:
    parts = name.split("_")
    return (
        len(parts) == 3
        and parts[0] == "3D"
        and parts[1] == "SPK"
        and parts[2].isdigit()
    )


def _tidy_language_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) >= 2 and path.parent.is_dir()


def _cnceleb2_audio_root(input_dir: Path) -> Path:
    """Return the ``data`` root under ``input_dir`` or ``input_dir`` itself."""
    d = input_dir.resolve()
    data_root = d / "data"
    if data_root.is_dir():
        return data_root
    if d.name == "data" and d.is_dir():
        return d
    raise FileNotFoundError(
        f"CN-Celeb2: expected data/ under {input_dir} "
        "(or pass the data directory itself)"
    )


def _kespeech_audio_root(input_dir: Path) -> Path:
    """Return ``Audio`` under ``input_dir`` or ``input_dir`` itself."""
    d = input_dir.resolve()
    audio_root = d / "Audio"
    if audio_root.is_dir():
        return audio_root
    if d.name == "Audio" and d.is_dir():
        return d
    raise FileNotFoundError(
        f"KeSpeech: expected Audio/ under {input_dir} "
        "(or pass the Audio directory itself)"
    )


def _normalize_voxceleb2_rel_path(input_dir: Path, rel_posix: str) -> str:
    """Prefix ``rel_path`` with ``dev/`` or ``test/`` when ``input_dir`` is that split root.

    Paths are then consistent with :func:`asv.manifest.durations.voxceleb2_rel_from_any_path`
    and duration parquet keys that contain ``dev/aac/`` or ``test/aac/``.
    """
    root_name = input_dir.resolve().name
    if root_name not in ("dev", "test"):
        return rel_posix
    if rel_posix.startswith(f"{root_name}/"):
        return rel_posix
    return f"{root_name}/{rel_posix}"


def _normalize_voxceleb1_rel_path(input_dir: Path, rel_posix: str) -> str:
    """Prefix ``rel_path`` with ``wav/`` when ``input_dir`` is the wav root."""
    if input_dir.resolve().name != "wav":
        return rel_posix
    if rel_posix.startswith("wav/"):
        return rel_posix
    return f"wav/{rel_posix}"


def _normalize_cnceleb2_rel_path(input_dir: Path, rel_posix: str) -> str:
    """Prefix ``rel_path`` with ``data/`` when ``input_dir`` is the data root."""
    if input_dir.resolve().name != "data":
        return rel_posix
    if rel_posix.startswith("data/"):
        return rel_posix
    return f"data/{rel_posix}"


def _normalize_kespeech_rel_path(input_dir: Path, rel_posix: str) -> str:
    """Prefix ``rel_path`` with ``Audio/`` when needed."""
    if input_dir.resolve().name != "Audio":
        return rel_posix
    if rel_posix.startswith("Audio/"):
        return rel_posix
    return f"Audio/{rel_posix}"


def _parse_cnceleb2_stem(stem: str) -> tuple[str, str]:
    """Map ``genre-session-utterance`` to manifest IDs."""
    parts = stem.rsplit("-", 2)
    if len(parts) == 3:
        genre, session_id, utter_id = parts
        return f"{genre}-{session_id}", utter_id
    return stem, stem


def _parse_3d_speaker_stem(stem: str) -> tuple[str, str] | None:
    """Map ``3D_SPK_xxxxx_utt_Device.._Distance.._Dialect..`` to IDs."""
    parts = stem.split("_")
    if len(parts) < 7:
        return None
    speaker_id = "_".join(parts[:3])
    if not _is_3d_speaker_dir(speaker_id):
        return None
    video_id = parts[3]
    segment_id = "_".join(parts[4:])
    return video_id, segment_id


def _parse_cnceleb_eval_test_stem(stem: str) -> tuple[str, str, str] | None:
    """Map ``idXXXXX-genre-session-utterance`` to manifest IDs."""
    speaker_id, sep, tail = stem.partition("-")
    if not sep or not _is_voxceleb_speaker_dir(speaker_id):
        return None
    video_base, segment_id = _parse_cnceleb2_stem(tail)
    return speaker_id, f"eval-test-{video_base}", segment_id


def _parse_cnceleb_eval_enroll_stem(stem: str) -> tuple[str, str, str] | None:
    """Map ``idXXXXX-enroll`` to manifest IDs."""
    speaker_id, sep, suffix = stem.partition("-")
    if not sep or suffix != "enroll" or not _is_voxceleb_speaker_dir(speaker_id):
        return None
    return speaker_id, "eval-enroll", "enroll"


def iter_voxceleb1_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from VoxCeleb1 under ``wav/<speaker>/<video>/<segment>.wav``.

    Supports the usual layout::

        wav/<idXXXXX>/<youtube_id>/<segment>.wav

    and a flat fallback::

        wav/<idXXXXX>/<stem>.wav

    where ``stem`` may be ``YTID_utt`` for video/segment parsing.

    ``rel_path`` is normalized to start with ``wav/`` when ``input_dir`` is the
    ``wav`` subtree so duration parquet keys remain consistent.
    """
    d = input_dir.resolve()
    root = _voxceleb1_audio_root(input_dir)
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        cur = Path(dirpath)
        for name in filenames:
            lower = name.lower()
            if not any(lower.endswith(ext) for ext in _VOXCELEB1_AUDIO_EXTS):
                continue
            full = (cur / name).resolve()
            try:
                rel = _normalize_voxceleb1_rel_path(d, full.relative_to(d).as_posix())
            except ValueError:
                rel = str(full)
            stem = Path(name).stem

            if _is_voxceleb_speaker_dir(cur.name):
                speaker_id = cur.name
                if "_" in stem:
                    video_id, _, rest = stem.partition("_")
                    segment_id = rest if rest else stem
                else:
                    video_id = stem
                    segment_id = stem
            elif _is_voxceleb_speaker_dir(cur.parent.name):
                speaker_id = cur.parent.name
                video_id = cur.name
                segment_id = stem
            else:
                continue

            audio_id = f"{speaker_id}/{video_id}/{segment_id}"
            dur = duration_by_rel.get(rel) if duration_by_rel is not None else None
            yield (str(full), rel, speaker_id, video_id, segment_id, audio_id, dur)


def iter_voxceleb2_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from VoxCeleb2 under ``dev/aac|wav`` (and test splits).

    Supports the usual layout::

        dev/aac/<idXXXXX>/<youtube_id>/<segment>.m4a

    and a flat fallback::

        dev/aac/<idXXXXX>/<stem>.m4a

    where ``stem`` may be ``YTID_utt`` for video/segment parsing.

    ``rel_path`` is always normalized to start with ``dev/`` or ``test/`` (even when
    ``input_dir`` is only the ``dev`` or ``test`` subtree) so duration parquet keys match.
    """
    d = input_dir.resolve()
    for root in _voxceleb2_audio_roots(input_dir):
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            cur = Path(dirpath)
            for name in filenames:
                lower = name.lower()
                if not any(lower.endswith(ext) for ext in _VOXCELEB2_AUDIO_EXTS):
                    continue
                full = (cur / name).resolve()
                try:
                    rel = _normalize_voxceleb2_rel_path(d, full.relative_to(d).as_posix())
                except ValueError:
                    rel = str(full)
                stem = Path(name).stem

                if _is_voxceleb_speaker_dir(cur.name):
                    # Flat: dev/aac/id00012/foo.m4a
                    speaker_id = cur.name
                    if "_" in stem:
                        video_id, _, rest = stem.partition("_")
                        segment_id = rest if rest else stem
                    else:
                        video_id = stem
                        segment_id = stem
                elif _is_voxceleb_speaker_dir(cur.parent.name):
                    # Nested: dev/aac/id00012/<youtube_id>/00156.m4a
                    speaker_id = cur.parent.name
                    video_id = cur.name
                    segment_id = stem
                else:
                    continue

                audio_id = f"{speaker_id}/{video_id}/{segment_id}"
                dur = duration_by_rel.get(rel) if duration_by_rel is not None else None
                yield (str(full), rel, speaker_id, video_id, segment_id, audio_id, dur)


def count_voxceleb2_speakers(input_dir: Path) -> int:
    """Count speaker folders (``id*``) under discovered VoxCeleb2 audio roots."""
    ids: set[str] = set()
    for root in _voxceleb2_audio_roots(input_dir):
        for child in root.iterdir():
            if child.is_dir():
                name = child.name
                if name.startswith("id") and len(name) > 2 and name[2:].isdigit():
                    ids.add(name)
    return len(ids)


def count_voxceleb1_speakers(input_dir: Path) -> int:
    """Count speaker folders (``id*``) under the discovered VoxCeleb1 wav root."""
    ids: set[str] = set()
    root = _voxceleb1_audio_root(input_dir)
    for child in root.iterdir():
        if child.is_dir() and _is_voxceleb_speaker_dir(child.name):
            ids.add(child.name)
    return len(ids)


def iter_cnceleb2_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from CN-Celeb2 under ``data/<speaker>/*.flac``."""
    d = input_dir.resolve()
    root = _cnceleb2_audio_root(input_dir)
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        cur = Path(dirpath)
        if not _is_voxceleb_speaker_dir(cur.name):
            continue
        speaker_id = cur.name
        for name in filenames:
            lower = name.lower()
            if not any(lower.endswith(ext) for ext in _CNCELEB2_AUDIO_EXTS):
                continue
            full = (cur / name).resolve()
            try:
                rel = _normalize_cnceleb2_rel_path(
                    d,
                    full.relative_to(d).as_posix(),
                )
            except ValueError:
                rel = str(full)
            video_id, segment_id = _parse_cnceleb2_stem(Path(name).stem)
            audio_id = f"{speaker_id}/{video_id}/{segment_id}"
            dur = duration_by_rel.get(rel) if duration_by_rel is not None else None
            yield (str(full), rel, speaker_id, video_id, segment_id, audio_id, dur)


def count_cnceleb2_speakers(input_dir: Path) -> int:
    """Count speaker folders (``id*``) under the CN-Celeb2 data root."""
    ids: set[str] = set()
    root = _cnceleb2_audio_root(input_dir)
    for child in root.iterdir():
        if child.is_dir() and _is_voxceleb_speaker_dir(child.name):
            ids.add(child.name)
    return len(ids)


def iter_cnceleb_all_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield CN-Celeb rows from ``data`` plus ``eval/enroll`` and ``eval/test``."""
    root = input_dir.resolve()

    yield from iter_cnceleb2_rows(root, duration_by_rel)

    eval_root = root / "eval"
    if not eval_root.is_dir():
        return

    for subdir, parser in (
        ("enroll", _parse_cnceleb_eval_enroll_stem),
        ("test", _parse_cnceleb_eval_test_stem),
    ):
        cur_root = eval_root / subdir
        if not cur_root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(cur_root, followlinks=False):
            cur = Path(dirpath)
            for name in filenames:
                lower = name.lower()
                if not any(lower.endswith(ext) for ext in _CNCELEB2_AUDIO_EXTS):
                    continue
                parsed = parser(Path(name).stem)
                if parsed is None:
                    continue
                speaker_id, video_id, segment_id = parsed
                full = (cur / name).resolve()
                try:
                    rel = full.relative_to(root).as_posix()
                except ValueError:
                    rel = str(full)
                audio_id = f"{speaker_id}/{video_id}/{segment_id}"
                dur = duration_by_rel.get(rel) if duration_by_rel is not None else None
                yield (str(full), rel, speaker_id, video_id, segment_id, audio_id, dur)


def count_cnceleb_all_speakers(input_dir: Path) -> int:
    """Count distinct speakers across CN-Celeb ``data`` and split metadata."""
    root = input_dir.resolve()
    ids: set[str] = set()

    try:
        data_root = _cnceleb2_audio_root(root)
    except FileNotFoundError:
        data_root = None
    if data_root is not None:
        for child in data_root.iterdir():
            if child.is_dir() and _is_voxceleb_speaker_dir(child.name):
                ids.add(child.name)

    dev_list = root / "dev" / "dev.lst"
    if dev_list.is_file():
        with dev_list.open(encoding="utf-8") as f:
            for line in f:
                speaker_id = line.strip()
                if _is_voxceleb_speaker_dir(speaker_id):
                    ids.add(speaker_id)

    enroll_root = root / "eval" / "enroll"
    if enroll_root.is_dir():
        for child in enroll_root.iterdir():
            if not child.is_file():
                continue
            parsed = _parse_cnceleb_eval_enroll_stem(child.stem)
            if parsed is not None:
                ids.add(parsed[0])

    test_root = root / "eval" / "test"
    if test_root.is_dir():
        for child in test_root.iterdir():
            if not child.is_file():
                continue
            parsed = _parse_cnceleb_eval_test_stem(child.stem)
            if parsed is not None:
                ids.add(parsed[0])

    return len(ids)


def iter_3d_speaker_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from ``<root>/3D_SPK_xxxxx/*.wav``."""
    root = input_dir.resolve()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        cur = Path(dirpath)
        if not _is_3d_speaker_dir(cur.name):
            continue
        speaker_id = cur.name
        for name in filenames:
            lower = name.lower()
            if not any(lower.endswith(ext) for ext in _3D_SPEAKER_AUDIO_EXTS):
                continue
            parsed = _parse_3d_speaker_stem(Path(name).stem)
            if parsed is None:
                continue
            video_id, segment_id = parsed
            full = (cur / name).resolve()
            try:
                rel = full.relative_to(root).as_posix()
            except ValueError:
                rel = str(full)
            audio_id = f"{speaker_id}/{video_id}/{segment_id}"
            dur = duration_by_rel.get(rel) if duration_by_rel is not None else None
            yield (str(full), rel, speaker_id, video_id, segment_id, audio_id, dur)


def count_3d_speaker_speakers(input_dir: Path) -> int:
    """Count top-level 3D-Speaker speaker folders."""
    root = input_dir.resolve()
    return sum(
        1
        for child in root.iterdir()
        if child.is_dir() and _is_3d_speaker_dir(child.name)
    )


def iter_ogg_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from ``audio/**/*.ogg`` under ``input_dir``."""
    audio_root = input_dir / AUDIO_SUBDIR
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing {AUDIO_SUBDIR}/ under {input_dir}")

    input_dir = input_dir.resolve()
    for dirpath, _dirnames, filenames in os.walk(audio_root, followlinks=False):
        dpath = Path(dirpath)
        for name in filenames:
            if not name.lower().endswith(".ogg"):
                continue
            full = (dpath / name).resolve()
            try:
                rel = full.relative_to(input_dir).as_posix()
            except ValueError:
                rel = str(full)

            parts = Path(rel).parts
            if len(parts) < 4 or parts[0] != AUDIO_SUBDIR:
                continue

            speaker_id, video_id = parts[1], parts[2]
            segment_id = Path(name).stem
            audio_id = f"{speaker_id}/{video_id}/{segment_id}"
            dur = duration_by_rel.get(rel) if duration_by_rel is not None else None
            yield (str(full), rel, speaker_id, video_id, segment_id, audio_id, dur)


def count_kryp_speakers(input_dir: Path) -> int:
    """Count distinct ``speaker_id`` values in ``train.csv`` under the Kryp dataset root."""
    csv_path = input_dir.resolve() / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Kryp: missing {csv_path}")
    speakers: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        sid_key = fields.get("speaker_id")
        if sid_key is None:
            raise ValueError(f"Kryp train.csv must include speaker_id column; got {reader.fieldnames!r}")
        for row in reader:
            speakers.add(str(row[sid_key]).strip())
    return len(speakers)


def iter_kryp_train_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
    *,
    probe_durations: bool = False,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from Kryp speaker-id training data.

    Expected layout::

        <input_dir>/train.csv   # columns: speaker_id, filepath
        <input_dir>/train/<speaker_id>/<NNNNN>.flac

    Each CSV ``filepath`` is relative to ``input_dir`` (e.g. ``train/spk/00000.flac``).

    ``video_id`` is the fixed string ``train``; ``segment_id`` is the FLAC stem.
    """
    root = input_dir.resolve()
    csv_path = root / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Kryp: missing {csv_path}")
    train_root = root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Kryp: missing {train_root}")

    video_id = "train"
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        sid_key = fields.get("speaker_id")
        fp_key = fields.get("filepath")
        if sid_key is None or fp_key is None:
            raise ValueError(
                f"Kryp train.csv must include speaker_id and filepath columns; got {reader.fieldnames!r}"
            )
        for row in reader:
            rel = str(row[fp_key]).strip().replace("\\", "/")
            if not rel:
                continue
            speaker_id = str(row[sid_key]).strip()
            if not speaker_id:
                continue
            full = (root / rel).resolve()
            if not full.is_file():
                continue
            segment_id = full.stem
            audio_id = f"{speaker_id}/{video_id}/{segment_id}"
            dur: float | None = None
            if duration_by_rel is not None:
                dur = duration_by_rel.get(rel)
            if dur is None and probe_durations:
                info = sf.info(str(full))
                dur = float(info.frames) / float(info.samplerate)
            yield (str(full), rel, speaker_id, video_id, segment_id, audio_id, dur)


def iter_tidy_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from ``<root>/<speaker>/<lang>/<lang>_<utt>.wav``."""
    root = input_dir.resolve()
    for speaker_dir in sorted(root.iterdir()):
        if (
            not speaker_dir.is_dir()
            or not _is_voxceleb_speaker_dir(speaker_dir.name)
        ):
            continue
        speaker_id = speaker_dir.name
        for language_dir in sorted(speaker_dir.iterdir()):
            if not _tidy_language_dir(language_dir):
                continue
            video_id = language_dir.name
            for audio_path in sorted(language_dir.iterdir()):
                if not audio_path.is_file():
                    continue
                if not any(
                    audio_path.name.lower().endswith(ext)
                    for ext in _TIDY_AUDIO_EXTS
                ):
                    continue
                full = audio_path.resolve()
                try:
                    rel = full.relative_to(root).as_posix()
                except ValueError:
                    rel = str(full)
                segment_id = audio_path.stem
                audio_id = f"{speaker_id}/{video_id}/{segment_id}"
                dur = (
                    duration_by_rel.get(rel)
                    if duration_by_rel is not None
                    else None
                )
                yield (
                    str(full),
                    rel,
                    speaker_id,
                    video_id,
                    segment_id,
                    audio_id,
                    dur,
                )


def iter_kespeech_rows(
    input_dir: Path,
    duration_by_rel: dict[str, float] | None,
) -> Iterator[ManifestRow]:
    """Yield manifest rows from ``Audio/<speaker>/<phase>/<utt>.wav``."""
    d = input_dir.resolve()
    root = _kespeech_audio_root(input_dir)
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for phase_dir in sorted(speaker_dir.iterdir()):
            if not phase_dir.is_dir():
                continue
            video_id = phase_dir.name
            for audio_path in sorted(phase_dir.iterdir()):
                if not audio_path.is_file():
                    continue
                if not any(
                    audio_path.name.lower().endswith(ext)
                    for ext in _KESPEECH_AUDIO_EXTS
                ):
                    continue
                full = audio_path.resolve()
                try:
                    rel = _normalize_kespeech_rel_path(
                        d,
                        full.relative_to(d).as_posix(),
                    )
                except ValueError:
                    rel = str(full)
                segment_id = audio_path.stem
                audio_id = f"{speaker_id}/{video_id}/{segment_id}"
                dur = (
                    duration_by_rel.get(rel)
                    if duration_by_rel is not None
                    else None
                )
                yield (
                    str(full),
                    rel,
                    speaker_id,
                    video_id,
                    segment_id,
                    audio_id,
                    dur,
                )


def count_tidy_speakers(input_dir: Path) -> int:
    """Count top-level speaker folders under the tidy dataset root."""
    root = input_dir.resolve()
    return sum(
        1
        for child in root.iterdir()
        if child.is_dir() and _is_voxceleb_speaker_dir(child.name)
    )


def count_kespeech_speakers(input_dir: Path) -> int:
    """Count top-level speaker folders under the KeSpeech audio root."""
    root = _kespeech_audio_root(input_dir)
    return sum(1 for child in root.iterdir() if child.is_dir())
