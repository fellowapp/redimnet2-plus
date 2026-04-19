#!/usr/bin/env python3
"""Build per-augmentation and per-codec audio datasets from VoxCeleb trial utterances.

Reads unique utterance paths from a trial file (e.g. ``voxceleb1_o.txt``), loads each
waveform from ``--wav-root``, and writes:

* One output tree per **waveform** augmentation (from ``configs/augment/default.yaml``
  ``waveform_levels`` only — no spectrogram / ``feature_levels``).
* One output tree per registered **codec** preset (encode/decode round-trip).

Example::

    python scripts/build_voxceleb_trial_audio_variants.py \\
        /path/to/voxceleb1_o.txt /path/to/out \\
        --wav-root /path/to/VoxCeleb1/test/wav

Requires ``MUSAN_PATH`` / ``RIRS_PATH`` in the environment, or ``--run-sh`` (paths are
applied to ``os.environ`` so ``${oc.env:MUSAN_PATH}`` in the YAML can resolve).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

# Repo-root imports (run from ASV directory).
from asv.codec.availability import check_preset_available
from asv.codec.pipeline import CodecPipeline, CodecUnavailableError
from asv.codec.registry import list_presets
from asv.data.decode import ensure_torchaudio_io_compat, load_with_torchcodec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trial_txt", type=Path, help="Trial list, e.g. voxceleb1_o.txt")
    p.add_argument("output_dir", type=Path, help="Root directory for variant datasets")
    p.add_argument(
        "--wav-root",
        type=Path,
        required=True,
        help="Directory containing trial-relative wav paths (e.g. .../test/wav with id10270/...)",
    )
    p.add_argument(
        "--augment-config",
        type=Path,
        default=Path("configs/augment/default.yaml"),
        help="YAML with waveform_levels (feature_levels ignored).",
    )
    p.add_argument("--sample-rate", type=int, default=16000, help="Decode / write sample rate for augments")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed (per-utterance salt applied)")
    p.add_argument(
        "--run-sh",
        type=Path,
        default=None,
        help="Parse MUSAN_PATH / RIRS_PATH from run.sh and set os.environ (needed for ${oc.env:...} in YAML)",
    )
    p.add_argument("--skip-augmentations", action="store_true", help="Only run codec variants")
    p.add_argument("--skip-codecs", action="store_true", help="Only run augmentation variants")
    return p.parse_args()


def _module_name(target: str) -> str:
    return target.split(".")[-1]


def _safe_dir_name(parts: list[str]) -> str:
    raw = "__".join(parts)
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw).strip("_")


def _load_paths_from_run_sh(run_sh_path: Path) -> tuple[str | None, str | None]:
    if not run_sh_path.is_file():
        return None, None
    text = run_sh_path.read_text(encoding="utf-8")
    musan_match = re.search(r'^MUSAN_PATH="([^"]+)"', text, flags=re.MULTILINE)
    rirs_match = re.search(r'^RIRS_PATH="([^"]+)"', text, flags=re.MULTILINE)
    return (musan_match.group(1) if musan_match else None, rirs_match.group(1) if rirs_match else None)


def _missing_required_paths(target: str, transform_cfg: dict[str, Any]) -> bool:
    if target.endswith("AddBackgroundNoise"):
        return len(transform_cfg.get("background_paths", [])) == 0
    if target.endswith("ApplyImpulseResponse"):
        return len(transform_cfg.get("ir_paths", [])) == 0
    return False


def _inject_paths(
    target: str,
    transform_cfg: dict[str, Any],
    musan_path: str | None,
    rirs_path: str | None,
) -> dict[str, Any]:
    cfg = dict(transform_cfg)
    if target.endswith("AddBackgroundNoise") and not cfg.get("background_paths") and musan_path:
        cfg["background_paths"] = [musan_path]
    if target.endswith("ApplyImpulseResponse") and not cfg.get("ir_paths") and rirs_path:
        cfg["ir_paths"] = [rirs_path]
    return cfg


def parse_unique_utterances(trial_path: Path) -> list[str]:
    paths: set[str] = set()
    for line_num, line in enumerate(trial_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 3:
            raise ValueError(f"Malformed trial line {trial_path}:{line_num}: {line!r}")
        paths.add(parts[1])
        paths.add(parts[2])
    return sorted(paths)


def resolve_wav_path(wav_root: Path, rel: str) -> Path:
    root = wav_root.expanduser().resolve()
    rel_path = Path(rel)
    direct = root / rel_path
    if direct.is_file():
        return direct
    if rel_path.parts and rel_path.parts[0] == "wav":
        nested = root / Path(*rel_path.parts[1:])
        if nested.is_file():
            return nested
    return direct


def _ensure_1d_mono(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.squeeze(0)
    if x.ndim == 2:
        x = x.mean(dim=0)
    if x.ndim != 1:
        raise ValueError(f"expected 1D mono waveform, got shape {tuple(x.shape)}")
    return x


def _apply_waveform_transform(module: torch.nn.Module, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
    try:
        out = module(samples, sample_rate)
    except TypeError:
        out = module(samples=samples, sample_rate=sample_rate)
    if isinstance(out, dict) and "samples" in out:
        out = out["samples"]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"unexpected transform output: {type(out)!r}")
    return out


def _write_wav(path: Path, waveform_1d: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = waveform_1d.detach().cpu().numpy().astype(np.float32)
    sf.write(str(path), arr, samplerate=int(sample_rate), subtype="PCM_16")


def _tensor_to_mono_numpy(wave: torch.Tensor) -> np.ndarray:
    x = _ensure_1d_mono(wave)
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


def _load_waveform(wav_path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    wave, sr = load_with_torchcodec(str(wav_path), sample_rate=sample_rate)
    if wave.ndim == 2:
        wave = wave.mean(dim=0)
    return wave.flatten().to(torch.float32), int(sr)


def _set_seed_for_utterance(base_seed: int, rel_path: str) -> None:
    salt = zlib.adler32(rel_path.encode("utf-8")) & 0xFFFFFFFF
    s = (base_seed ^ salt) & 0x7FFFFFFF
    torch.manual_seed(s)
    np.random.seed(s)


def run_augment_variants(
    utterances: list[str],
    wav_root: Path,
    output_dir: Path,
    augment_cfg: Any,
    sample_rate: int,
    base_seed: int,
    musan_path: str | None,
    rirs_path: str | None,
) -> None:
    for level in augment_cfg.get("waveform_levels", []):
        level_name = str(level.get("name", "waveform_level"))
        for tcfg in level.get("transforms", []):
            tcfg_dict: dict[str, Any] = OmegaConf.to_container(tcfg, resolve=True)
            target = str(tcfg_dict.get("_target_", "unknown"))
            tcfg_dict = _inject_paths(target, tcfg_dict, musan_path, rirs_path)
            transform_name = _module_name(target)
            prefix = _safe_dir_name(["aug", level_name, transform_name])
            if _missing_required_paths(target, tcfg_dict):
                print(f"[SKIP] {prefix}: missing background_paths or ir_paths", file=sys.stderr)
                continue
            try:
                module = instantiate(OmegaConf.create(tcfg_dict), _convert_="all")
            except Exception as exc:
                print(f"[SKIP] {prefix}: instantiate failed: {exc}", file=sys.stderr)
                continue
            module.train()
            if hasattr(module, "p"):
                setattr(module, "p", 1.0)

            out_root = output_dir / prefix
            out_root.mkdir(parents=True, exist_ok=True)
            ok = skip = 0
            for rel in tqdm(utterances, desc=prefix, unit="utt"):
                src = resolve_wav_path(wav_root, rel)
                if not src.is_file():
                    skip += 1
                    continue
                try:
                    wave, sr = _load_waveform(src, sample_rate)
                    _set_seed_for_utterance(base_seed, rel)
                    samples = wave.unsqueeze(0).unsqueeze(0)
                    out = _apply_waveform_transform(module, samples, sr)
                    out_1d = _ensure_1d_mono(out)
                    _write_wav(out_root / rel, out_1d, sr)
                    ok += 1
                except Exception as exc:
                    print(f"[WARN] {rel}: {exc}", file=sys.stderr)
                    skip += 1
            print(f"[done] {prefix}: wrote {ok}, skipped {skip}", file=sys.stderr)


def run_codec_variants(
    utterances: list[str],
    wav_root: Path,
    output_dir: Path,
    sample_rate: int,
) -> None:
    ensure_torchaudio_io_compat()
    for preset in list_presets():
        if not check_preset_available(preset):
            print(f"[SKIP] codec__{preset}: not available on this system", file=sys.stderr)
            continue
        try:
            pipeline = CodecPipeline(preset, check_available=False)
        except Exception as exc:
            print(f"[SKIP] codec__{preset}: {exc}", file=sys.stderr)
            continue

        prefix = _safe_dir_name(["codec", preset])
        out_root = output_dir / prefix
        out_root.mkdir(parents=True, exist_ok=True)
        ok = skip = 0
        for rel in tqdm(utterances, desc=prefix, unit="utt"):
            src = resolve_wav_path(wav_root, rel)
            if not src.is_file():
                skip += 1
                continue
            try:
                wave, sr = _load_waveform(src, sample_rate)
                arr = _tensor_to_mono_numpy(wave)
                try:
                    decoded, out_sr = pipeline.roundtrip(arr, sr)
                except CodecUnavailableError as exc:
                    print(f"[WARN] {preset} {rel}: {exc}", file=sys.stderr)
                    skip += 1
                    continue
                if decoded.ndim == 2:
                    decoded = decoded.mean(axis=0)
                out_t = torch.from_numpy(decoded.astype(np.float32, copy=False))
                _write_wav(out_root / rel, out_t.flatten(), int(out_sr))
                ok += 1
            except Exception as exc:
                print(f"[WARN] codec {preset} {rel}: {exc}", file=sys.stderr)
                skip += 1
        print(f"[done] {prefix}: wrote {ok}, skipped {skip}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    trial_path = args.trial_txt.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    wav_root = args.wav_root.expanduser().resolve()

    if not trial_path.is_file():
        raise SystemExit(f"trial file not found: {trial_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    utterances = parse_unique_utterances(trial_path)
    if not utterances:
        raise SystemExit("no utterances parsed from trial file")

    musan_path, rirs_path = _load_paths_from_run_sh(args.run_sh) if args.run_sh else (None, None)
    if args.run_sh and not args.run_sh.is_file():
        print(f"[WARN] --run-sh not found: {args.run_sh}", file=sys.stderr)

    # OmegaConf resolves ${oc.env:MUSAN_PATH} during to_container(resolve=True) on each
    # transform — the variable must exist in os.environ before that runs.
    if musan_path:
        os.environ.setdefault("MUSAN_PATH", musan_path)
    if rirs_path:
        os.environ.setdefault("RIRS_PATH", rirs_path)

    ensure_torchaudio_io_compat()

    if not args.skip_augmentations:
        aug_path = args.augment_config.expanduser().resolve()
        if not aug_path.is_file():
            raise SystemExit(f"augment config not found: {aug_path}")
        augment_cfg = OmegaConf.load(aug_path)
        run_augment_variants(
            utterances,
            wav_root,
            out_root,
            augment_cfg,
            int(args.sample_rate),
            int(args.seed),
            musan_path,
            rirs_path,
        )

    if not args.skip_codecs:
        run_codec_variants(utterances, wav_root, out_root, int(args.sample_rate))


if __name__ == "__main__":
    main()
