#!/usr/bin/env python3
"""Build a hard Augment-only dataset from selected VoxCeleb trial utterances.

The script is aligned with ``build_voxceleb_trial_audio_variants.py``:

* reads unique utterances from a VoxCeleb trial text file
* resolves relative utterance paths from ``--wav-root``
* optionally uses ``--run-sh`` to fill MUSAN/RIRS paths

Unlike ``build_voxceleb_trial_audio_variants.py``, this script simulates the
training augmentation policy and writes several augmented copies per selected
utterance, each copy saved as a unique audio file with sidecar metadata.

Example:

```
python scripts/build_voxceleb_hard_augmentation_dataset.py \
    /home/kirill/mnt/ssd_1tb_1/VoxCeleb/voxceleb1_o.txt \
    /home/kirill/mnt/ssd_1tb_1/VoxCeleb/Hard \
    --wav-root /home/kirill/mnt/ssd_1tb_1/VoxCeleb \
    --codec-preset mp3_16k \
    --num-variants-per-utterance 8
```
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import collections

import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from asv.codec.availability import check_preset_available
from asv.codec.preset_loader import load_presets
from asv.codec.pipeline import CodecPipeline
from asv.codec.registry import list_presets, register_preset
from asv.data import CodecAugmentProbabilities, pad_random
from asv.data.decode import ensure_torchaudio_io_compat, load_with_torchcodec


@dataclass
class _LevelDesc:
    name: str
    p: float
    one_of: bool
    shuffle: bool
    transforms: list[tuple[str, torch.nn.Module]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trial_txt", type=Path, help="VoxCeleb trial file (e.g. voxceleb1_o.txt).")
    p.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("/home/kirill/mnt/ssd_1tb_1/VoxCeleb/Hard"),
        help="Directory where augmented outputs are written (default: /home/kirill/mnt/ssd_1tb_1/VoxCeleb/Hard).",
    )
    p.add_argument(
        "--wav-root",
        type=Path,
        default=Path("/home/kirill/mnt/ssd_1tb_1/VoxCeleb"),
        help="Dataset root for trial-relative utterance paths.",
    )
    p.add_argument(
        "--augment-config",
        type=Path,
        default=Path("configs/augment/default.yaml"),
        help="Augmentation config used by training.",
    )
    p.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/default.yaml"),
        help="Data config with transform_probabilities and codec options.",
    )
    p.add_argument(
        "--num-variants-per-utterance",
        type=int,
        default=4,
        help="How many augmented variants to generate for each selected utterance.",
    )
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--sample-rate", type=int, default=None, help="Override sample rate (defaults to data config).")
    p.add_argument(
        "--trim-num-samples",
        type=int,
        default=None,
        help="Override trim/pad length in samples (defaults to data config).",
    )
    p.add_argument(
        "--max-attempts-per-variant",
        type=int,
        default=30,
        help="Retry attempts if a sample stays unmodified (keeps mode-only outputs out).",
    )
    p.add_argument(
        "--run-sh",
        type=Path,
        default=Path("/home/kirill/ASV/run.sh"),
        help="Shell file to read MUSAN_PATH / RIRS_PATH from (defaults to /home/kirill/ASV/run.sh).",
    )
    p.add_argument(
        "--codec-preset",
        type=str,
        default=None,
        help=(
            "Logical codec preset name (e.g. mp3_16k, opus_16k, amr_nb). "
            "Must match a key in the codec registry — not a path to YAML. "
            "Use --codec-presets-file to load extra definitions from a YAML file first."
        ),
    )
    p.add_argument(
        "--codec-presets-file",
        type=Path,
        default=None,
        help=(
            "Optional YAML file in the same format as asv/codec/presets.yaml; "
            "presets are registered (overwrite allowed) before --codec-preset is resolved."
        ),
    )
    p.add_argument(
        "--codec-min-duration-sec",
        type=float,
        default=None,
        help="Override data.codec_min_duration_sec (optional).",
    )
    return p.parse_args()


def _looks_like_codec_preset_path(s: str) -> bool:
    """True if the user likely passed a filesystem path instead of a registry key."""
    p = Path(s)
    if p.suffix.lower() in {".yaml", ".yml"}:
        return True
    if s.startswith(("/", "./", "../", "~/")):
        return True
    try:
        if p.expanduser().is_file():
            return True
    except OSError:
        pass
    return False


def _register_codec_presets_from_yaml(path: Path) -> None:
    loaded = load_presets(path.expanduser().resolve())
    for preset in loaded.values():
        register_preset(preset, overwrite=True)


def _module_name(target: str) -> str:
    return target.split(".")[-1]


def _load_paths_from_run_sh(run_sh_path: Path) -> tuple[str | None, str | None]:
    if not run_sh_path.is_file():
        return None, None
    text = run_sh_path.read_text(encoding="utf-8")
    musan_match = re.search(r'^MUSAN_PATH="([^"]+)"', text, flags=re.MULTILINE)
    rirs_match = re.search(r'^RIRS_PATH="([^"]+)"', text, flags=re.MULTILINE)
    return (musan_match.group(1) if musan_match else None, rirs_match.group(1) if rirs_match else None)


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    try:
        return dict(cfg)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"expected dict-like config node, got {type(cfg)!r}") from exc


def _missing_required_paths(target: str, transform_cfg: dict[str, Any]) -> bool:
    if target.endswith("AddBackgroundNoise"):
        return len(transform_cfg.get("background_paths", [])) == 0
    if target.endswith("ApplyImpulseResponse"):
        return len(transform_cfg.get("ir_paths", [])) == 0
    return False


def _inject_required_paths(
    target: str,
    transform_cfg: dict[str, Any],
    musan_path: str | None,
    rirs_path: str | None,
) -> dict[str, Any]:
    cfg = dict(transform_cfg)
    if target.endswith("AddBackgroundNoise") and len(cfg.get("background_paths", [])) == 0 and musan_path:
        cfg["background_paths"] = [musan_path]
    if target.endswith("ApplyImpulseResponse") and len(cfg.get("ir_paths", [])) == 0 and rirs_path:
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
    candidates: list[Path] = []

    if rel_path.is_absolute():
        candidates.append(rel_path)
    else:
        candidates.append(root / rel_path)
        candidates.append(root / "wav" / rel_path)
        if rel_path.parts and rel_path.parts[0] == "wav":
            candidates.append(root / Path(*rel_path.parts[1:]))
            candidates.append(root / "wav" / Path(*rel_path.parts[1:]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _trim_if_long(waveform: torch.Tensor, trim_num_samples: int) -> torch.Tensor:
    if trim_num_samples <= 0 or waveform.shape[-1] <= trim_num_samples:
        return waveform
    max_start = waveform.shape[-1] - trim_num_samples
    start = random.randint(0, int(max_start))
    return waveform[..., start : start + trim_num_samples]


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        return waveform.mean(dim=0)
    raise ValueError(f"Expected waveform [T] or [C,T], got {tuple(waveform.shape)}")


def _tensor_changed(before: torch.Tensor, after: torch.Tensor) -> bool:
    if before.shape != after.shape:
        return True
    if before.numel() == 0:
        return False
    return not bool(torch.allclose(before, after, atol=1e-6, rtol=1e-6))


def _as_waveform_tensor_output(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict) and "samples" in x:
        return x["samples"]
    raise TypeError(f"unexpected transform output type: {type(x)!r}")


def _run_waveform_transform(
    module: torch.nn.Module,
    samples: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    # Keep default training logic, but keep p handled outside so this is a single-shot call.
    orig_p = getattr(module, "p", None)
    if orig_p is not None:
        module.p = 1.0
    try:
        out = module(samples, sample_rate)
    except TypeError:
        try:
            out = module(samples=samples, sample_rate=sample_rate)
        except TypeError:
            out = module(samples=samples, sample_rate_hz=sample_rate)
    finally:
        if orig_p is not None:
            module.p = orig_p
    return _as_waveform_tensor_output(out)


def _set_variant_seed(base_seed: int, rel_path: str, variant_idx: int, attempt_idx: int) -> int:
    salt = zlib.adler32(f"{rel_path}|{variant_idx}|{attempt_idx}".encode("utf-8")) & 0xFFFFFFFF
    seed = int((base_seed ^ salt) & 0x7FFFFFFF)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def _build_waveform_levels(
    augment_cfg: Any,
    musan_path: str | None,
    rirs_path: str | None,
) -> list[_LevelDesc]:
    levels: list[_LevelDesc] = []
    for level in augment_cfg.get("waveform_levels", []):
        level_dict = _to_plain_dict(level)
        level_name = str(level_dict.get("name", "waveform"))
        level_p = float(level_dict.get("p", 1.0))
        one_of = bool(level_dict.get("one_of", False))
        shuffle = bool(level_dict.get("shuffle", False))
        transforms: list[tuple[str, torch.nn.Module]] = []
        for tcfg in level_dict.get("transforms", []):
            tcfg_dict = _to_plain_dict(tcfg)
            target = str(tcfg_dict.get("_target_", "unknown"))
            tcfg_dict = _inject_required_paths(target, tcfg_dict, musan_path, rirs_path)
            if _missing_required_paths(target, tcfg_dict):
                print(
                    f"[WARN] level={level_name}: skipping transform {target} (missing paths in config).",
                    file=sys.stderr,
                )
                continue
            try:
                module = instantiate(OmegaConf.create(tcfg_dict), _convert_="all")
            except Exception as exc:
                print(
                    f"[WARN] level={level_name}: failed to instantiate {target}: {exc}",
                    file=sys.stderr,
                )
                continue
            module.train()
            transforms.append((_module_name(target), module))
        if transforms:
            levels.append(_LevelDesc(name=level_name, p=level_p, one_of=one_of, shuffle=shuffle, transforms=transforms))
    return levels


def _apply_waveform_levels(
    samples: torch.Tensor,
    sample_rate: int,
    levels: list[_LevelDesc],
) -> tuple[torch.Tensor, list[dict[str, Any]], int]:
    x = samples
    squeezed = False
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
        squeezed = True
    elif x.ndim == 2:
        x = x.unsqueeze(0)
        squeezed = True
    elif x.ndim != 3:
        raise ValueError(f"expected waveform [T], [C,T], or [B,C,T], got {tuple(x.shape)}")

    events: list[dict[str, Any]] = []
    applied_count = 0

    for level in levels:
        if len(level.transforms) == 0 or torch.rand(()) > level.p:
            events.append(
                {
                    "type": "level",
                    "level": level.name,
                    "status": "skipped",
                    "reason": "level_p",
                }
            )
            continue

        modules = list(level.transforms)
        if level.shuffle:
            perm = torch.randperm(len(modules))
            modules = [modules[int(i.item())] for i in perm]
        if level.one_of:
            modules = [modules[int(torch.randint(0, len(modules), (1,)).item())]]

        for transform_name, module in modules:
            before = x.clone()
            module_p = getattr(module, "p", None)
            should_apply = True
            if module_p is not None and module.training and torch.rand(()) > float(module_p):
                should_apply = False

            if not should_apply:
                events.append(
                    {
                        "type": "transform",
                        "level": level.name,
                        "transform": transform_name,
                        "applied": False,
                        "reason": "module_p",
                    }
                )
                continue

            after = _run_waveform_transform(module, x, sample_rate)
            changed = _tensor_changed(before, after)
            event = {
                "type": "transform",
                "level": level.name,
                "transform": transform_name,
                "applied": True,
                "changed": changed,
                "module_p": float(module_p) if module_p is not None else None,
            }
            events.append(event)
            x = after
            if changed:
                applied_count += 1

    if squeezed:
        x = x.squeeze(0)

    return x, events, applied_count


def _codec_roundtrip(
    samples: torch.Tensor,
    sample_rate: int,
    codec_pipeline: CodecPipeline | None,
    min_duration_sec: float | None,
    full_duration_sec: float,
) -> tuple[torch.Tensor, int, dict[str, Any], int]:
    if codec_pipeline is None:
        return samples, sample_rate, {"type": "codec", "applied": False, "reason": "codec_not_configured"}, 0
    if min_duration_sec is not None and full_duration_sec < float(min_duration_sec):
        return (
            samples,
            sample_rate,
            {"type": "codec", "applied": False, "reason": "too_short_for_codec", "min_duration_sec": float(min_duration_sec)},
            0,
        )

    arr = samples.detach().to(dtype=torch.float32, device=torch.device("cpu")).numpy()
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    try:
        decoded_np, out_sr = codec_pipeline.roundtrip(arr, sample_rate)
    except Exception as exc:  # pragma: no cover
        return (
            samples,
            sample_rate,
            {"type": "codec", "applied": False, "reason": "codec_roundtrip_failed", "error": str(exc)},
            0,
        )
    out = torch.from_numpy(decoded_np.astype(np.float32, copy=False))
    if out.ndim == 2 and out.shape[0] == 1:
        out = out.squeeze(0)
    event = {
        "type": "codec",
        "applied": True,
        "name": str(codec_pipeline.preset.name),
        "output_sr": int(out_sr),
    }
    changed = 1 if _tensor_changed(samples, out) else 0
    return out, int(out_sr), event, changed


def main() -> None:
    args = parse_args()
    trial_path = args.trial_txt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    wav_root = args.wav_root.expanduser().resolve()

    if not trial_path.is_file():
        raise SystemExit(f"trial file not found: {trial_path}")

    augment_cfg_path = args.augment_config.expanduser().resolve()
    data_cfg_path = args.data_config.expanduser().resolve()
    if not augment_cfg_path.is_file():
        raise SystemExit(f"augment config not found: {augment_cfg_path}")
    if not data_cfg_path.is_file():
        raise SystemExit(f"data config not found: {data_cfg_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "hard_dataset_variants_metadata.jsonl"
    manifest_fp = open(manifest_path, "w", encoding="utf-8")

    ensure_torchaudio_io_compat()

    try:
        augment_cfg = OmegaConf.load(augment_cfg_path)
        data_cfg = OmegaConf.load(data_cfg_path)
    except Exception as exc:
        manifest_fp.close()
        raise SystemExit(f"failed to load yaml config: {exc}")

    musan_path, rirs_path = _load_paths_from_run_sh(args.run_sh)
    if musan_path is not None:
        os.environ.setdefault("MUSAN_PATH", musan_path)
    if rirs_path is not None:
        os.environ.setdefault("RIRS_PATH", rirs_path)

    # Resolve ${oc.env:...} placeholders in augmentation config.
    augment_cfg_dict: dict[str, Any] = OmegaConf.to_container(augment_cfg, resolve=True)

    sample_rate = int(args.sample_rate) if args.sample_rate is not None else int(data_cfg.get("sample_rate", 16000))
    trim_num_samples = args.trim_num_samples if args.trim_num_samples is not None else data_cfg.get("trim_num_samples")
    trim_num_samples = None if trim_num_samples is None else int(trim_num_samples)
    if trim_num_samples is not None and trim_num_samples <= 0:
        trim_num_samples = None

    codec_min_duration_sec = args.codec_min_duration_sec
    if codec_min_duration_sec is None:
        codec_min_duration_sec = data_cfg.get("codec_min_duration_sec")

    transform_probabilities = data_cfg.get("transform_probabilities")
    if transform_probabilities is None:
        manifest_fp.close()
        raise SystemExit("data config missing transform_probabilities")
    probabilities = CodecAugmentProbabilities(**OmegaConf.to_container(transform_probabilities, resolve=True))

    if args.codec_presets_file is not None:
        presets_yaml = args.codec_presets_file.expanduser().resolve()
        if not presets_yaml.is_file():
            manifest_fp.close()
            raise SystemExit(f"--codec-presets-file not found: {presets_yaml}")
        _register_codec_presets_from_yaml(presets_yaml)
        print(f"[INFO] Loaded codec presets from {presets_yaml}")

    selected_codec = args.codec_preset if args.codec_preset is not None else data_cfg.get("codec_preset")
    codec_pipeline: CodecPipeline | None = None
    if selected_codec is not None:
        preset = str(selected_codec).strip()
        if preset == "":
            selected_codec = None
        else:
            if _looks_like_codec_preset_path(preset):
                manifest_fp.close()
                known = ", ".join(list_presets())
                raise SystemExit(
                    f"--codec-preset must be a logical name (e.g. mp3_16k), not a file path ({preset!r}).\n"
                    f"  Example:  --codec-preset mp3_16k\n"
                    f"  If you have a custom presets YAML:  --codec-presets-file /path/to/presets.yaml --codec-preset <key>\n"
                    f"  Known preset keys: {known}"
                )
            try:
                if check_preset_available(preset):
                    codec_pipeline = CodecPipeline(preset, check_available=False)
                    print(f"[INFO] Using codec preset: {preset}")
                else:
                    print(
                        f"[WARN] codec preset {preset!r} unavailable on this machine; codec mode will be treated as no-op",
                        file=sys.stderr,
                    )
            except Exception as exc:
                manifest_fp.close()
                raise SystemExit(
                    f"Invalid --codec-preset {preset!r}: {exc}\n"
                    f"Known keys: {', '.join(list_presets())}"
                ) from exc

    waveform_levels = _build_waveform_levels(augment_cfg_dict, musan_path, rirs_path)
    if not waveform_levels:
        print("[WARN] no usable waveform augmentation levels found after config checks", file=sys.stderr)

    utterances = parse_unique_utterances(trial_path)
    if not utterances:
        manifest_fp.close()
        raise SystemExit(f"no utterances parsed from {trial_path}")

    total_generated = 0
    total_attempts = 0
    total_failed = 0
    missing_sources = 0

    augment_counter: dict[str, int] = collections.Counter()
    count_by_num_augments: dict[int, int] = collections.Counter()
    mode_request_counter: dict[str, int] = collections.Counter()
    mode_saved_counter: dict[str, int] = collections.Counter()

    print(f"Selected utterances: {len(utterances)}")
    print(f"Output root: {output_dir}")
    print(f"Target samples per utterance: {args.num_variants_per_utterance}")
    print(f"Trim/pad: {trim_num_samples if trim_num_samples is not None else 'disabled'}")

    for rel in tqdm(utterances, desc="utterances", unit="utt"):
        rel_path = Path(rel)
        source_path = resolve_wav_path(wav_root, rel)
        if not source_path.is_file():
            print(f"[WARN] missing source file: {source_path} (from rel={rel})", file=sys.stderr)
            missing_sources += 1
            continue

        try:
            source_waveform, source_sr = load_with_torchcodec(str(source_path), sample_rate=sample_rate)
        except Exception as exc:
            print(f"[WARN] failed to load {source_path}: {exc}", file=sys.stderr)
            missing_sources += 1
            continue
        source_waveform = source_waveform.to(torch.float32)
        full_duration_sec = float(source_waveform.shape[-1]) / float(source_sr) if source_sr > 0 else 0.0

        target_dir = output_dir / rel_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        variant_stem = rel_path.stem

        for variant_idx in range(max(1, int(args.num_variants_per_utterance))):
            saved = False
            variant_attempts = 0
            while not saved and variant_attempts < int(args.max_attempts_per_variant):
                variant_attempts += 1
                total_attempts += 1
                seed = _set_variant_seed(
                    base_seed=int(args.seed),
                    rel_path=str(rel),
                    variant_idx=variant_idx,
                    attempt_idx=variant_attempts,
                )
                mode = probabilities.choose_mode()
                mode_request_counter[mode] += 1

                try:
                    x = _trim_if_long(source_waveform, trim_num_samples) if trim_num_samples else source_waveform
                    sample_sr = int(source_sr)
                    applied_events: list[dict[str, Any]] = []
                    applied_count = 0
                    codec_out = {"type": "codec", "applied": False, "reason": "not_selected"}
                    if mode in {"codec", "codec_augment"}:
                        x, sample_sr, codec_out, codec_count = _codec_roundtrip(
                            x,
                            sample_sr,
                            codec_pipeline,
                            None if codec_min_duration_sec is None else float(codec_min_duration_sec),
                            full_duration_sec=full_duration_sec,
                        )
                        applied_count += codec_count
                        applied_events.append(codec_out)
                        if codec_out.get("applied"):
                            aug_key = f"codec/{codec_out.get('name', 'unknown')}"
                            augment_counter[aug_key] += 1
                    elif mode == "keep":
                        applied_events.append({"type": "mode", "name": "keep", "applied": False, "reason": "keep_mode"})

                    if mode in {"augment", "codec_augment"}:
                        x, level_events, level_count = _apply_waveform_levels(x, sample_sr, waveform_levels)
                        applied_events.extend(level_events)
                        applied_count += level_count
                        for event in level_events:
                            if not event.get("applied", False):
                                continue
                            aug_key = f"transform/{event['level']}/{event['transform']}"
                            augment_counter[aug_key] += 1
                    if applied_count == 0:
                        # We must not write unchanged audios; retry with new randomness.
                        continue

                    x = _to_mono(x)
                    if trim_num_samples is not None:
                        x = pad_random(x, trim_num_samples)

                    out_path = target_dir / f"{variant_stem}__hard_{variant_idx:04d}.wav"
                    sf.write(out_path, x.detach().cpu().numpy().astype(np.float32), samplerate=int(sample_sr), subtype="PCM_16")

                    metadata = {
                        "source_path": str(source_path),
                        "source_rel_path": str(rel_path),
                        "variant_index": variant_idx,
                        "attempt": variant_attempts,
                        "seed": seed,
                        "mode": mode,
                        "sample_rate_in": int(source_sr),
                        "sample_rate_out": int(sample_sr),
                        "full_duration_sec": full_duration_sec,
                        "applied_augmentations": applied_count,
                        "trim_num_samples": trim_num_samples,
                        "events": applied_events,
                    }
                    metadata_path = out_path.with_suffix(".json")
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    manifest_fp.write(json.dumps(metadata) + "\n")
                    manifest_fp.flush()

                    total_generated += 1
                    count_by_num_augments[applied_count] += 1
                    mode_saved_counter[mode] += 1
                    saved = True
                except Exception as exc:  # pragma: no cover
                    print(f"[WARN] attempt failed for {rel} variant {variant_idx}: {exc}", file=sys.stderr)
                    continue

            if not saved:
                total_failed += 1

    manifest_fp.close()

    print("\n=== Hard augmentation summary ===")
    print(f"Output: {output_dir}")
    print(f"Generated variants: {total_generated}")
    print(f"Total attempts: {total_attempts}")
    print(f"Failed variants (no valid augmentation after retries): {total_failed}")
    print(f"Missing/unreadable sources: {missing_sources}")

    print("\nAugmentations per output file:")
    for name, count in sorted(augment_counter.items()):
        print(f"  {name}: {count}")

    print("\nHow many audios had N augmentation events:")
    for count, num_files in sorted(count_by_num_augments.items(), key=lambda item: item[0]):
        print(f"  {count}: {num_files}")

    print("\nMode request counts:")
    for mode_name, count in sorted(mode_request_counter.items()):
        saved_count = mode_saved_counter.get(mode_name, 0)
        print(f"  {mode_name}: requested={count}, saved={saved_count}")


if __name__ == "__main__":
    main()
