#!/usr/bin/env python3
"""Generate per-augmentation outputs for a single input audio file.

Outputs:
- One WAV file for each configured waveform transform.
- One PNG image for each configured spectrogram transform.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from asv.data.collate import _cutmix, _mask_along_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply each augmentation independently to one audio file."
    )
    parser.add_argument("input_audio", type=Path, help="Path to a single audio file.")
    parser.add_argument(
        "--augment_config",
        type=Path,
        default=Path("configs/augment/default.yaml"),
        help="Path to augment YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/augmentation_variants"),
        help="Directory to save generated outputs.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for decoding and waveform outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _module_name(target: str) -> str:
    return target.split(".")[-1]


def _safe_name(parts: list[str]) -> str:
    raw = "__".join(parts)
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw)
    return safe.strip("_")


def _ensure_1d_mono(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.squeeze(0)
    if x.ndim == 2:
        x = x.mean(dim=0)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D/2D/3D waveform tensor, got {tuple(x.shape)}")
    return x


def _write_wav(path: Path, waveform_1d: torch.Tensor, sample_rate: int) -> None:
    arr = waveform_1d.detach().cpu().numpy().astype(np.float32)
    sf.write(str(path), arr, samplerate=sample_rate, format="WAV", subtype="PCM_16")


def _load_waveform(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    # This utility never uses torchaudio for audio loading.
    arr, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1 if waveform.shape[0] > waveform.shape[1] else 0)
    waveform = waveform.to(torch.float32).flatten()
    if src_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), src_sr, sample_rate).squeeze(0)
        src_sr = sample_rate
    return waveform, int(src_sr)


def _write_spec_image(path: Path, spec: torch.Tensor) -> None:
    # spec: [F, T]
    arr = spec.detach().cpu().numpy().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        norm = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    # Flip so low frequencies are at the bottom in the image.
    img = Image.fromarray(np.flipud(norm), mode="L")
    img.save(str(path))


def _apply_waveform_transform(module: torch.nn.Module, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
    try:
        out = module(samples, sample_rate)
    except TypeError:
        out = module(samples=samples, sample_rate=sample_rate)
    if isinstance(out, dict) and "samples" in out:
        out = out["samples"]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Unexpected transform output type: {type(out)!r}")
    return out


def _missing_required_paths(target: str, transform_cfg: dict[str, Any]) -> bool:
    if target.endswith("AddBackgroundNoise"):
        return len(transform_cfg.get("background_paths", [])) == 0
    if target.endswith("ApplyImpulseResponse"):
        return len(transform_cfg.get("ir_paths", [])) == 0
    return False


def _load_paths_from_run_sh(run_sh_path: Path) -> tuple[str | None, str | None]:
    if not run_sh_path.is_file():
        return None, None
    text = run_sh_path.read_text(encoding="utf-8")
    musan_match = re.search(r'^MUSAN_PATH="([^"]+)"', text, flags=re.MULTILINE)
    rirs_match = re.search(r'^RIRS_PATH="([^"]+)"', text, flags=re.MULTILINE)
    musan_path = musan_match.group(1) if musan_match else None
    rirs_path = rirs_match.group(1) if rirs_match else None
    return musan_path, rirs_path


def _inject_required_paths_from_run_sh(
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


def _list_audio_files(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    return sorted([p for p in base.rglob("*") if p.suffix.lower() in exts])


def _fallback_add_background_noise(
    waveform_1d: torch.Tensor,
    sample_rate: int,
    background_paths: list[str],
    snr_db: float = 10.0,
) -> torch.Tensor:
    candidates: list[Path] = []
    for root in background_paths:
        candidates.extend(_list_audio_files(root))
    if not candidates:
        raise RuntimeError("No background audio files found for fallback AddBackgroundNoise")
    noise, noise_sr = _load_waveform(candidates[0], sample_rate=sample_rate)
    if noise_sr != sample_rate:
        noise = torchaudio.functional.resample(noise.unsqueeze(0), noise_sr, sample_rate).squeeze(0)
    if noise.numel() < waveform_1d.numel():
        reps = int(np.ceil(waveform_1d.numel() / float(noise.numel())))
        noise = noise.repeat(reps)
    noise = noise[: waveform_1d.numel()]
    clean_power = waveform_1d.pow(2).mean().clamp_min(1e-12)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power).sqrt()
    mixed = waveform_1d + scale * noise
    return mixed.clamp(-1.0, 1.0)


def _fallback_apply_impulse_response(
    waveform_1d: torch.Tensor,
    sample_rate: int,
    ir_paths: list[str],
) -> torch.Tensor:
    candidates: list[Path] = []
    for root in ir_paths:
        candidates.extend(_list_audio_files(root))
    if not candidates:
        raise RuntimeError("No IR audio files found for fallback ApplyImpulseResponse")
    rir, rir_sr = _load_waveform(candidates[0], sample_rate=sample_rate)
    if rir_sr != sample_rate:
        rir = torchaudio.functional.resample(rir.unsqueeze(0), rir_sr, sample_rate).squeeze(0)
    peak_idx = int(rir.abs().argmax().item())
    rir = rir[peak_idx:]
    if rir.numel() == 0:
        raise RuntimeError("RIR became empty after peak alignment")
    rir = rir[: min(rir.numel(), sample_rate * 2)]
    rir = rir - rir.mean()
    rir = rir / rir.norm(p=2).clamp_min(1e-12)
    conv = torch.nn.functional.conv1d(
        waveform_1d.view(1, 1, -1),
        rir.flip(0).view(1, 1, -1),
        padding=rir.numel() - 1,
    ).view(-1)
    conv = conv[: waveform_1d.numel()]
    in_rms = waveform_1d.pow(2).mean().sqrt().clamp_min(1e-12)
    out_rms = conv.pow(2).mean().sqrt().clamp_min(1e-12)
    conv = conv * (in_rms / out_rms)
    peak = conv.abs().max().clamp_min(1e-12)
    if peak > 0.999:
        conv = conv * (0.999 / peak)
    return conv.clamp(-1.0, 1.0)


def _build_spec_from_waveform(waveform_1d: torch.Tensor, sample_rate: int) -> torch.Tensor:
    # Keep this independent of model code so script can run standalone.
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        power=2.0,
    )
    spec = mel(waveform_1d.unsqueeze(0))  # [1, F, T]
    spec = torch.log(spec + 1e-6).squeeze(0)  # [F, T]
    return spec


def _ensure_torchaudio_compat() -> None:
    # Some environments expose torchaudio APIs that rely on torchcodec, which may
    # be unavailable. For this standalone script we patch required APIs to
    # soundfile-based equivalents.
    def _sf_load(
        path: str,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        start = int(frame_offset)
        frames = -1 if num_frames is None or num_frames < 0 else int(num_frames)
        arr, sr = sf.read(path, dtype="float32", always_2d=True, start=start, frames=frames)
        tensor = torch.from_numpy(arr.T.copy())  # [C, T]
        return tensor, int(sr)

    if hasattr(torchaudio, "info"):
        pass

    class _Info:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
            self.sample_rate = int(sample_rate)
            self.num_frames = int(num_frames)
            self.num_channels = int(num_channels)

    def _info(path: str) -> _Info:
        arr, sr = sf.read(path, dtype="float32", always_2d=True)
        return _Info(sample_rate=sr, num_frames=arr.shape[0], num_channels=arr.shape[1])

    setattr(torchaudio, "info", _info)
    setattr(torchaudio, "load", _sf_load)
    setattr(torchaudio, "load_with_torchcodec", _sf_load)


def _generate_waveform_variants(
    augment_cfg: Any,
    waveform_1d: torch.Tensor,
    sample_rate: int,
    output_dir: Path,
    musan_path: str | None,
    rirs_path: str | None,
) -> tuple[int, int, int]:
    generated = 0
    skipped = 0
    failed = 0

    for level in augment_cfg.get("waveform_levels", []):
        level_name = str(level.get("name", "waveform_level"))
        transforms = level.get("transforms", [])
        for tcfg in transforms:
            tcfg_dict: dict[str, Any] = OmegaConf.to_container(tcfg, resolve=True)
            target = str(tcfg_dict.get("_target_", "unknown"))
            tcfg_dict = _inject_required_paths_from_run_sh(target, tcfg_dict, musan_path, rirs_path)
            transform_name = _module_name(target)
            item_name = _safe_name([level_name, transform_name])
            out_path = output_dir / f"waveform__{item_name}.wav"

            if _missing_required_paths(target, tcfg_dict):
                skipped += 1
                print(f"[SKIP] waveform {item_name}: missing required *_paths in augment config")
                continue

            try:
                module = instantiate(OmegaConf.create(tcfg_dict), _convert_="all")
                module.train()
                if hasattr(module, "p"):
                    setattr(module, "p", 1.0)
                samples = waveform_1d.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                out = _apply_waveform_transform(module, samples, sample_rate)
                out_1d = _ensure_1d_mono(out)
                _write_wav(out_path, out_1d, sample_rate)
                generated += 1
                print(f"[OK]   waveform {item_name} -> {out_path}")
            except Exception as error:  # pragma: no cover
                # Fallbacks for environments where torch_audiomentations+torchaudio
                # cannot execute these two transforms directly.
                try:
                    if target.endswith("AddBackgroundNoise"):
                        out_1d = _fallback_add_background_noise(
                            waveform_1d,
                            sample_rate=sample_rate,
                            background_paths=list(tcfg_dict.get("background_paths", [])),
                            snr_db=10.0,
                        )
                        _write_wav(out_path, out_1d, sample_rate)
                        generated += 1
                        print(f"[OK]   waveform {item_name} (fallback) -> {out_path}")
                        continue
                    if target.endswith("ApplyImpulseResponse"):
                        out_1d = _fallback_apply_impulse_response(
                            waveform_1d,
                            sample_rate=sample_rate,
                            ir_paths=list(tcfg_dict.get("ir_paths", [])),
                        )
                        _write_wav(out_path, out_1d, sample_rate)
                        generated += 1
                        print(f"[OK]   waveform {item_name} (fallback) -> {out_path}")
                        continue
                except Exception as fallback_error:
                    failed += 1
                    print(f"[FAIL] waveform {item_name}: {error}; fallback failed: {fallback_error}")
                    continue
                failed += 1
                print(f"[FAIL] waveform {item_name}: {error}")
    return generated, skipped, failed


def _generate_spectrogram_variants(
    augment_cfg: Any,
    waveform_1d: torch.Tensor,
    sample_rate: int,
    output_dir: Path,
) -> tuple[int, int]:
    generated = 0
    failed = 0

    base_spec = _build_spec_from_waveform(waveform_1d, sample_rate)  # [F, T]
    # Build a pseudo-batch for cutmix to be meaningful.
    alt_spec = torch.roll(base_spec, shifts=max(1, base_spec.shape[1] // 10), dims=1)
    batch_spec = torch.stack([base_spec, alt_spec], dim=0)  # [2, F, T]

    # Save baseline spectrogram image once.
    baseline_path = output_dir / "spectrogram__original.png"
    _write_spec_image(baseline_path, base_spec)
    print(f"[OK]   spectrogram original -> {baseline_path}")

    for level in augment_cfg.get("feature_levels", []):
        if str(level.get("name", "")) != "spec_augment":
            continue
        sa = level.get("spec_augment", {})
        specs = [
            ("freq_mask", int(sa.get("freq_mask", {}).get("max_width", 8))),
            ("time_mask", int(sa.get("time_mask", {}).get("max_width", 10))),
            (
                "cutmix",
                (
                    int(sa.get("cutmix", {}).get("min_frames", 5)),
                    int(sa.get("cutmix", {}).get("max_frames", 30)),
                ),
            ),
        ]
        for name, params in specs:
            out_path = output_dir / f"spectrogram__{name}.png"
            try:
                cur = batch_spec.clone()
                if name == "freq_mask":
                    cur = _mask_along_axis(cur, dim=1, max_width=int(params))
                elif name == "time_mask":
                    cur = _mask_along_axis(cur, dim=2, max_width=int(params))
                else:
                    min_frames, max_frames = params
                    cur = _cutmix(cur, min_frames=min_frames, max_frames=max_frames)
                _write_spec_image(out_path, cur[0])
                generated += 1
                print(f"[OK]   spectrogram {name} -> {out_path}")
            except Exception as error:  # pragma: no cover
                failed += 1
                print(f"[FAIL] spectrogram {name}: {error}")
    return generated, failed


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    _ensure_torchaudio_compat()

    input_audio = args.input_audio.expanduser().resolve()
    augment_config = args.augment_config.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    sample_rate = int(args.sample_rate)

    if not input_audio.is_file():
        print(f"Input audio file does not exist: {input_audio}", file=sys.stderr)
        raise SystemExit(2)
    if not augment_config.is_file():
        print(f"Augment config does not exist: {augment_config}", file=sys.stderr)
        raise SystemExit(2)

    augment_cfg = OmegaConf.load(augment_config)
    waveform, loaded_sr = _load_waveform(input_audio, sample_rate=sample_rate)
    if loaded_sr != sample_rate:
        print(f"[WARN] decoded sample rate={loaded_sr}, expected={sample_rate}")
    waveform_1d = _ensure_1d_mono(waveform)

    output_dir = output_root / input_audio.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    run_sh_path = Path("run.sh").resolve()
    musan_path, rirs_path = _load_paths_from_run_sh(run_sh_path)
    if musan_path:
        print(f"[INFO] Using MUSAN_PATH from run.sh: {musan_path}")
    if rirs_path:
        print(f"[INFO] Using RIRS_PATH from run.sh: {rirs_path}")

    wf_gen, wf_skip, wf_fail = _generate_waveform_variants(
        augment_cfg=augment_cfg,
        waveform_1d=waveform_1d,
        sample_rate=sample_rate,
        output_dir=output_dir,
        musan_path=musan_path,
        rirs_path=rirs_path,
    )
    sp_gen, sp_fail = _generate_spectrogram_variants(
        augment_cfg=augment_cfg,
        waveform_1d=waveform_1d,
        sample_rate=sample_rate,
        output_dir=output_dir,
    )

    print(
        "Done. "
        f"waveform_generated={wf_gen} waveform_skipped={wf_skip} waveform_failed={wf_fail} "
        f"spectrogram_generated={sp_gen} spectrogram_failed={sp_fail} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
