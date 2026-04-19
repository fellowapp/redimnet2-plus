"""Инференс полных utterance'ов через PyTorch: эмбеддинги + опционально VoxCeleb EER.

Грузит веса так же, как ``submission/run_submission.py``: принимает либо
Accelerate-чекпоинт-директорию, либо отдельный файл весов (``.safetensors``/``.bin``/``.pt``).
Дальше два режима источника данных:
- ``--manifest-path <parquet>`` — берёт файлы из manifest'а (тот же формат, что в трейне);
- ``--audio-root <dir>`` — рекурсивно сканирует директорию (для submission-трека).

На выходе pickle со словарём ``{embeddings, audio_ids, trial_rel_paths, audio_paths}``;
эмбеддинги L2-нормализованы для дальнейшего реранка."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from asv.data.collate import clone_frozen_frontend, compute_spectrogram_batch, pad_random
from asv.data.decode import load_with_torchcodec
from asv.models.assemble import build_model
from asv.validation.voxceleb import compute_eer, parse_trial_setups


DEFAULT_MODEL_CONFIG = _REPO_ROOT / "configs" / "model" / "redimnet2_b6_sphereface2.yaml"


def _strip_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip common DDP/FSDP/torch.compile wrappers from state dict keys."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            while new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _load_state_dict_from_path(
    path: Path,
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any] | None]:
    """Return ``(state_dict, kind, model_config)`` for local checkpoint files."""
    if path.is_dir():
        candidates = [
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "pytorch_model_fsdp.bin",
            path / "pytorch_model_fsdp_1.bin",
            path / "model.pt",
        ]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"No recognised model file inside {path}. Expected one of: "
                "model.safetensors, pytorch_model.bin, "
                "pytorch_model_fsdp.bin, pytorch_model_fsdp_1.bin, model.pt"
            )
        path = found

    suffix = path.suffix.lower()
    kind = "asv"
    model_config: dict[str, Any] | None = None
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(path))
    else:
        loaded: Any = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            sd = loaded["state_dict"]
            if "model_config" in loaded:
                kind = "redimnet_release"
                model_config = loaded["model_config"]
        elif isinstance(loaded, dict):
            sd = loaded
        else:
            raise ValueError(f"Unsupported checkpoint payload type at {path}: {type(loaded)}")
    return _strip_prefix(sd), kind, model_config


def _resolve_accelerate_checkpoint_dir(path: Path) -> Path | None:
    """Resolve an ``accelerator.save_state()`` directory from training outputs."""
    if not path.is_dir():
        return None
    if (path / "training_state.json").is_file():
        return path.resolve()
    latest = path / "latest.json"
    if not latest.is_file():
        return None
    meta = json.loads(latest.read_text(encoding="utf-8"))
    save_dir = str(meta.get("save_dir", "")).strip()
    if save_dir:
        candidate = path / save_dir
        if candidate.is_dir():
            return candidate.resolve()
    epoch = int(meta["epoch"])
    global_step = int(meta.get("global_step", 0))
    candidate = path / f"epoch_{epoch:04d}_step_{global_step:08d}"
    if candidate.is_dir():
        return candidate.resolve()
    legacy = path / f"epoch_{epoch:04d}"
    if legacy.is_dir():
        return legacy.resolve()
    raise FileNotFoundError(
        f"Could not resolve a checkpoint directory from {latest}. "
        f"Checked {candidate} and {legacy}."
    )


def _compose_model_config(model_config_path: Path) -> Any:
    """Recursively merge local Hydra-style ``defaults`` entries for model configs."""
    raw = OmegaConf.load(str(model_config_path))
    merged = OmegaConf.create()
    defaults = raw.get("defaults") if OmegaConf.is_config(raw) else None
    if defaults:
        for item in defaults:
            if item == "_self_":
                continue
            if isinstance(item, str):
                default_name = item
            elif isinstance(item, dict):
                if "_self_" in item:
                    continue
                if len(item) != 1:
                    raise ValueError(
                        f"Unsupported defaults entry in {model_config_path}: {item}"
                    )
                _, value = next(iter(item.items()))
                default_name = str(value)
            else:
                raise ValueError(
                    f"Unsupported defaults entry type in {model_config_path}: "
                    f"{type(item)}"
                )
            default_path = model_config_path.parent / f"{default_name}.yaml"
            if not default_path.is_file():
                raise FileNotFoundError(
                    f"Model default config not found: {default_path}"
                )
            merged = OmegaConf.merge(
                merged,
                _compose_model_config(default_path),
            )

    current = OmegaConf.create(
        OmegaConf.to_container(raw, resolve=False),
    )
    if "defaults" in current:
        del current["defaults"]
    return OmegaConf.merge(merged, current)


def _load_model_config(model_config_path: Path) -> Any:
    """Load a model yaml and resolve local defaults/interpolations."""
    raw = _compose_model_config(model_config_path)
    wrapped = OmegaConf.create({"model": raw})
    OmegaConf.resolve(wrapped)
    return wrapped.model


def _normalize_trial_rel_path(path_str: str, wav_root: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(wav_root)
        except ValueError:
            path = Path(path.name)
    rel = path.as_posix()
    return rel[4:] if rel.startswith("wav/") else rel


_AUDIO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".flac",
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".opus",
        ".aac",
        ".wma",
    }
)


def _discover_audio_files(audio_root: Path) -> tuple[list[str], list[str], list[str]]:
    """Recursively collect audio files; ids are paths relative to *audio_root*."""
    root = audio_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _AUDIO_SUFFIXES:
            continue
        found.append(path)
    found.sort(key=lambda p: p.as_posix().lower())
    if not found:
        raise RuntimeError(
            f"No audio files with extensions {sorted(_AUDIO_SUFFIXES)} under {root}"
        )
    audio_paths = [str(p.resolve()) for p in found]
    audio_ids = [p.resolve().relative_to(root).as_posix() for p in found]
    root_prefix = root.name.strip()
    if root_prefix:
        trial_rel_paths = [f"{root_prefix}/{audio_id}" for audio_id in audio_ids]
    else:
        trial_rel_paths = list(audio_ids)
    return audio_paths, audio_ids, trial_rel_paths


def _read_manifest_records(
    manifest_path: Path,
    voxceleb_root: Path,
) -> tuple[list[str], list[str], list[str]]:
    table = pq.read_table(
        manifest_path,
        columns=["audio_path", "audio_id", "rel_path"],
    )
    audio_paths = [str(x) for x in table.column("audio_path").to_pylist()]
    audio_ids = [str(x) for x in table.column("audio_id").to_pylist()]
    rel_paths = [str(x) for x in table.column("rel_path").to_pylist()]

    wav_root = voxceleb_root / "wav"
    trial_rel_paths: list[str] = []
    for audio_path, rel_path in zip(audio_paths, rel_paths):
        rel_candidate = rel_path.strip()
        if rel_candidate:
            trial_rel_paths.append(_normalize_trial_rel_path(rel_candidate, wav_root))
            continue
        trial_rel_paths.append(_normalize_trial_rel_path(audio_path, wav_root))
    return audio_paths, audio_ids, trial_rel_paths


class _ManifestAudioDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        audio_paths: list[str],
        sample_rate: int,
        trim_num_samples: int | None,
    ) -> None:
        self._audio_paths = audio_paths
        self._sample_rate = int(sample_rate)
        self._trim_num_samples = int(trim_num_samples) if trim_num_samples else None

    def __len__(self) -> int:
        return len(self._audio_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = Path(self._audio_paths[index]).expanduser().resolve()
        waveform, _ = load_with_torchcodec(str(path), sample_rate=self._sample_rate)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected [T] or [C,T] waveform, got "
                f"{tuple(waveform.shape)} for {path}"
            )
        if self._trim_num_samples is not None:
            waveform = pad_random(waveform, self._trim_num_samples)
        return {"waveform": waveform, "index": index}


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    waveforms = torch.stack([item["waveform"] for item in batch], dim=0).unsqueeze(1)
    indices = torch.tensor([int(item["index"]) for item in batch], dtype=torch.long)
    return {"inputs": waveforms, "indices": indices}


@torch.no_grad()
def _compute_embeddings(
    model: torch.nn.Module,
    spec_frontend: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    num_items: int,
) -> torch.Tensor | None:
    raw_model = accelerator.unwrap_model(model)
    embed_dim = int(getattr(raw_model.encoder, "embed_dim"))
    embeddings_by_index: dict[int, torch.Tensor] = {}
    progress = tqdm(
        loader,
        desc="embeddings",
        disable=not accelerator.is_local_main_process,
    )
    for batch in progress:
        spec = compute_spectrogram_batch(
            batch["inputs"],
            spec_frontend,
            training=False,
        )
        embeds = model(spec, return_embeddings=True)
        embeds = F.normalize(embeds.detach().float(), dim=1)
        gathered_embeds = accelerator.gather_for_metrics(embeds).cpu()
        gathered_indices = accelerator.gather_for_metrics(batch["indices"]).cpu()
        if accelerator.is_main_process:
            for idx in range(gathered_indices.shape[0]):
                item_index = int(gathered_indices[idx].item())
                if item_index not in embeddings_by_index:
                    embeddings_by_index[item_index] = gathered_embeds[idx]

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None
    missing = [idx for idx in range(num_items) if idx not in embeddings_by_index]
    if missing:
        preview = missing[:10]
        raise RuntimeError(
            f"Missing embeddings for {len(missing)} items after distributed gather: "
            f"{preview}{' ...' if len(missing) > len(preview) else ''}"
        )
    ordered = [embeddings_by_index[idx] for idx in range(num_items)]
    return torch.stack(ordered, dim=0) if ordered else torch.empty((0, embed_dim), dtype=torch.float32)


def _default_trial_files(voxceleb_root: Path) -> list[Path]:
    return [
        voxceleb_root / "voxceleb1_o.txt",
        voxceleb_root / "voxceleb_e.txt",
        voxceleb_root / "voxceleb_h.txt",
    ]


def _compute_trial_eers(
    embeddings: torch.Tensor,
    voxceleb_root: Path,
    trial_files: list[Path],
    trial_rel_paths: list[str],
) -> dict[str, float]:
    wav_root = voxceleb_root / "wav"
    _, utt_to_id, setups = parse_trial_setups(
        trial_files=[str(p) for p in trial_files],
        data_root=str(wav_root),
    )
    manifest_rel_to_index = {rel: idx for idx, rel in enumerate(trial_rel_paths)}
    utt_ids_sorted = sorted(utt_to_id.items(), key=lambda item: item[1])
    missing = [rel for rel, _ in utt_ids_sorted if rel not in manifest_rel_to_index]
    if missing:
        preview = missing[:5]
        raise RuntimeError(
            "Trial files reference utterances missing from the embedding manifest: "
            f"{preview}{' ...' if len(missing) > len(preview) else ''}"
        )
    remapped_indices = np.asarray(
        [manifest_rel_to_index[rel] for rel, _ in utt_ids_sorted],
        dtype=np.int32,
    )

    emb = embeddings.cpu().numpy()
    metrics: dict[str, float] = {}
    for setup in setups:
        score_vec = np.sum(
            emb[remapped_indices[setup.utt1_ids]] * emb[remapped_indices[setup.utt2_ids]],
            axis=1,
        )
        metrics[f"{setup.name}.txt"] = compute_eer(setup.labels, score_vec)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Parquet manifest with columns audio_path, audio_id, rel_path.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Directory tree to scan for audio files (.flac/.wav/.mp3/...); "
        "mutually exclusive with --manifest-path.",
    )
    parser.add_argument(
        "--voxceleb-root",
        type=Path,
        default=None,
        help="Required with --manifest-path (trial-relative path normalization / EER).",
    )
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--skip-eer",
        action="store_true",
        help="Compute and save embeddings only; skip trial-file EER evaluation.",
    )
    parser.add_argument(
        "--trial-files",
        type=Path,
        nargs="*",
        default=None,
        help="Defaults to voxceleb1_o/e/h under --voxceleb-root.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--trim-num-samples",
        type=int,
        default=0,
        help="If >0, pad/crop each utterance to this many samples (0 = use full audio; "
        "requires batch_size=1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if (args.manifest_path is None) == (args.audio_root is None):
        raise SystemExit("Specify exactly one of --manifest-path or --audio-root.")
    if args.manifest_path is not None and args.voxceleb_root is None:
        raise SystemExit("--voxceleb-root is required with --manifest-path.")
    if args.audio_root is not None and args.voxceleb_root is not None:
        raise SystemExit("--voxceleb-root is not used with --audio-root; omit it.")
    if args.audio_root is not None and not args.skip_eer:
        raise SystemExit(
            "Trial EER is only supported with --manifest-path and --voxceleb-root; "
            "pass --skip-eer when using --audio-root."
        )

    if args.trim_num_samples == 0 and args.batch_size != 1:
        raise SystemExit(
            "When --trim-num-samples=0 (variable length), "
            "--batch-size must be 1."
        )

    accelerator = Accelerator(cpu=(args.device == "cpu"))

    manifest_path: Path | None = None
    audio_root: Path | None = None
    voxceleb_root: Path | None = None
    if args.manifest_path is not None:
        manifest_path = args.manifest_path.expanduser().resolve()
        voxceleb_root = args.voxceleb_root.expanduser().resolve()  # type: ignore[union-attr]
    else:
        audio_root = args.audio_root.expanduser().resolve()  # type: ignore[union-attr]

    weights_path = args.weights.expanduser().resolve()
    resolved_weights = _resolve_accelerate_checkpoint_dir(weights_path) or weights_path
    state_dict, kind, rd_model_config = _load_state_dict_from_path(resolved_weights)

    model_cfg = _load_model_config(args.model_config.expanduser().resolve())
    model_cfg.classifier = OmegaConf.create(
        {"_target_": "asv.models.classifier.IdentityClassifier"}
    )
    model = build_model(model_cfg)
    if kind == "redimnet_release":
        from asv.models.redimnet2_vendor.redimnet2 import ReDimNet2Wrap

        if accelerator.is_main_process:
            print(
                "[info] rebuilding encoder.model from checkpoint "
                f"model_config (ignoring {args.model_config.name})"
            )
        rebuilt = ReDimNet2Wrap(**rd_model_config)
        missing, unexpected = rebuilt.load_state_dict(state_dict, strict=True)
        model.encoder.model = rebuilt
        model.encoder.spec_frontend = rebuilt.spec
        model.encoder.embed_dim = rebuilt.embed_dim
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if accelerator.is_main_process:
        print(f"[info] loaded {kind} weights from {resolved_weights}")
        if missing:
            print(
                f"[warn] missing keys ({len(missing)}): "
                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            print(
                f"[warn] unexpected keys ({len(unexpected)}): "
                f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}"
            )

    model = accelerator.prepare(model)
    model.eval()
    raw_model = accelerator.unwrap_model(model)
    spec_frontend = clone_frozen_frontend(
        raw_model.encoder.spec_frontend,
        device=accelerator.device,
    )

    if manifest_path is not None:
        assert voxceleb_root is not None
        audio_paths, audio_ids, trial_rel_paths = _read_manifest_records(
            manifest_path=manifest_path,
            voxceleb_root=voxceleb_root,
        )
        if not audio_paths:
            raise SystemExit(f"No rows found in {manifest_path}")
    else:
        assert audio_root is not None
        audio_paths, audio_ids, trial_rel_paths = _discover_audio_files(audio_root)
    if len(set(trial_rel_paths)) != len(trial_rel_paths):
        raise RuntimeError("Duplicate utterance relative paths / ids in the input set")

    dataset = _ManifestAudioDataset(
        audio_paths=audio_paths,
        sample_rate=args.sample_rate,
        trim_num_samples=args.trim_num_samples or None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(accelerator.device.type == "cuda"),
        collate_fn=_collate,
        drop_last=False,
    )
    loader = accelerator.prepare(loader)

    embeddings = _compute_embeddings(
        model,
        spec_frontend,
        loader,
        accelerator,
        num_items=len(audio_paths),
    )
    if not accelerator.is_main_process:
        return
    if embeddings is None:
        raise RuntimeError("Main process did not receive embeddings")

    eers: dict[str, float] = {}
    if not args.skip_eer:
        assert voxceleb_root is not None
        trial_files = (
            [path.expanduser().resolve() for path in args.trial_files]
            if args.trial_files
            else _default_trial_files(voxceleb_root)
        )
        eers = _compute_trial_eers(
            embeddings=embeddings,
            voxceleb_root=voxceleb_root,
            trial_files=trial_files,
            trial_rel_paths=trial_rel_paths,
        )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(resolved_weights),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "audio_root": str(audio_root) if audio_root is not None else None,
        "voxceleb_root": str(voxceleb_root) if voxceleb_root is not None else None,
        "model_config": str(args.model_config.expanduser().resolve()),
        "sample_rate": int(args.sample_rate),
        "trim_num_samples": int(args.trim_num_samples),
        "audio_paths": audio_paths,
        "audio_ids": audio_ids,
        "trial_rel_paths": trial_rel_paths,
        "embeddings": embeddings.numpy(),
        "eers": eers,
    }
    with args.output_pkl.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] wrote embeddings to {args.output_pkl}")
    if not args.skip_eer:
        eer_json_path = args.output_pkl.with_suffix(".eers.json")
        eer_json_path.write_text(json.dumps(eers, indent=2), encoding="utf-8")
        print(f"[done] wrote EER summary to {eer_json_path}")
        for name, value in eers.items():
            print(f"[eer] {name}: {value:.6f}")


if __name__ == "__main__":
    main()
