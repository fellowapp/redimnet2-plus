"""Compute speaker embeddings and find k-nearest neighbours for a test CSV.

Inputs:
  --weights      Path to model weights (file: .pt/.pth/.bin/.safetensors),
                 an ``accelerator.save_state`` checkpoint directory, or a
                 checkpoint root containing ``latest.json``.
  --test-csv     CSV with a ``filepath`` column of audio paths (relative to
                 ``--audio-root`` or absolute).
  --audio-root   Root directory used to resolve relative ``filepath`` entries.
  --output-csv   Where to write ``filepath,neighbours``.
  --model-config Hydra-style model yaml (default: configs/model/redimnet2_b6.yaml).
  --top-k        Neighbours per query (default: 10).

The model is built from the model config (encoder + bridge + classifier), but only
the encoder + spec frontend are used to produce L2-normalised embeddings.
Neighbours are ranked by cosine similarity, excluding self.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from asv.data.collate import clone_frozen_frontend, compute_spectrogram_batch, pad_random
from asv.data.decode import load_with_torchcodec
from asv.models.assemble import build_model


DEFAULT_MODEL_CONFIG = _REPO_ROOT / "configs" / "model" / "redimnet2_b6.yaml"


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
    """Return `(state_dict, kind, model_config)` for local checkpoint files."""
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
    """Resolve an `accelerator.save_state()` directory from `asv/train` outputs."""
    if not path.is_dir():
        return None
    if (path / "training_state.json").is_file():
        return path
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
    """Recursively merge local Hydra-style `defaults` entries for model configs."""
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


class _AudioCsvDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rel_paths: list[str],
        audio_root: Path,
        sample_rate: int,
        trim_num_samples: int | None,
    ):
        self._rel_paths = rel_paths
        self._audio_root = audio_root
        self._sample_rate = int(sample_rate)
        self._trim_num_samples = int(trim_num_samples) if trim_num_samples else None

    def __len__(self) -> int:
        return len(self._rel_paths)

    def _resolve_audio_path(self, rel: str) -> Path:
        path = Path(rel)
        if path.is_absolute():
            return path
        direct = self._audio_root / path
        if direct.exists():
            return direct
        if path.parts and path.parts[0] == self._audio_root.name:
            nested = self._audio_root.parent / path
            if nested.exists():
                return nested
            stripped = self._audio_root / Path(*path.parts[1:])
            if stripped.exists():
                return stripped
        return direct

    def __getitem__(self, index: int) -> dict[str, Any]:
        rel = self._rel_paths[index]
        path = self._resolve_audio_path(rel)
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


def _read_filepaths(test_csv: Path) -> list[str]:
    with test_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "filepath" not in reader.fieldnames:
            raise ValueError(
                f"{test_csv} must contain a 'filepath' column, "
                f"got {reader.fieldnames}"
            )
        return [row["filepath"].strip() for row in reader if row.get("filepath", "").strip()]


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


def _topk_neighbours(
    embeddings: torch.Tensor,
    top_k: int,
    device: torch.device,
    chunk: int = 512,
) -> torch.Tensor:
    """Return `[N, top_k] neighbour indices by cosine similarity, excluding self."""
    n = embeddings.shape[0]
    k = min(top_k, max(0, n - 1))
    if k == 0:
        return torch.empty((n, 0), dtype=torch.long)
    emb_dev = embeddings.to(device)
    result = torch.empty((n, k), dtype=torch.long)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = emb_dev[start:end] @ emb_dev.T  # [b, n]
        rows = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), rows] = float("-inf")
        top = torch.topk(sims, k=k, dim=1).indices
        result[start:end] = top.cpu()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root for relative filepaths in the CSV.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
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

    if args.trim_num_samples == 0 and args.batch_size != 1:
        raise SystemExit(
            "When --trim-num-samples=0 (variable length), "
            "--batch-size must be 1."
        )

    accelerator = Accelerator(
        cpu=(args.device == "cpu"),
        mixed_precision="no",
    )
    if (
        args.device not in {"cpu", "cuda", str(accelerator.device)}
        and accelerator.is_main_process
    ):
        print(
            f"[warn] --device={args.device!r} is not used by Accelerator. "
            "Use `accelerate launch` config or CUDA_VISIBLE_DEVICES "
            "to control GPU placement."
        )

    weights_path = args.weights.expanduser().resolve()
    resolved_weights = _resolve_accelerate_checkpoint_dir(weights_path) or weights_path
    state_dict, kind, rd_model_config = _load_state_dict_from_path(resolved_weights)

    model_cfg = _load_model_config(args.model_config)
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

    rel_paths = _read_filepaths(args.test_csv)
    if not rel_paths:
        raise SystemExit(f"No filepaths found in {args.test_csv}")

    dataset = _AudioCsvDataset(
        rel_paths=rel_paths,
        audio_root=args.audio_root.expanduser().resolve(),
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
        num_items=len(rel_paths),
    )
    if not accelerator.is_main_process:
        return
    if embeddings is None:
        raise RuntimeError("Main process did not receive embeddings")
    neighbours = _topk_neighbours(embeddings, args.top_k, accelerator.device)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filepath", "neighbours"])
        for i, rel in enumerate(rel_paths):
            neigh_str = ",".join(str(int(j)) for j in neighbours[i].tolist())
            writer.writerow([rel, neigh_str])

    print(f"Wrote {len(rel_paths)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
