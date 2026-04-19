"""ReDimNet2 encoder integration for ASV."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from asv.models.redimnet2_vendor.redimnet2 import ReDimNet2Wrap


_URL_TEMPLATE = "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{model_name}-{dataset}-{train_type}.pt"

_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "b0": {
        "C": 12,
        "out_channels": 64,
        "stages_setup": [
            ((1, 1), 2, 2, [(3, 3)], 36),
            ((2, 1), 3, 1, [(3, 3)], 36),
            ((1, 2), 4, 1, [(3, 3)], 36),
            ((2, 1), 5, 1, [(3, 3)], 36),
            ((1, 2), 4, 1, [(3, 3)], 18),
            ((2, 1), 3, 1, [(3, 3)], 18),
        ],
    },
    "b1": {
        "C": 16,
        "stages_setup": [
            ((1, 1), 2, 2, [(3, 3)], 32),
            ((2, 1), 3, 1, [(3, 3)], 32),
            ((1, 2), 4, 1, [(3, 3)], 32),
            ((2, 1), 5, 1, [(3, 3)], 32),
            ((1, 2), 4, 1, [(3, 3)], 16),
            ((2, 1), 3, 1, [(3, 3)], 16),
        ],
    },
    "b2": {
        "C": 20,
        "stages_setup": [
            ((1, 1), 2, 2, [(3, 5)], 40),
            ((2, 1), 3, 1, [(3, 5)], 30),
            ((1, 2), 4, 1, [(3, 5)], 30),
            ((3, 1), 5, 1, [(3, 5)], 20),
            ((1, 2), 4, 1, [(3, 7)], 20),
            ((2, 1), 3, 1, [(3, 7)], 10),
        ],
    },
    "b3": {"C": 24},
    "b4": {
        "C": 32,
        "stages_setup": [
            ((1, 1), 2, 4, [(3, 3)], 24),
            ((2, 1), 3, 3, [(3, 3)], 24),
            ((1, 2), 4, 2, [(3, 3)], 24),
            ((2, 1), 5, 1, [(3, 3)], 24),
            ((1, 2), 4, 1, [(3, 3)], 24),
            ((2, 1), 3, 1, [(3, 3)], 24),
        ],
    },
    "b5": {
        "C": 48,
        "out_channels": 256,
        "stages_setup": [
            ((1, 1), 2, 4, [(3, 3)], 48),
            ((2, 1), 3, 3, [(3, 3)], 48),
            ((1, 2), 4, 2, [(3, 3)], 48),
            ((2, 1), 5, 1, [(3, 3)], 48),
            ((1, 2), 4, 1, [(3, 3)], 32),
            ((2, 1), 3, 1, [(3, 3)], 32),
        ],
    },
    "b6": {
        "C": 64,
        "out_channels": 224,
        "stages_setup": [
            ((1, 1), 3, 3, [(3, 3)], 64),
            ((2, 1), 4, 2, [(3, 3)], 64),
            ((1, 2), 5, 2, [(3, 3)], 48),
            ((2, 1), 5, 1, [(3, 3)], 48),
            ((1, 2), 4, 0.75, [(3, 3)], 32),
            ((2, 1), 3, 0.5, [(3, 3)], 24),
        ],
    },
}


def _load_release_payload(model_name: str, train_type: str, dataset: str) -> dict[str, Any]:
    if train_type not in {"ptn", "lm"}:
        raise ValueError(f"Unsupported train_type={train_type}; expected 'ptn' or 'lm'")
    url = _URL_TEMPLATE.format(model_name=model_name, dataset=dataset, train_type=train_type)
    payload = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
    if "model_config" not in payload or "state_dict" not in payload:
        raise ValueError("Unexpected checkpoint format")
    return payload


def load_pretrained_redimnet2(
    model_name: str = "b0",
    train_type: str = "lm",
    dataset: str = "vox2",
    strict: bool = True,
) -> ReDimNet2Wrap:
    payload = _load_release_payload(model_name=model_name, train_type=train_type, dataset=dataset)
    model = ReDimNet2Wrap(**payload["model_config"])
    load_result = model.load_state_dict(payload["state_dict"], strict=strict)
    if strict and (load_result.missing_keys or load_result.unexpected_keys):
        raise RuntimeError(
            f"strict load failed; missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )
    return model


class ReDimNet2Encoder(nn.Module):
    """ASV encoder wrapper for scratch/pretrained ReDimNet2.

    The mel-spectrogram frontend (``self.spec_frontend``) is exposed so that
    callers (e.g. :class:`SpectrogramCollator`) can compute spectrograms
    *outside* the model forward pass.  ``forward()`` now expects spectrogram
    input ``[B, F, T]`` or ``[B, 1, F, T]``.
    """

    def __init__(
        self,
        model_name: str = "b0",
        pretrained: bool = False,
        train_type: str = "lm",
        dataset: str = "vox2",
        strict_load: bool = True,
        model_overrides: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        if pretrained:
            model = load_pretrained_redimnet2(
                model_name=model_name,
                train_type=train_type,
                dataset=dataset,
                strict=strict_load,
            )
        else:
            if model_name not in _MODEL_PRESETS:
                raise ValueError(f"Unsupported model_name={model_name}")
            base_cfg = {"causal": "none", "feat_type": "tf"}
            base_cfg.update(_MODEL_PRESETS[model_name])
            if model_overrides:
                base_cfg.update(model_overrides)
            model = ReDimNet2Wrap(**base_cfg)
        self.model = model
        # Expose the spec frontend for external use (collate / preprocessing).
        self.spec_frontend = model.spec
        self.embed_dim = model.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
