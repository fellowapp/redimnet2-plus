"""Optional torch-friendly adapter for ``CodecPipeline``."""

from __future__ import annotations

from typing import Any

from asv.codec.pipeline import CodecPipeline


class TorchCodecPipelineAdapter:
    """Thin adapter for tensor in/tensor out usage."""

    def __init__(self, pipeline: CodecPipeline) -> None:
        self.pipeline = pipeline

    def forward(self, waveform: Any, sample_rate: int) -> tuple[Any, int]:
        import torch

        out, sr = self.pipeline.roundtrip(waveform, sample_rate)
        return torch.from_numpy(out), sr
