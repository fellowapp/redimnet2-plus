from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from asv.data import (
    CodecAugmentProbabilities,
    ParquetAudioDataset,
    collate_waveform_batch,
    pad_random,
)


def _write_manifest(path: Path) -> None:
    table = pa.table(
        {
            "audio_path": ["/tmp/a.wav", "/tmp/b.wav"],
            "rel_path": ["audio/spk1/v1/00001.ogg", "audio/spk2/v1/00002.ogg"],
            "speaker_id": ["spk1", "spk2"],
            "video_id": ["v1", "v1"],
            "segment_id": ["00001", "00002"],
            "audio_id": ["spk1/v1/00001", "spk2/v1/00002"],
            "duration_sec": [1.0, 2.0],
        }
    )
    pq.write_table(table, path, row_group_size=1)


def test_probabilities_validate_total() -> None:
    with pytest.raises(ValueError):
        CodecAugmentProbabilities(keep_original=0.5, codec_only=0.5, augment_only=0.5, codec_then_augment=0.5)


def test_dataset_reads_mmap_rows_and_maps_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)

    def fake_decode(path: str, sample_rate: int, *, window_num_samples: int | None = None) -> tuple[torch.Tensor, int]:
        del window_num_samples
        base = 0.1 if path.endswith("a.wav") else 0.2
        return torch.full((16_000,), base), sample_rate

    monkeypatch.setattr("asv.data.parquet_dataset.load_with_torchcodec", fake_decode)
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=8_000,
        probabilities=CodecAugmentProbabilities(
            keep_original=1.0,
            codec_only=0.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        ),
    )

    item0 = ds[0]
    item1 = ds[1]
    assert len(ds) == 2
    assert item0["waveform"].shape[0] == 8_000
    assert item1["waveform"].shape[0] == 8_000
    assert item0["label"] != item1["label"]


def test_dataset_downmixes_multichannel_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)

    def fake_decode(path: str, sample_rate: int, *, window_num_samples: int | None = None) -> tuple[torch.Tensor, int]:
        del path, sample_rate, window_num_samples
        left = torch.ones(16_000, dtype=torch.float32)
        right = torch.full((16_000,), 3.0, dtype=torch.float32)
        return torch.stack([left, right], dim=0), 16_000

    monkeypatch.setattr("asv.data.parquet_dataset.load_with_torchcodec", fake_decode)
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=8_000,
        probabilities=CodecAugmentProbabilities(
            keep_original=1.0,
            codec_only=0.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        ),
    )

    item = ds[0]
    assert item["waveform"].ndim == 1
    assert item["waveform"].shape[0] == 8_000
    assert torch.isclose(item["waveform"].mean(), torch.tensor(2.0), atol=1e-5)


def test_dataset_falls_back_to_previous_item_when_decode_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)

    def fake_decode(
        path: str,
        sample_rate: int,
        *,
        window_num_samples: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        del window_num_samples
        if path.endswith("b.wav"):
            raise RuntimeError("decode fail")
        return torch.full((16_000,), 0.25, dtype=torch.float32), sample_rate

    monkeypatch.setattr("asv.data.parquet_dataset.load_with_torchcodec", fake_decode)
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=4_000,
        probabilities=CodecAugmentProbabilities(
            keep_original=1.0,
            codec_only=0.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        ),
        fallback_on_decode_error=True,
    )
    with caplog.at_level(logging.ERROR):
        item = ds[1]
    assert item["waveform"].shape[0] == 4_000
    assert torch.allclose(item["waveform"], torch.full((4_000,), 0.25))
    assert item["label"] == ds[0]["label"]
    assert "Failed to decode audio with torchcodec" in caplog.text


def test_dataset_falls_forward_when_first_item_decode_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)

    def fake_decode(
        path: str,
        sample_rate: int,
        *,
        window_num_samples: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        del window_num_samples
        if path.endswith("a.wav"):
            raise RuntimeError("decode fail")
        return torch.full((16_000,), 0.5, dtype=torch.float32), sample_rate

    monkeypatch.setattr("asv.data.parquet_dataset.load_with_torchcodec", fake_decode)
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=4_000,
        probabilities=CodecAugmentProbabilities(
            keep_original=1.0,
            codec_only=0.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        ),
        fallback_on_decode_error=True,
    )
    with caplog.at_level(logging.WARNING):
        item = ds[0]
    assert item["waveform"].shape[0] == 4_000
    assert torch.allclose(item["waveform"], torch.full((4_000,), 0.5))
    assert item["label"] == ds[1]["label"]
    assert "Falling back from dataset index 0" in caplog.text


def test_dataset_crops_only_long_audio_before_augment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)
    seen_lengths: list[int] = []

    def fake_decode(path: str, sample_rate: int, *, window_num_samples: int | None = None) -> tuple[torch.Tensor, int]:
        del window_num_samples
        if path.endswith("a.wav"):
            return torch.arange(16_000, dtype=torch.float32), sample_rate
        return torch.arange(2_000, dtype=torch.float32), sample_rate

    class RecordingAugmentor:
        def apply_to_waveform(
            self, waveform: torch.Tensor, sample_rate: int
        ) -> torch.Tensor:
            del sample_rate
            seen_lengths.append(int(waveform.shape[-1]))
            return waveform

    monkeypatch.setattr(
        "asv.data.parquet_dataset.load_with_torchcodec",
        fake_decode,
    )
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=8_000,
        probabilities=CodecAugmentProbabilities(
            keep_original=0.0,
            codec_only=0.0,
            augment_only=1.0,
            codec_then_augment=0.0,
        ),
    )
    ds.augmentor = RecordingAugmentor()

    long_item = ds[0]
    short_item = ds[1]

    assert seen_lengths == [8_000, 2_000]
    assert long_item["waveform"].shape[0] == 8_000
    assert short_item["waveform"].shape[0] == 8_000


def test_dataset_skips_codec_below_duration_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)

    def fake_decode(path: str, sample_rate: int, *, window_num_samples: int | None = None) -> tuple[torch.Tensor, int]:
        del path, window_num_samples
        return torch.zeros(16_000, dtype=torch.float32), sample_rate

    codec_calls: list[int] = []

    def fake_apply_codec(self: ParquetAudioDataset, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        del self
        codec_calls.append(int(waveform.shape[-1]))
        return waveform + 1.0, sample_rate

    monkeypatch.setattr("asv.data.parquet_dataset.load_with_torchcodec", fake_decode)
    monkeypatch.setattr(ParquetAudioDataset, "_apply_codec", fake_apply_codec)
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=8_000,
        codec_preset="mp3_16k",
        codec_min_duration_sec=1.5,
        probabilities=CodecAugmentProbabilities(
            keep_original=0.0,
            codec_only=1.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        ),
    )

    first = ds[0]
    second = ds[1]

    assert codec_calls == [8_000]
    assert torch.all(first["waveform"] < 1.0)
    assert torch.all(second["waveform"] >= 1.0)


def test_dataset_randomly_selects_codec_preset_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)

    def fake_decode(
        path: str,
        sample_rate: int,
        *,
        window_num_samples: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        del path, window_num_samples
        return torch.zeros(16_000, dtype=torch.float32), sample_rate

    class FakeCodecPipeline:
        def __init__(
            self,
            preset_name: str,
            *,
            backend: str = "auto",
            check_available: bool = True,
        ) -> None:
            del backend, check_available
            self.preset_name = preset_name

        def roundtrip(self, arr, sample_rate: int):  # type: ignore[no-untyped-def]
            value = {"preset_a": 1.0, "preset_b": 2.0}[self.preset_name]
            return arr + value, sample_rate

    preset_sequence = iter(["preset_a", "preset_b"])

    def fake_choice(options: list[str]) -> str:
        assert options == ["preset_a", "preset_b"]
        return next(preset_sequence)

    monkeypatch.setattr("asv.data.parquet_dataset.load_with_torchcodec", fake_decode)
    monkeypatch.setattr(
        "asv.data.parquet_dataset.list_presets",
        lambda: ["preset_a", "preset_b"],
    )
    monkeypatch.setattr(
        "asv.data.parquet_dataset.check_preset_available",
        lambda name: True,
    )
    monkeypatch.setattr("asv.data.parquet_dataset.CodecPipeline", FakeCodecPipeline)
    monkeypatch.setattr("asv.data.parquet_dataset.random.choice", fake_choice)
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=8_000,
        codec_preset=None,
        probabilities=CodecAugmentProbabilities(
            keep_original=0.0,
            codec_only=1.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        ),
    )

    first = ds[0]
    second = ds[1]

    assert torch.all(first["waveform"] == 1.0)
    assert torch.all(second["waveform"] == 2.0)


def test_dataset_downmixes_gt_stereo_before_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.parquet"
    _write_manifest(manifest)

    def fake_decode(
        path: str,
        sample_rate: int,
        *,
        window_num_samples: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        del path, sample_rate, window_num_samples
        channels = [
            torch.full((16_000,), float(i), dtype=torch.float32)
            for i in range(8)
        ]
        return torch.stack(channels, dim=0), 16_000

    class FakeCodecPipeline:
        def __init__(
            self,
            preset_name: str,
            *,
            backend: str = "auto",
            check_available: bool = True,
        ) -> None:
            del preset_name, backend, check_available

        def roundtrip(self, arr, sample_rate: int):  # type: ignore[no-untyped-def]
            assert arr.shape == (1, 8_000)
            assert float(arr.mean()) == pytest.approx(3.5)
            return arr, sample_rate

    monkeypatch.setattr("asv.data.parquet_dataset.load_with_torchcodec", fake_decode)
    monkeypatch.setattr("asv.data.parquet_dataset.CodecPipeline", FakeCodecPipeline)
    ds = ParquetAudioDataset(
        manifest,
        sample_rate=16_000,
        trim_num_samples=8_000,
        codec_preset="mp3_16k",
        probabilities=CodecAugmentProbabilities(
            keep_original=0.0,
            codec_only=1.0,
            augment_only=0.0,
            codec_then_augment=0.0,
        ),
    )

    item = ds[0]
    assert item["waveform"].ndim == 1
    assert item["waveform"].shape[0] == 8_000
    assert torch.isclose(item["waveform"].mean(), torch.tensor(3.5), atol=1e-5)


def test_pad_random_trim_and_repeat() -> None:
    x = torch.arange(100, dtype=torch.float32)
    y = pad_random(x, max_len=80)
    assert y.shape[0] == 80
    z = pad_random(torch.tensor([1.0, 2.0, 3.0]), max_len=8)
    assert z.shape[0] == 8
    assert torch.equal(z[:3], torch.tensor([1.0, 2.0, 3.0]))


def test_pad_random_handles_multichannel_time_axis() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    y = pad_random(x, max_len=8)
    assert y.shape == (2, 8)
    assert torch.equal(y[0, :3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(y[1, :3], torch.tensor([10.0, 20.0, 30.0]))


def test_waveform_collate_outputs() -> None:
    batch = [
        {"waveform": torch.randn(6_400), "label": 0},
        {"waveform": torch.randn(6_400), "label": 1},
    ]
    wave_batch = collate_waveform_batch(batch)
    assert wave_batch["inputs"].shape == (2, 1, 6_400)
    assert wave_batch["labels"].shape == (2,)


def test_waveform_collate_outputs_multichannel() -> None:
    batch = [
        {"waveform": torch.randn(2, 6_400), "label": 0},
        {"waveform": torch.randn(2, 6_400), "label": 1},
    ]
    wave_batch = collate_waveform_batch(batch)
    assert wave_batch["inputs"].shape == (2, 2, 6_400)
    assert wave_batch["labels"].shape == (2,)
