from __future__ import annotations

import numpy as np
from pydub import AudioSegment

from asv.codec.conversion import segment_to_float_pcm


def test_segment_to_float_pcm_supports_32bit_samples() -> None:
    samples = np.array([0, 2**29, -(2**29), 2**30], dtype=np.int32)
    seg = AudioSegment(
        data=samples.tobytes(),
        sample_width=4,
        frame_rate=16000,
        channels=1,
    )
    out, sr = segment_to_float_pcm(seg)
    assert sr == 16000
    assert out.shape == (1, samples.size)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
