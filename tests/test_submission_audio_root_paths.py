from pathlib import Path

from submission.reranking.run_embeddings import _discover_audio_files


def test_discover_audio_files_keeps_audio_root_folder_in_trial_paths(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    nested = audio_root / "speaker_a"
    nested.mkdir(parents=True)
    sample = nested / "clip.wav"
    sample.write_bytes(b"RIFF")

    audio_paths, audio_ids, trial_rel_paths = _discover_audio_files(audio_root)

    assert audio_paths == [str(sample.resolve())]
    assert audio_ids == ["speaker_a/clip.wav"]
    assert trial_rel_paths == ["audio/speaker_a/clip.wav"]
