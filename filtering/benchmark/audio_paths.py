"""Small helpers for audio files used in benchmark scripts."""

from __future__ import annotations

from pathlib import Path


AUDIO_SUFFIXES = (".wav", ".WAV", ".flac", ".FLAC")


def resolve_audio_path(audio_dir: Path, filename: str) -> Path:
    path = audio_dir / filename
    if path.is_file():
        return path

    stem = Path(filename).stem
    for suffix in AUDIO_SUFFIXES:
        candidate = audio_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Missing audio file for {filename} in {audio_dir}")

