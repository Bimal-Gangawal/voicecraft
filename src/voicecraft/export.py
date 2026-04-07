"""Export synthesized audio to WAV and/or MP3 formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console

console = Console()

# XTTS v2 output sample rate
OUTPUT_SAMPLE_RATE = 24000


def normalize_audio(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio to a target level."""
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform * (target_peak / peak)
    return waveform


def save_wav(waveform: np.ndarray, path: str | Path, sr: int = OUTPUT_SAMPLE_RATE) -> Path:
    """Save waveform as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = normalize_audio(waveform)
    sf.write(str(path), waveform, sr)
    console.print(f"[green]Saved WAV: {path}[/green]")
    return path


def save_mp3(waveform: np.ndarray, path: str | Path, sr: int = OUTPUT_SAMPLE_RATE) -> Path:
    """Save waveform as an MP3 file (requires ffmpeg)."""
    from pydub import AudioSegment

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = normalize_audio(waveform)

    # Convert float32 [-1, 1] to int16 for pydub
    int16_audio = (waveform * 32767).astype(np.int16)

    audio_segment = AudioSegment(
        data=int16_audio.tobytes(),
        sample_width=2,  # 16-bit
        frame_rate=sr,
        channels=1,
    )

    audio_segment.export(str(path), format="mp3", bitrate="192k")
    console.print(f"[green]Saved MP3: {path}[/green]")
    return path


def save_audio(
    waveform: np.ndarray,
    path: str | Path,
    fmt: str = "wav",
    sr: int = OUTPUT_SAMPLE_RATE,
) -> Path:
    """Save audio in the specified format.

    Args:
        waveform: Audio data as a numpy array.
        path: Output file path. Extension will be adjusted to match fmt.
        fmt: Output format ('wav' or 'mp3').
        sr: Sample rate.

    Returns:
        Path to the saved file.
    """
    path = Path(path)

    # Ensure correct extension
    if fmt == "mp3" and path.suffix != ".mp3":
        path = path.with_suffix(".mp3")
    elif fmt == "wav" and path.suffix != ".wav":
        path = path.with_suffix(".wav")

    if fmt == "mp3":
        return save_mp3(waveform, path, sr)
    else:
        return save_wav(waveform, path, sr)
