"""Audio loading, preprocessing, and validation utilities."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

from voicecraft.config import (
    MAX_SAMPLE_DURATION,
    MIN_SAMPLE_DURATION,
    SAMPLE_RATE,
    WARN_SAMPLE_DURATION,
)


class AudioValidationError(Exception):
    """Raised when an audio sample fails validation."""


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file and return (waveform_numpy, sample_rate).

    Supports any format that torchaudio/soundfile/librosa can handle
    (wav, mp3, flac, ogg, m4a, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = torchaudio.load(str(path))
        # Convert to numpy (channels, samples) → (samples,) mono
        waveform_np = waveform.numpy()
    except Exception:
        # Fallback to librosa for formats torchaudio can't handle
        waveform_np, sr = librosa.load(str(path), sr=None, mono=False)
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[np.newaxis, :]

    return waveform_np, sr


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels.

    Expects shape (channels, samples) or (samples,).
    Returns shape (samples,).
    """
    if waveform.ndim == 1:
        return waveform
    return np.mean(waveform, axis=0)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample a mono waveform to the target sample rate."""
    if orig_sr == target_sr:
        return waveform
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(waveform_tensor).squeeze(0).numpy()
    return resampled


def trim_silence(waveform: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Trim leading and trailing silence from a waveform."""
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed


def normalize(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize waveform to [-1, 1]."""
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return waveform


def get_duration(waveform: np.ndarray, sr: int) -> float:
    """Return duration in seconds."""
    return len(waveform) / sr


def validate_sample(waveform: np.ndarray, sr: int) -> list[str]:
    """Validate a voice sample. Returns a list of warning messages.

    Raises AudioValidationError if the sample is too short.
    """
    warnings: list[str] = []
    duration = get_duration(waveform, sr)

    if duration < MIN_SAMPLE_DURATION:
        raise AudioValidationError(
            f"Sample is too short ({duration:.1f}s). "
            f"Minimum required: {MIN_SAMPLE_DURATION:.0f}s."
        )

    if duration < WARN_SAMPLE_DURATION:
        warnings.append(
            f"Sample is only {duration:.1f}s. "
            f"For best results, use at least {WARN_SAMPLE_DURATION:.0f}s of audio."
        )

    # Check for very low energy (might be silence / bad recording)
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 0.005:
        warnings.append("Sample has very low energy — it may be mostly silence or a bad recording.")

    return warnings


def preprocess(path: str | Path) -> tuple[str, list[str]]:
    """Full preprocessing pipeline for a voice sample.

    Returns (path_to_processed_wav, list_of_warnings).
    The processed WAV is saved as a temporary file next to the original.
    """
    path = Path(path)
    warnings: list[str] = []

    # Load
    waveform, sr = load_audio(path)

    # Convert to mono
    waveform = to_mono(waveform)

    # Resample
    waveform = resample(waveform, sr, SAMPLE_RATE)
    sr = SAMPLE_RATE

    # Trim silence
    waveform = trim_silence(waveform, sr)

    # Validate
    warnings.extend(validate_sample(waveform, sr))

    # Truncate if too long
    max_samples = int(MAX_SAMPLE_DURATION * sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
        warnings.append(f"Sample trimmed to {MAX_SAMPLE_DURATION:.0f}s.")

    # Normalize
    waveform = normalize(waveform)

    # Save processed audio to a temp file
    processed_path = path.parent / f"{path.stem}_processed.wav"
    sf.write(str(processed_path), waveform, sr)

    return str(processed_path), warnings
