"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Create a synthetic 10-second WAV file for testing."""
    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate a simple tone with some variation to simulate speech-like content
    waveform = (
        0.3 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.sin(2 * np.pi * 880 * t)
        + 0.05 * np.random.randn(len(t))
    ).astype(np.float32)

    path = tmp_path / "test_sample.wav"
    sf.write(str(path), waveform, sr)
    return path


@pytest.fixture
def short_wav(tmp_path: Path) -> Path:
    """Create a 3-second WAV file (too short for voice cloning)."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = tmp_path / "short_sample.wav"
    sf.write(str(path), waveform, sr)
    return path


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    """Create a stereo WAV file."""
    sr = 22050
    duration = 8.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo_sample.wav"
    sf.write(str(path), stereo, sr)
    return path


@pytest.fixture
def silence_wav(tmp_path: Path) -> Path:
    """Create a WAV file that is mostly silence."""
    sr = 22050
    duration = 10.0
    waveform = np.zeros(int(sr * duration), dtype=np.float32)
    # Add a tiny bit of noise so it's not perfectly zero
    waveform += 0.0001 * np.random.randn(len(waveform)).astype(np.float32)
    path = tmp_path / "silence_sample.wav"
    sf.write(str(path), waveform, sr)
    return path
