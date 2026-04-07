"""Tests for the export module."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicecraft.export import normalize_audio, save_audio, save_wav


class TestNormalizeAudio:
    def test_normalizes_to_target_peak(self) -> None:
        waveform = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        result = normalize_audio(waveform, target_peak=0.95)
        assert abs(np.max(np.abs(result)) - 0.95) < 1e-6

    def test_handles_silence(self) -> None:
        waveform = np.zeros(100, dtype=np.float32)
        result = normalize_audio(waveform)
        np.testing.assert_array_equal(result, waveform)


class TestSaveWav:
    def test_saves_valid_wav(self, tmp_path: Path) -> None:
        waveform = np.random.randn(24000).astype(np.float32) * 0.5
        out_path = tmp_path / "test_output.wav"
        result = save_wav(waveform, out_path, sr=24000)
        assert result.exists()
        # Verify it's a valid WAV
        data, sr = sf.read(str(result))
        assert sr == 24000
        assert len(data) == len(waveform)


class TestSaveAudio:
    def test_save_wav_format(self, tmp_path: Path) -> None:
        waveform = np.random.randn(24000).astype(np.float32) * 0.5
        out_path = tmp_path / "output.wav"
        result = save_audio(waveform, out_path, fmt="wav")
        assert result.suffix == ".wav"
        assert result.exists()

    def test_save_corrects_extension(self, tmp_path: Path) -> None:
        waveform = np.random.randn(24000).astype(np.float32) * 0.5
        out_path = tmp_path / "output.mp3"
        result = save_audio(waveform, out_path, fmt="wav")
        assert result.suffix == ".wav"
