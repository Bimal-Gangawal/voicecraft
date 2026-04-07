"""Tests for the audio preprocessing module."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicecraft.audio import (
    AudioValidationError,
    get_duration,
    load_audio,
    normalize,
    preprocess,
    resample,
    to_mono,
    trim_silence,
    validate_sample,
)


class TestLoadAudio:
    def test_loads_wav(self, sample_wav: Path) -> None:
        waveform, sr = load_audio(sample_wav)
        assert waveform is not None
        assert sr == 22050
        assert waveform.ndim >= 1

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/file.wav")


class TestToMono:
    def test_mono_passthrough(self) -> None:
        mono = np.random.randn(44100).astype(np.float32)
        result = to_mono(mono)
        np.testing.assert_array_equal(result, mono)

    def test_stereo_to_mono(self) -> None:
        stereo = np.random.randn(2, 44100).astype(np.float32)
        result = to_mono(stereo)
        assert result.ndim == 1
        assert result.shape[0] == 44100

    def test_stereo_averages_channels(self) -> None:
        left = np.ones(100, dtype=np.float32)
        right = np.full(100, 3.0, dtype=np.float32)
        stereo = np.stack([left, right])
        result = to_mono(stereo)
        np.testing.assert_allclose(result, 2.0, atol=1e-6)


class TestResample:
    def test_same_rate_passthrough(self) -> None:
        waveform = np.random.randn(22050).astype(np.float32)
        result = resample(waveform, 22050, 22050)
        np.testing.assert_array_equal(result, waveform)

    def test_downsample(self) -> None:
        waveform = np.random.randn(44100).astype(np.float32)
        result = resample(waveform, 44100, 22050)
        # Should be approximately half the length
        assert abs(len(result) - 22050) < 100

    def test_upsample(self) -> None:
        waveform = np.random.randn(16000).astype(np.float32)
        result = resample(waveform, 16000, 22050)
        assert len(result) > 16000


class TestTrimSilence:
    def test_trims_leading_trailing_silence(self) -> None:
        sr = 22050
        silence = np.zeros(sr, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
        audio = np.concatenate([silence, tone, silence])
        trimmed = trim_silence(audio, sr)
        # Trimmed should be shorter than original
        assert len(trimmed) < len(audio)
        # Trimmed should contain the tone
        assert len(trimmed) >= sr * 0.8


class TestNormalize:
    def test_normalizes_to_unit_peak(self) -> None:
        waveform = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        result = normalize(waveform)
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-6

    def test_handles_silence(self) -> None:
        waveform = np.zeros(100, dtype=np.float32)
        result = normalize(waveform)
        np.testing.assert_array_equal(result, waveform)


class TestGetDuration:
    def test_correct_duration(self) -> None:
        sr = 22050
        duration_s = 5.0
        waveform = np.zeros(int(sr * duration_s), dtype=np.float32)
        assert abs(get_duration(waveform, sr) - duration_s) < 0.01


class TestValidateSample:
    def test_rejects_short_sample(self, short_wav: Path) -> None:
        waveform, sr = load_audio(short_wav)
        waveform = to_mono(waveform)
        with pytest.raises(AudioValidationError, match="too short"):
            validate_sample(waveform, sr)

    def test_warns_on_borderline_sample(self, tmp_path: Path) -> None:
        sr = 22050
        duration = 8.0  # Between MIN (6) and WARN (10)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        warnings = validate_sample(waveform, sr)
        assert any("at least" in w for w in warnings)

    def test_warns_on_low_energy(self, silence_wav: Path) -> None:
        waveform, sr = load_audio(silence_wav)
        waveform = to_mono(waveform)
        warnings = validate_sample(waveform, sr)
        assert any("low energy" in w for w in warnings)

    def test_passes_good_sample(self, sample_wav: Path) -> None:
        waveform, sr = load_audio(sample_wav)
        waveform = to_mono(waveform)
        warnings = validate_sample(waveform, sr)
        # No critical issues, just possibly the duration warning
        assert not any("too short" in w for w in warnings)


class TestPreprocess:
    def test_full_pipeline(self, sample_wav: Path) -> None:
        processed_path, warnings = preprocess(sample_wav)
        assert Path(processed_path).exists()
        # Processed file should be a valid WAV
        waveform, sr = sf.read(processed_path)
        assert sr == 22050
        assert waveform.ndim == 1
        # Clean up
        Path(processed_path).unlink(missing_ok=True)

    def test_stereo_to_mono_pipeline(self, stereo_wav: Path) -> None:
        processed_path, warnings = preprocess(stereo_wav)
        waveform, sr = sf.read(processed_path)
        assert waveform.ndim == 1  # Should be mono
        Path(processed_path).unlink(missing_ok=True)
