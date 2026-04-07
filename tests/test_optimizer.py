"""Tests for the optimizer module (latent optimization without requiring the XTTS model)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from voicecraft.optimizer import (
    OptimizationConfig,
    _audio_to_mel,
    _load_reference_audios,
    restore_original_latents,
)


class TestOptimizationConfig:
    def test_defaults(self) -> None:
        config = OptimizationConfig()
        assert config.steps == 150
        assert config.learning_rate == 1e-3
        assert config.mel_loss_weight == 1.0
        assert config.embedding_loss_weight == 0.5
        assert config.regularization_weight == 0.01

    def test_custom_values(self) -> None:
        config = OptimizationConfig(steps=50, learning_rate=5e-4)
        assert config.steps == 50
        assert config.learning_rate == 5e-4


class TestAudioToMel:
    def test_1d_input(self) -> None:
        audio = torch.randn(22050)  # 1 second at 22050 Hz
        mel = _audio_to_mel(audio, sr=22050)
        assert mel.dim() == 3  # (1, n_mels, time)
        assert mel.shape[1] == 80  # 80 mel bins

    def test_2d_input(self) -> None:
        audio = torch.randn(1, 22050)
        mel = _audio_to_mel(audio, sr=22050)
        assert mel.dim() == 3
        assert mel.shape[1] == 80

    def test_output_is_log_scale(self) -> None:
        audio = torch.randn(22050) * 0.5
        mel = _audio_to_mel(audio, sr=22050)
        # Log mel should contain finite values (no inf/nan from log transform)
        assert torch.isfinite(mel).all()
        # Values should be in a reasonable log-mel range (not raw magnitude)
        assert mel.max() < 20.0


class TestLoadReferenceAudios:
    def _make_wav(self, path: Path, duration: float = 10.0, sr: int = 22050) -> Path:
        """Create a synthetic WAV file."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        waveform = (
            0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.1 * np.sin(2 * np.pi * 440 * t)
            + 0.05 * np.random.randn(len(t))
        ).astype(np.float32)
        sf.write(str(path), waveform, sr)
        return path

    def test_loads_audio_from_directory(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        self._make_wav(audio_dir / "sample1.wav")
        self._make_wav(audio_dir / "sample2.wav")

        refs = _load_reference_audios(audio_dir)
        assert len(refs) == 2
        for waveform, sr in refs:
            assert isinstance(waveform, torch.Tensor)
            assert sr == 22050

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "empty"
        audio_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No audio files"):
            _load_reference_audios(audio_dir)

    def test_respects_max_files(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for i in range(5):
            self._make_wav(audio_dir / f"sample{i}.wav")

        refs = _load_reference_audios(audio_dir, max_files=2)
        assert len(refs) == 2

    def test_ignores_non_audio_files(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        self._make_wav(audio_dir / "sample.wav")
        (audio_dir / "readme.txt").write_text("not audio")
        (audio_dir / "data.json").write_text("{}")

        refs = _load_reference_audios(audio_dir)
        assert len(refs) == 1


class TestRestoreOriginalLatents:
    def test_restore_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("voicecraft.optimizer.VOICES_DIR", tmp_path)

        profile_dir = tmp_path / "test_voice"
        profile_dir.mkdir()

        # Save "optimized" latents
        torch.save(
            {
                "gpt_cond_latent": torch.randn(1, 1024, 4),
                "speaker_embedding": torch.randn(1, 512, 1),
            },
            profile_dir / "latents.pt",
        )

        # Save original backup
        original_latents = {
            "gpt_cond_latent": torch.zeros(1, 1024, 4),
            "speaker_embedding": torch.zeros(1, 512, 1),
        }
        torch.save(original_latents, profile_dir / "latents_original.pt")

        # Save metadata
        metadata = {
            "name": "test_voice",
            "sample_file": "sample.wav",
            "num_samples": 1,
            "sample_duration_s": 15.0,
            "optimized": True,
            "optimization_steps": 150,
            "optimization_loss": 0.005,
            "optimized_at": "2025-01-01 12:00:00",
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))

        restore_original_latents("test_voice")

        # Verify latents were restored
        restored = torch.load(profile_dir / "latents.pt", map_location="cpu", weights_only=True)
        assert torch.allclose(restored["gpt_cond_latent"], torch.zeros(1, 1024, 4))

        # Verify metadata was cleaned up
        meta = json.loads((profile_dir / "metadata.json").read_text())
        assert "optimized" not in meta
        assert "optimization_steps" not in meta
        assert "restored_at" in meta

    def test_restore_missing_backup_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("voicecraft.optimizer.VOICES_DIR", tmp_path)

        profile_dir = tmp_path / "test_voice"
        profile_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No original latents backup"):
            restore_original_latents("test_voice")
