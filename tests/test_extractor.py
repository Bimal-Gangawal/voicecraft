"""Tests for the extractor module (only tests that don't require the XTTS model)."""

import json
from pathlib import Path

from voicecraft.config import VOICES_DIR
from voicecraft.extractor import list_voice_profiles


class TestListVoiceProfiles:
    def test_empty_when_no_profiles(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("voicecraft.extractor.VOICES_DIR", tmp_path / "empty_voices")
        profiles = list_voice_profiles()
        assert profiles == []

    def test_returns_profiles(self, tmp_path: Path, monkeypatch) -> None:
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        profile_dir = voices_dir / "test_voice"
        profile_dir.mkdir()

        metadata = {
            "name": "test_voice",
            "sample_file": "sample.wav",
            "sample_duration_s": 15.5,
            "created_at": "2025-01-01 12:00:00",
            "model": "xtts_v2",
        }
        (profile_dir / "metadata.json").write_text(json.dumps(metadata))

        monkeypatch.setattr("voicecraft.extractor.VOICES_DIR", voices_dir)
        profiles = list_voice_profiles()

        assert len(profiles) == 1
        assert profiles[0]["name"] == "test_voice"
        assert profiles[0]["sample_duration_s"] == 15.5
