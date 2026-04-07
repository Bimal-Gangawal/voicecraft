"""Tests for the synthesizer module (text processing — synthesis requires the model)."""

import pytest

from voicecraft.synthesizer import (
    ACCENT_PRESETS,
    VoiceSettings,
    _get_trailing_pause,
    _normalize_text,
    _split_into_chunks,
    _split_into_sentences,
    get_preset,
    PAUSE_COMMA,
    PAUSE_COLON,
    PAUSE_ELLIPSIS,
    PAUSE_SENTENCE_END,
)


class TestNormalizeText:
    def test_expands_ampersand_en(self) -> None:
        assert "and" in _normalize_text("Tom & Jerry", "en")

    def test_expands_ampersand_hi(self) -> None:
        assert "aur" in _normalize_text("Tom & Jerry", "hi")

    def test_expands_percent(self) -> None:
        assert "percent" in _normalize_text("50%", "en")

    def test_adds_period_if_missing_en(self) -> None:
        result = _normalize_text("Hello world", "en")
        assert result.endswith(".")

    def test_adds_purna_viram_if_missing_hi(self) -> None:
        result = _normalize_text("नमस्ते दुनिया", "hi")
        assert result.endswith("।")

    def test_preserves_existing_punctuation(self) -> None:
        result = _normalize_text("Is this a question?", "en")
        assert result.endswith("?")
        assert not result.endswith("?.")

    def test_normalizes_whitespace(self) -> None:
        result = _normalize_text("too   many   spaces", "en")
        assert "   " not in result


class TestGetTrailingPause:
    def test_sentence_end(self) -> None:
        assert _get_trailing_pause("Hello world.", "en") == PAUSE_SENTENCE_END

    def test_question_mark(self) -> None:
        assert _get_trailing_pause("Really?", "en") == PAUSE_SENTENCE_END

    def test_exclamation(self) -> None:
        assert _get_trailing_pause("Wow!", "en") == PAUSE_SENTENCE_END

    def test_comma(self) -> None:
        assert _get_trailing_pause("Well,", "en") == PAUSE_COMMA

    def test_semicolon(self) -> None:
        assert _get_trailing_pause("First part;", "en") == PAUSE_COMMA

    def test_colon(self) -> None:
        assert _get_trailing_pause("Note:", "en") == PAUSE_COLON

    def test_ellipsis(self) -> None:
        assert _get_trailing_pause("Hmm...", "en") == PAUSE_ELLIPSIS

    def test_hindi_purna_viram(self) -> None:
        assert _get_trailing_pause("नमस्ते।", "hi") == PAUSE_SENTENCE_END


class TestSplitIntoChunks:
    def test_short_text_stays_merged(self) -> None:
        text = "Hello world. How are you? I am fine!"
        chunks = _split_into_chunks(text, "en")
        assert len(chunks) == 1
        assert "Hello world." in chunks[0]

    def test_long_multi_sentence_splits(self) -> None:
        text = (
            "This is the first sentence that is reasonably long and detailed. "
            "This is the second sentence with even more detail and explanation. "
            "And here is a third sentence that pushes the total length over the limit. "
            "Finally a fourth sentence that definitely exceeds the chunk size threshold. "
            "Plus a fifth sentence for good measure and to be absolutely sure it splits."
        )
        chunks = _split_into_chunks(text, "en")
        assert len(chunks) >= 2

    def test_hindi_short_text(self) -> None:
        text = "नमस्ते दुनिया। आप कैसे हैं? मैं ठीक हूँ।"
        chunks = _split_into_chunks(text, "hi")
        assert len(chunks) >= 1

    def test_single_sentence(self) -> None:
        text = "Just one sentence"
        chunks = _split_into_chunks(text, "en")
        assert len(chunks) == 1

    def test_long_text_chunking(self) -> None:
        text = "This is a very long sentence that goes on and on " * 20
        chunks = _split_into_chunks(text.strip(), "en")
        # Allow small buffer for edge cases
        assert all(len(c.strip()) <= 260 for c in chunks)

    def test_empty_after_strip(self) -> None:
        text = "   "
        chunks = _split_into_chunks(text, "en")
        assert len(chunks) == 1

    def test_merges_short_fragments(self) -> None:
        text = "Hi. Ok. Sure. Yes."
        chunks = _split_into_chunks(text, "en")
        assert len(chunks) <= 2

    def test_splits_long_sentence_at_commas(self) -> None:
        text = (
            "First clause with lots of words, "
            "second clause with even more words, "
            "third clause that keeps going on, "
            "fourth clause that makes it long, "
            "fifth clause adding more length, "
            "sixth clause to push it over, "
            "seventh clause for good measure, "
            "eighth clause to really exceed it."
        )
        chunks = _split_into_chunks(text, "en")
        assert len(chunks) >= 2

    def test_alias_works(self) -> None:
        # _split_into_sentences is an alias for backwards compat
        text = "Hello world."
        assert _split_into_sentences(text, "en") == _split_into_chunks(text, "en")


class TestVoiceSettings:
    def test_defaults(self) -> None:
        s = VoiceSettings()
        assert s.temperature == 0.75
        assert s.speed == 1.0
        assert s.repetition_penalty == 10.0

    def test_custom_values(self) -> None:
        s = VoiceSettings(temperature=0.5, speed=1.2, repetition_penalty=5.0)
        assert s.temperature == 0.5
        assert s.speed == 1.2
        assert s.repetition_penalty == 5.0


class TestAccentPresets:
    def test_neutral_preset_matches_defaults(self) -> None:
        neutral = ACCENT_PRESETS["neutral"]
        default = VoiceSettings()
        assert neutral.temperature == default.temperature
        assert neutral.speed == default.speed
        assert neutral.repetition_penalty == default.repetition_penalty

    def test_indian_english_preset_exists(self) -> None:
        preset = ACCENT_PRESETS["indian-english"]
        assert preset.temperature == 0.5
        assert preset.speed == 0.95
        assert preset.repetition_penalty == 12.0
        assert preset.top_k == 30
        assert preset.top_p == 0.75

    def test_indian_hindi_preset_exists(self) -> None:
        preset = ACCENT_PRESETS["indian-hindi"]
        assert preset.temperature == 0.55
        assert preset.speed == 0.90
        assert preset.repetition_penalty == 11.0
        assert preset.top_k == 35
        assert preset.top_p == 0.80

    def test_get_preset_valid(self) -> None:
        preset = get_preset("indian-english")
        assert isinstance(preset, VoiceSettings)
        assert preset.temperature == 0.5

    def test_get_preset_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent-accent")

    def test_all_presets_are_voice_settings(self) -> None:
        for name, preset in ACCENT_PRESETS.items():
            assert isinstance(preset, VoiceSettings), f"Preset '{name}' is not a VoiceSettings"
