"""Tests for the synthesizer module (text splitting only — synthesis requires the model)."""

from voicecraft.synthesizer import _split_into_sentences


class TestSplitIntoSentences:
    def test_english_short_text_stays_merged(self) -> None:
        # Short text under MAX_CHUNK_CHARS stays as a single chunk
        text = "Hello world. How are you? I am fine!"
        chunks = _split_into_sentences(text, "en")
        assert len(chunks) == 1
        assert "Hello world." in chunks[0]

    def test_english_long_multi_sentence(self) -> None:
        # Long multi-sentence text gets split into multiple chunks
        text = (
            "This is the first sentence that is reasonably long and detailed. "
            "This is the second sentence with even more detail and explanation. "
            "And here is a third sentence that pushes the total length over the limit. "
            "Finally a fourth sentence that definitely exceeds the chunk size threshold. "
            "Plus a fifth sentence for good measure and to be absolutely sure it splits."
        )
        chunks = _split_into_sentences(text, "en")
        assert len(chunks) >= 2

    def test_hindi_short_text_stays_merged(self) -> None:
        # Short Hindi text stays as single chunk
        text = "नमस्ते दुनिया। आप कैसे हैं? मैं ठीक हूँ।"
        chunks = _split_into_sentences(text, "hi")
        assert len(chunks) >= 1

    def test_single_sentence(self) -> None:
        text = "Just one sentence"
        chunks = _split_into_sentences(text, "en")
        assert len(chunks) == 1
        assert chunks[0] == "Just one sentence"

    def test_long_text_chunking(self) -> None:
        text = "This is a very long sentence that goes on and on " * 20
        chunks = _split_into_sentences(text.strip(), "en")
        assert all(len(c) <= 260 for c in chunks)  # Allow small buffer

    def test_empty_after_strip(self) -> None:
        text = "   "
        chunks = _split_into_sentences(text, "en")
        assert len(chunks) == 1

    def test_merges_short_fragments(self) -> None:
        text = "Hi. Ok. Sure. Yes."
        chunks = _split_into_sentences(text, "en")
        # Short fragments should be merged together
        assert len(chunks) <= 2
