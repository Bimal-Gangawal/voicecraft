"""Synthesize speech using a cloned voice profile."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from voicecraft.config import DEVICE, OUTPUT_DIR, SUPPORTED_LANGUAGES
from voicecraft.extractor import load_voice_profile
from voicecraft.model_manager import load_model

console = Console()

# Maximum characters per chunk to avoid quality degradation on long text
MAX_CHUNK_CHARS = 250

# XTTS v2 output sample rate
XTTS_SAMPLE_RATE = 24000

# ── Pause durations (seconds) inserted at punctuation boundaries ────────────
PAUSE_SENTENCE_END = 0.35   # . ! ? | (purna viram)
PAUSE_COMMA = 0.18          # , ;
PAUSE_COLON = 0.25          # : —
PAUSE_ELLIPSIS = 0.45       # ...
PAUSE_PARAGRAPH = 0.55      # \n\n


@dataclass
class VoiceSettings:
    """Tunable parameters for XTTS v2 inference that affect accent and prosody."""
    temperature: float = 0.75       # Lower = more stable/monotone, higher = more expressive
    repetition_penalty: float = 10.0  # Penalises repeated tokens; higher = less repetition
    top_k: int = 50                 # Limits sampling pool; lower = more focused
    top_p: float = 0.85            # Nucleus sampling threshold
    speed: float = 1.0             # Speech rate multiplier (0.5 = half speed, 2.0 = double)
    length_penalty: float = 1.0    # Beam search length penalty


# ── Accent presets ───────────────────────────────────────────────────────────
ACCENT_PRESETS: dict[str, VoiceSettings] = {
    "neutral": VoiceSettings(),
    "indian-english": VoiceSettings(
        temperature=0.5,
        repetition_penalty=12.0,
        top_k=30,
        top_p=0.75,
        speed=0.95,
        length_penalty=1.0,
    ),
    "indian-hindi": VoiceSettings(
        temperature=0.55,
        repetition_penalty=11.0,
        top_k=35,
        top_p=0.80,
        speed=0.90,
        length_penalty=1.0,
    ),
}


def get_preset(name: str) -> VoiceSettings:
    """Return a VoiceSettings preset by name. Raises ValueError if unknown."""
    if name not in ACCENT_PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(ACCENT_PRESETS.keys())}"
        )
    return ACCENT_PRESETS[name]


# ── Text preprocessing ──────────────────────────────────────────────────────

def _normalize_text(text: str, lang: str) -> str:
    """Clean and normalize text for better synthesis quality."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Expand common abbreviations / symbols
    text = text.replace("&", " and " if lang == "en" else " aur ")
    text = text.replace("@", " at ")
    text = text.replace("%", " percent " if lang == "en" else " pratishat ")

    # Normalize quotes and dashes
    text = re.sub(r'[\u201c\u201d\u201e]', '"', text)
    text = re.sub(r'[\u2018\u2019\u201a]', "'", text)
    text = re.sub(r'[\u2013\u2014]', ' - ', text)

    # Ensure sentence-ending punctuation for better prosody
    # If text doesn't end with punctuation, add a period
    if lang == "hi":
        if not re.search(r"[।\|\.!\?]$", text.strip()):
            text = text.strip() + "।"
    else:
        if not re.search(r"[.!?]$", text.strip()):
            text = text.strip() + "."

    return text


def _get_trailing_pause(chunk: str, lang: str) -> float:
    """Determine the pause duration to insert after a chunk based on its trailing punctuation."""
    chunk = chunk.rstrip()

    if re.search(r"\.\.\.", chunk):
        return PAUSE_ELLIPSIS
    if lang == "hi" and re.search(r"[।\|]$", chunk):
        return PAUSE_SENTENCE_END
    if re.search(r"[.!?]$", chunk):
        return PAUSE_SENTENCE_END
    if re.search(r"[;,]$", chunk):
        return PAUSE_COMMA
    if re.search(r"[:—]$", chunk):
        return PAUSE_COLON
    return PAUSE_COMMA  # Default small pause between chunks


def _split_into_chunks(text: str, lang: str) -> list[str]:
    """Split text into synthesis-friendly chunks with punctuation-aware boundaries.

    XTTS v2 quality degrades on very long text, so we chunk at natural boundaries.
    """
    # First split on paragraph breaks
    paragraphs = re.split(r"\n\s*\n", text.strip())

    all_chunks: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split on sentence-ending punctuation
        if lang == "hi":
            parts = re.split(r"(?<=[।\|\.!\?])\s+", para)
        else:
            parts = re.split(r"(?<=[.!?])\s+", para)

        # Further split long sentences at commas / semicolons / colons
        refined: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > MAX_CHUNK_CHARS:
                # Split at comma/semicolon/colon boundaries
                sub_parts = re.split(r"(?<=[,;:])\s+", part)
                for sp in sub_parts:
                    sp = sp.strip()
                    if not sp:
                        continue
                    if refined and len(refined[-1]) + len(sp) + 1 < MAX_CHUNK_CHARS:
                        refined[-1] = refined[-1] + " " + sp
                    else:
                        refined.append(sp)
            else:
                refined.append(part)

        # Merge very short fragments
        merged: list[str] = []
        for chunk in refined:
            if merged and len(merged[-1]) + len(chunk) + 1 < MAX_CHUNK_CHARS:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        # If still too long, hard-break at word boundaries
        final: list[str] = []
        for chunk in merged:
            if len(chunk) <= MAX_CHUNK_CHARS:
                final.append(chunk)
            else:
                while chunk:
                    if len(chunk) <= MAX_CHUNK_CHARS:
                        final.append(chunk)
                        break
                    idx = chunk.rfind(" ", 0, MAX_CHUNK_CHARS)
                    if idx == -1:
                        idx = MAX_CHUNK_CHARS
                    final.append(chunk[:idx].strip())
                    chunk = chunk[idx:].strip()

        all_chunks.extend(final)

        # Mark paragraph boundary (we'll insert a longer pause here)
        if all_chunks:
            all_chunks[-1] = all_chunks[-1] + "\n\n"

    # Clean trailing marker from last chunk
    if all_chunks:
        all_chunks[-1] = all_chunks[-1].rstrip()

    return all_chunks if all_chunks else [text.strip()]


# Keep old name as alias for tests
_split_into_sentences = _split_into_chunks


def _build_waveform(
    chunks: list[str],
    model: object,
    lang: str,
    gpt_cond_latent: torch.Tensor,
    speaker_embedding: torch.Tensor,
    settings: VoiceSettings,
) -> np.ndarray:
    """Run inference on chunks and stitch with punctuation-aware pauses."""
    all_audio: list[np.ndarray] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating speech...", total=len(chunks))

        for i, chunk in enumerate(chunks):
            # Check for paragraph boundary marker
            is_paragraph_end = chunk.endswith("\n\n")
            clean_chunk = chunk.strip()

            progress.update(task, description=f"Chunk {i + 1}/{len(chunks)}: {clean_chunk[:40]}...")

            out = model.inference(
                text=clean_chunk,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=settings.temperature,
                length_penalty=settings.length_penalty,
                repetition_penalty=settings.repetition_penalty,
                top_k=settings.top_k,
                top_p=settings.top_p,
                speed=settings.speed,
            )

            wav = out["wav"]
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            elif isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)

            wav = wav.squeeze()
            all_audio.append(wav)

            # Insert appropriate pause after this chunk
            if i < len(chunks) - 1:
                if is_paragraph_end:
                    pause_s = PAUSE_PARAGRAPH
                else:
                    pause_s = _get_trailing_pause(clean_chunk, lang)
                silence = np.zeros(int(pause_s * XTTS_SAMPLE_RATE), dtype=np.float32)
                all_audio.append(silence)

            progress.advance(task)

    return np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]


def synthesize(
    text: str,
    voice_name: str,
    lang: str = "en",
    settings: VoiceSettings | None = None,
) -> np.ndarray:
    """Synthesize speech from text using a cloned voice.

    Args:
        text: The text to speak.
        voice_name: Name of a saved voice profile.
        lang: Language code ('en' or 'hi').
        settings: Optional VoiceSettings to tune accent/prosody.

    Returns:
        numpy array of the generated waveform (24 kHz).
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{lang}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    settings = settings or VoiceSettings()

    # Load voice profile
    console.print(f"[cyan]Loading voice profile: [bold]{voice_name}[/bold][/cyan]")
    gpt_cond_latent, speaker_embedding, metadata = load_voice_profile(voice_name)

    # Load model
    tts = load_model()
    model = tts.synthesizer.tts_model

    # Move latents to device
    gpt_cond_latent = gpt_cond_latent.to(DEVICE)
    speaker_embedding = speaker_embedding.to(DEVICE)

    # Preprocess text
    text = _normalize_text(text, lang)

    # Split text into chunks
    chunks = _split_into_chunks(text, lang)
    console.print(f"[cyan]Synthesizing {len(chunks)} chunk(s) in {SUPPORTED_LANGUAGES[lang]}...[/cyan]")

    full_audio = _build_waveform(chunks, model, lang, gpt_cond_latent, speaker_embedding, settings)

    console.print("[green]Synthesis complete.[/green]")
    return full_audio


def synthesize_oneshot(
    text: str,
    sample_path: str | Path,
    lang: str = "en",
    settings: VoiceSettings | None = None,
) -> np.ndarray:
    """One-shot synthesis: extract voice from sample and synthesize in one step.

    Does NOT save a voice profile — use `clone` + `speak` for reuse.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{lang}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    settings = settings or VoiceSettings()
    sample_path = Path(sample_path)

    # Preprocess audio
    from voicecraft.audio import preprocess

    console.print(f"[cyan]Preprocessing: {sample_path.name}[/cyan]")
    processed_path, warnings = preprocess(sample_path)
    for w in warnings:
        console.print(f"[yellow]  Warning: {w}[/yellow]")

    # Load model
    tts = load_model()
    model = tts.synthesizer.tts_model

    # Extract latents on the fly
    console.print("[cyan]Extracting voice characteristics...[/cyan]")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[processed_path]
    )

    gpt_cond_latent = gpt_cond_latent.to(DEVICE)
    speaker_embedding = speaker_embedding.to(DEVICE)

    # Preprocess text
    text = _normalize_text(text, lang)

    # Synthesize
    chunks = _split_into_chunks(text, lang)
    console.print(f"[cyan]Synthesizing {len(chunks)} chunk(s) in {SUPPORTED_LANGUAGES[lang]}...[/cyan]")

    full_audio = _build_waveform(chunks, model, lang, gpt_cond_latent, speaker_embedding, settings)

    # Cleanup
    Path(processed_path).unlink(missing_ok=True)

    console.print("[green]Synthesis complete.[/green]")
    return full_audio
