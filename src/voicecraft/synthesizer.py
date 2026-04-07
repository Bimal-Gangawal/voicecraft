"""Synthesize speech using a cloned voice profile."""

from __future__ import annotations

import re
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


def _split_into_sentences(text: str, lang: str) -> list[str]:
    """Split text into sentence-level chunks for synthesis.

    XTTS v2 quality degrades on very long text, so we chunk into sentences.
    """
    if lang == "hi":
        # Hindi uses | (purna viram) and || as sentence terminators, plus standard punctuation
        parts = re.split(r"(?<=[।\|\.!\?])\s+", text.strip())
    else:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())

    # Merge very short fragments with the previous sentence
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and len(merged[-1]) + len(part) < MAX_CHUNK_CHARS:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    # If nothing was split (no punctuation), chunk by character limit
    if len(merged) == 1 and len(merged[0]) > MAX_CHUNK_CHARS:
        long_text = merged[0]
        merged = []
        while long_text:
            # Try to break at a space near the limit
            if len(long_text) <= MAX_CHUNK_CHARS:
                merged.append(long_text)
                break
            idx = long_text.rfind(" ", 0, MAX_CHUNK_CHARS)
            if idx == -1:
                idx = MAX_CHUNK_CHARS
            merged.append(long_text[:idx].strip())
            long_text = long_text[idx:].strip()

    return merged if merged else [text.strip()]


def synthesize(
    text: str,
    voice_name: str,
    lang: str = "en",
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Synthesize speech from text using a cloned voice.

    Args:
        text: The text to speak.
        voice_name: Name of a saved voice profile.
        lang: Language code ('en' or 'hi').
        output_path: Optional path for saving. If None, generates in OUTPUT_DIR.

    Returns:
        numpy array of the generated waveform.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{lang}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    # Load voice profile
    console.print(f"[cyan]Loading voice profile: [bold]{voice_name}[/bold][/cyan]")
    gpt_cond_latent, speaker_embedding, metadata = load_voice_profile(voice_name)

    # Load model
    tts = load_model()
    model = tts.synthesizer.tts_model

    # Move latents to device
    gpt_cond_latent = gpt_cond_latent.to(DEVICE)
    speaker_embedding = speaker_embedding.to(DEVICE)

    # Split text into chunks
    chunks = _split_into_sentences(text, lang)
    console.print(f"[cyan]Synthesizing {len(chunks)} chunk(s) in {SUPPORTED_LANGUAGES[lang]}...[/cyan]")

    all_audio: list[np.ndarray] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating speech...", total=len(chunks))

        for i, chunk in enumerate(chunks):
            progress.update(task, description=f"Chunk {i + 1}/{len(chunks)}: {chunk[:40]}...")

            out = model.inference(
                text=chunk,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            )

            # out["wav"] is a torch tensor or list
            wav = out["wav"]
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            elif isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)

            wav = wav.squeeze()
            all_audio.append(wav)

            progress.advance(task)

    # Concatenate all chunks with a small silence gap between them
    silence_samples = int(0.15 * 24000)  # XTTS v2 outputs at 24000 Hz
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts = []
    for i, audio in enumerate(all_audio):
        parts.append(audio)
        if i < len(all_audio) - 1:
            parts.append(silence)

    full_audio = np.concatenate(parts) if len(parts) > 1 else all_audio[0]

    console.print("[green]Synthesis complete.[/green]")
    return full_audio


def synthesize_oneshot(
    text: str,
    sample_path: str | Path,
    lang: str = "en",
) -> np.ndarray:
    """One-shot synthesis: extract voice from sample and synthesize in one step.

    Does NOT save a voice profile — use `clone` + `speak` for reuse.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{lang}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

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

    # Synthesize
    chunks = _split_into_sentences(text, lang)
    console.print(f"[cyan]Synthesizing {len(chunks)} chunk(s) in {SUPPORTED_LANGUAGES[lang]}...[/cyan]")

    all_audio: list[np.ndarray] = []
    for chunk in chunks:
        out = model.inference(
            text=chunk,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        wav = out["wav"]
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        elif isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        all_audio.append(wav.squeeze())

    silence_samples = int(0.15 * 24000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts = []
    for i, audio in enumerate(all_audio):
        parts.append(audio)
        if i < len(all_audio) - 1:
            parts.append(silence)

    full_audio = np.concatenate(parts) if len(parts) > 1 else all_audio[0]

    # Cleanup
    Path(processed_path).unlink(missing_ok=True)

    console.print("[green]Synthesis complete.[/green]")
    return full_audio
