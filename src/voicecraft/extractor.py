"""Extract speaker conditioning latents from a voice sample and save as a voice profile."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from rich.console import Console

from voicecraft.audio import get_duration, load_audio, preprocess, to_mono
from voicecraft.config import VOICES_DIR
from voicecraft.model_manager import load_model

console = Console()


def extract_voice_profile(
    sample_path: str | Path,
    voice_name: str,
) -> Path:
    """Extract speaker embeddings from a voice sample and save as a reusable profile.

    Args:
        sample_path: Path to the voice sample audio file.
        voice_name: Name for the voice profile (used for storage and retrieval).

    Returns:
        Path to the saved voice profile directory.
    """
    sample_path = Path(sample_path)

    # Preprocess the audio
    console.print(f"[cyan]Preprocessing audio sample: {sample_path.name}[/cyan]")
    processed_path, warnings = preprocess(sample_path)
    for w in warnings:
        console.print(f"[yellow]  Warning: {w}[/yellow]")

    # Get sample duration for metadata
    waveform, sr = load_audio(processed_path)
    waveform = to_mono(waveform)
    duration = get_duration(waveform, sr)
    console.print(f"[cyan]Sample duration: {duration:.1f}s[/cyan]")

    # Load model and extract conditioning latents
    tts = load_model()
    model = tts.synthesizer.tts_model

    console.print("[cyan]Extracting speaker conditioning latents...[/cyan]")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[processed_path]
    )

    # Save voice profile
    profile_dir = VOICES_DIR / voice_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors
    torch.save(
        {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        },
        profile_dir / "latents.pt",
    )

    # Save metadata
    metadata = {
        "name": voice_name,
        "sample_file": sample_path.name,
        "sample_duration_s": round(duration, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "xtts_v2",
    }
    (profile_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Copy the processed sample for reference
    import shutil
    shutil.copy2(processed_path, profile_dir / "reference.wav")

    # Clean up processed temp file
    Path(processed_path).unlink(missing_ok=True)

    console.print(f"[green]Voice profile [bold]{voice_name}[/bold] saved to {profile_dir}[/green]")
    return profile_dir


def load_voice_profile(voice_name: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load a saved voice profile.

    Returns:
        (gpt_cond_latent, speaker_embedding, metadata)
    """
    profile_dir = VOICES_DIR / voice_name

    if not profile_dir.exists():
        raise FileNotFoundError(
            f"Voice profile '{voice_name}' not found. "
            f"Available voices: {list_voice_profiles()}"
        )

    latents = torch.load(profile_dir / "latents.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((profile_dir / "metadata.json").read_text())

    return latents["gpt_cond_latent"], latents["speaker_embedding"], metadata


def list_voice_profiles() -> list[dict]:
    """List all saved voice profiles with their metadata."""
    profiles = []
    if not VOICES_DIR.exists():
        return profiles

    for profile_dir in sorted(VOICES_DIR.iterdir()):
        meta_file = profile_dir / "metadata.json"
        if meta_file.exists():
            metadata = json.loads(meta_file.read_text())
            profiles.append(metadata)

    return profiles
