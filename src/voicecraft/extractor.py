"""Extract speaker conditioning latents from voice samples and save as a voice profile."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch
from rich.console import Console

from voicecraft.audio import get_duration, load_audio, preprocess, to_mono
from voicecraft.config import VOICES_DIR
from voicecraft.model_manager import load_model

console = Console()


def extract_voice_profile(
    sample_paths: str | Path | list[str | Path],
    voice_name: str,
) -> Path:
    """Extract speaker embeddings from one or more voice samples and save as a reusable profile.

    Using multiple samples significantly improves accent and voice characteristic capture.

    Args:
        sample_paths: Path(s) to voice sample audio file(s). Can be a single path,
                      a list of paths, or a path to a directory containing audio files.
        voice_name: Name for the voice profile (used for storage and retrieval).

    Returns:
        Path to the saved voice profile directory.
    """
    # Normalize to a list of paths
    if isinstance(sample_paths, (str, Path)):
        sample_paths = Path(sample_paths)
        if sample_paths.is_dir():
            # Collect all audio files from directory
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            sample_paths = sorted(
                p for p in sample_paths.iterdir()
                if p.suffix.lower() in audio_exts
            )
            if not sample_paths:
                raise FileNotFoundError(f"No audio files found in directory: {sample_paths}")
        else:
            sample_paths = [sample_paths]
    else:
        sample_paths = [Path(p) for p in sample_paths]

    console.print(f"[cyan]Processing {len(sample_paths)} audio sample(s)...[/cyan]")

    processed_paths: list[str] = []
    total_duration = 0.0
    all_warnings: list[str] = []
    source_files: list[str] = []

    for i, path in enumerate(sample_paths):
        path = Path(path)
        source_files.append(path.name)
        console.print(f"[cyan]  [{i+1}/{len(sample_paths)}] Preprocessing: {path.name}[/cyan]")

        processed_path, warnings = preprocess(path)
        all_warnings.extend(warnings)

        waveform, sr = load_audio(processed_path)
        waveform = to_mono(waveform)
        duration = get_duration(waveform, sr)
        total_duration += duration

        processed_paths.append(processed_path)

    for w in all_warnings:
        console.print(f"[yellow]  Warning: {w}[/yellow]")

    console.print(f"[cyan]Total sample duration: {total_duration:.1f}s across {len(processed_paths)} file(s)[/cyan]")

    # Load model and extract conditioning latents from ALL samples
    tts = load_model()
    model = tts.synthesizer.tts_model

    console.print("[cyan]Extracting speaker conditioning latents...[/cyan]")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=processed_paths
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
        "sample_file": ", ".join(source_files),
        "num_samples": len(processed_paths),
        "sample_duration_s": round(total_duration, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "xtts_v2",
    }
    (profile_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Copy first processed sample for reference
    shutil.copy2(processed_paths[0], profile_dir / "reference.wav")

    # Clean up processed temp files
    for p in processed_paths:
        Path(p).unlink(missing_ok=True)

    console.print(f"[green]Voice profile [bold]{voice_name}[/bold] saved to {profile_dir}[/green]")
    if len(processed_paths) > 1:
        console.print(f"[green]Used {len(processed_paths)} samples for better accent capture.[/green]")
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
