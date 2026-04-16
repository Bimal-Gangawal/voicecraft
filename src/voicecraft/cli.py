"""Voicecraft CLI — clone a voice and synthesize speech."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from voicecraft.config import OUTPUT_DIR, SUPPORTED_LANGUAGES

app = typer.Typer(
    name="voicecraft",
    help="Clone a voice from a 30s sample and synthesize speech in English & Hindi.",
    add_completion=False,
)
console = Console()


@app.command()
def setup() -> None:
    """Download the XTTS v2 model and translation models (run this once before using other commands)."""
    from voicecraft.model_manager import download_model
    from voicecraft.translator import ensure_translation_models

    console.print("[bold]Voicecraft Setup[/bold]")
    console.print()
    start = time.time()
    download_model()

    # Download translation models for live translation
    console.print()
    console.print("[cyan]Setting up translation models...[/cyan]")
    ensure_translation_models("en", "hi")
    ensure_translation_models("hi", "en")

    elapsed = time.time() - start
    console.print(f"[green]Setup complete in {elapsed:.0f}s. You're ready to clone voices![/green]")


@app.command()
def record(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output WAV file path."),
    duration: float = typer.Option(30.0, "--duration", "-d", help="Recording duration in seconds."),
    device: Optional[int] = typer.Option(None, "--device", help="Audio input device index."),
    list_devices_flag: bool = typer.Option(False, "--list-devices", help="List available audio input devices and exit."),
) -> None:
    """Record audio from your microphone and save as a WAV file."""
    from voicecraft.recorder import list_devices as _list_devices
    from voicecraft.recorder import record_audio

    if list_devices_flag:
        devices = _list_devices()
        if not devices:
            console.print("[yellow]No audio input devices found.[/yellow]")
            raise typer.Exit(1)

        table = Table(title="Audio Input Devices")
        table.add_column("Index", justify="right", style="bold")
        table.add_column("Name")
        table.add_column("Channels", justify="right")

        for d in devices:
            table.add_row(str(d["index"]), d["name"], str(d["channels"]))

        console.print(table)
        return

    if duration < 6:
        console.print("[red]Error: Duration must be at least 6 seconds for voice cloning.[/red]")
        raise typer.Exit(1)

    record_audio(duration=duration, output_path=output, device=device)


@app.command()
def clone(
    sample: Optional[List[Path]] = typer.Option(None, "--sample", "-s", help="Path to voice sample audio file(s). Repeat for multiple samples."),
    samples_dir: Optional[Path] = typer.Option(None, "--samples-dir", "-S", help="Directory containing voice sample audio files."),
    name: str = typer.Option(..., "--name", "-n", help="Name for the voice profile."),
    record_mic: bool = typer.Option(False, "--record", "-r", help="Record from microphone instead of using a file."),
    duration: float = typer.Option(30.0, "--duration", "-d", help="Recording duration in seconds (used with --record)."),
    device: Optional[int] = typer.Option(None, "--device", help="Audio input device index (used with --record)."),
) -> None:
    """Extract a voice profile from audio sample(s), a directory of samples, or a live mic recording.

    Use multiple samples for better accent and voice characteristic capture:
      voicecraft clone --sample a.wav --sample b.wav --name myvoice
      voicecraft clone --samples-dir ./my_recordings/ --name myvoice
    """
    from voicecraft.extractor import extract_voice_profile

    if record_mic:
        from voicecraft.recorder import record_audio

        console.print(f"[bold]Recording voice sample for profile: {name}[/bold]")
        recorded = record_audio(duration=duration, device=device)
        sample_input: str | Path | list[Path] = recorded
    elif samples_dir is not None:
        if not samples_dir.exists() or not samples_dir.is_dir():
            console.print(f"[red]Error: Directory not found: {samples_dir}[/red]")
            raise typer.Exit(1)
        sample_input = samples_dir
    elif sample:
        for s in sample:
            if not s.exists():
                console.print(f"[red]Error: File not found: {s}[/red]")
                raise typer.Exit(1)
        sample_input = sample if len(sample) > 1 else sample[0]
    else:
        console.print("[red]Error: Provide --sample <file>, --samples-dir <dir>, or use --record.[/red]")
        raise typer.Exit(1)

    if isinstance(sample_input, list):
        console.print(f"[bold]Cloning voice from {len(sample_input)} sample(s)[/bold]")
    elif isinstance(sample_input, Path) and sample_input.is_dir():
        console.print(f"[bold]Cloning voice from directory: {sample_input}[/bold]")
    else:
        console.print(f"[bold]Cloning voice from: {sample_input}[/bold]")
    console.print(f"[bold]Profile name: {name}[/bold]")
    console.print()

    start = time.time()
    extract_voice_profile(sample_input, name)
    elapsed = time.time() - start

    console.print()
    console.print(f"[green]Done in {elapsed:.1f}s. Use [bold]voicecraft speak --voice {name}[/bold] to generate speech.[/green]")


def _build_settings(
    speed: float,
    temperature: float,
    repetition_penalty: float,
    preset: str | None = None,
) -> "VoiceSettings":
    """Build a VoiceSettings from CLI flags, optionally starting from a preset."""
    from voicecraft.synthesizer import VoiceSettings, get_preset

    if preset:
        base = get_preset(preset)
        # Override only if the user explicitly passed a non-default value
        return VoiceSettings(
            speed=speed if speed != 1.0 else base.speed,
            temperature=temperature if temperature != 0.75 else base.temperature,
            repetition_penalty=repetition_penalty if repetition_penalty != 10.0 else base.repetition_penalty,
            top_k=base.top_k,
            top_p=base.top_p,
            length_penalty=base.length_penalty,
        )

    return VoiceSettings(
        speed=speed,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )


@app.command()
def speak(
    voice: str = typer.Option(..., "--voice", "-v", help="Name of a saved voice profile."),
    text: str = typer.Option(..., "--text", "-t", help="Text to synthesize."),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code: 'en' (English) or 'hi' (Hindi)."),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save audio to file (WAV/MP3). If omitted, plays directly."),
    fmt: str = typer.Option("wav", "--format", "-f", help="Output format when saving: 'wav' or 'mp3'."),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Accent preset: 'indian-english', 'indian-hindi', or 'neutral'."),
    speed: float = typer.Option(1.0, "--speed", help="Speech speed multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)."),
    temperature: float = typer.Option(0.75, "--temperature", help="Expressiveness (0.1 = flat, 0.75 = natural, 1.0+ = very expressive)."),
    repetition_penalty: float = typer.Option(10.0, "--repetition-penalty", help="Repetition penalty (higher = less repetition)."),
) -> None:
    """Generate speech in a cloned voice. Plays directly by default; use --save to write to file."""
    from voicecraft.export import play_audio, save_audio
    from voicecraft.synthesizer import synthesize

    if lang not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Error: Unsupported language '{lang}'. Use: {list(SUPPORTED_LANGUAGES.keys())}[/red]")
        raise typer.Exit(1)

    if save and fmt not in ("wav", "mp3"):
        console.print("[red]Error: Format must be 'wav' or 'mp3'.[/red]")
        raise typer.Exit(1)

    settings = _build_settings(speed, temperature, repetition_penalty, preset)

    console.print(f"[bold]Voice:[/bold] {voice}")
    console.print(f"[bold]Language:[/bold] {SUPPORTED_LANGUAGES[lang]}")
    if preset:
        console.print(f"[bold]Preset:[/bold] {preset}")
    console.print(f"[bold]Speed:[/bold] {settings.speed}x  [bold]Temperature:[/bold] {settings.temperature}")
    console.print(f"[bold]Text:[/bold] {text[:80]}{'...' if len(text) > 80 else ''}")
    console.print()

    start = time.time()
    waveform = synthesize(text, voice, lang, settings)
    elapsed = time.time() - start

    console.print(f"[green]Generated in {elapsed:.1f}s[/green]")

    if save:
        save_audio(waveform, save, fmt)
    else:
        play_audio(waveform)


@app.command()
def say(
    sample: Path = typer.Option(..., "--sample", "-s", help="Path to voice sample audio file."),
    text: str = typer.Option(..., "--text", "-t", help="Text to synthesize."),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code: 'en' or 'hi'."),
    save: Optional[Path] = typer.Option(None, "--save", help="Save audio to file. If omitted, plays directly."),
    fmt: str = typer.Option("wav", "--format", "-f", help="Output format when saving: 'wav' or 'mp3'."),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Accent preset: 'indian-english', 'indian-hindi', or 'neutral'."),
    speed: float = typer.Option(1.0, "--speed", help="Speech speed multiplier."),
    temperature: float = typer.Option(0.75, "--temperature", help="Expressiveness."),
    repetition_penalty: float = typer.Option(10.0, "--repetition-penalty", help="Repetition penalty."),
) -> None:
    """One-shot: clone a voice and synthesize speech in a single command (no saved profile)."""
    from voicecraft.export import play_audio, save_audio
    from voicecraft.synthesizer import synthesize_oneshot

    if not sample.exists():
        console.print(f"[red]Error: File not found: {sample}[/red]")
        raise typer.Exit(1)

    if lang not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Error: Unsupported language '{lang}'. Use: {list(SUPPORTED_LANGUAGES.keys())}[/red]")
        raise typer.Exit(1)

    settings = _build_settings(speed, temperature, repetition_penalty, preset)

    console.print(f"[bold]Sample:[/bold] {sample}")
    console.print(f"[bold]Language:[/bold] {SUPPORTED_LANGUAGES[lang]}")
    if preset:
        console.print(f"[bold]Preset:[/bold] {preset}")
    console.print(f"[bold]Text:[/bold] {text[:80]}{'...' if len(text) > 80 else ''}")
    console.print()

    start = time.time()
    waveform = synthesize_oneshot(text, sample, lang, settings)
    elapsed = time.time() - start

    console.print(f"[green]Generated in {elapsed:.1f}s[/green]")

    if save:
        save_audio(waveform, save, fmt)
    else:
        play_audio(waveform)


@app.command()
def voices() -> None:
    """List all saved voice profiles."""
    from voicecraft.extractor import list_voice_profiles

    profiles = list_voice_profiles()

    if not profiles:
        console.print("[yellow]No voice profiles found. Use [bold]voicecraft clone[/bold] to create one.[/yellow]")
        return

    table = Table(title="Saved Voice Profiles")
    table.add_column("Name", style="bold cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Source File(s)")
    table.add_column("Optimized", justify="center")
    table.add_column("Created")

    for p in profiles:
        table.add_row(
            p["name"],
            str(p.get("num_samples", 1)),
            f"{p['sample_duration_s']:.1f}s",
            p.get("sample_file", "—"),
            "Yes" if p.get("optimized") else "—",
            p.get("created_at", "—"),
        )

    console.print(table)


@app.command()
def presets() -> None:
    """List available accent presets and their settings."""
    from voicecraft.synthesizer import ACCENT_PRESETS

    table = Table(title="Accent Presets")
    table.add_column("Preset", style="bold cyan")
    table.add_column("Temperature", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Rep. Penalty", justify="right")
    table.add_column("Top-K", justify="right")
    table.add_column("Top-P", justify="right")

    for name, settings in ACCENT_PRESETS.items():
        table.add_row(
            name,
            f"{settings.temperature:.2f}",
            f"{settings.speed:.2f}",
            f"{settings.repetition_penalty:.1f}",
            str(settings.top_k),
            f"{settings.top_p:.2f}",
        )

    console.print(table)
    console.print()
    console.print("[dim]Use with: voicecraft speak --preset indian-english --voice <name> --text '...'[/dim]")


@app.command()
def optimize(
    voice: str = typer.Option(..., "--voice", "-v", help="Name of the voice profile to optimize."),
    audio_dir: Path = typer.Option(..., "--audio-dir", "-a", help="Directory with reference audio files (3-5 recommended)."),
    steps: int = typer.Option(150, "--steps", help="Optimization steps (more = better but slower)."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate for optimization."),
    restore: bool = typer.Option(False, "--restore", help="Restore original (pre-optimization) latents."),
) -> None:
    """Optimize voice profile latents for better accent fidelity.

    Refines conditioning latents by comparing against reference audio.
    Much faster than full model fine-tuning (~5-15 min on CPU).

    Provide 3-5 audio samples of the target voice for best results:
      voicecraft optimize --voice myvoice --audio-dir ./samples/
    """
    if restore:
        from voicecraft.optimizer import restore_original_latents

        restore_original_latents(voice)
        return

    if not audio_dir.exists() or not audio_dir.is_dir():
        console.print(f"[red]Error: Directory not found: {audio_dir}[/red]")
        raise typer.Exit(1)

    from voicecraft.optimizer import OptimizationConfig, optimize_latents

    config = OptimizationConfig(steps=steps, learning_rate=lr)

    console.print("[bold]Voice Latent Optimization[/bold]")
    console.print()

    start = time.time()
    optimize_latents(voice, audio_dir, config)
    elapsed = time.time() - start

    console.print()
    console.print(f"[green]Optimization finished in {elapsed:.1f}s.[/green]")
    console.print(f"[dim]To undo: voicecraft optimize --voice {voice} --restore[/dim]")


@app.command()
def translate(
    voice: str = typer.Option(..., "--voice", "-v", help="Name of a saved voice profile for TTS output."),
    from_lang: str = typer.Option("en", "--from", "-f", help="Source language to listen for: 'en' or 'hi'."),
    to_lang: str = typer.Option("hi", "--to", "-t", help="Target language to translate into: 'en' or 'hi'."),
    whisper_model: str = typer.Option("small", "--whisper-model", help="Whisper model: tiny, base, small, medium, large-v3."),
    device: Optional[int] = typer.Option(None, "--device", help="Audio input device index."),
    speed: float = typer.Option(1.0, "--speed", help="Speech speed for TTS output."),
    temperature: float = typer.Option(0.75, "--temperature", help="TTS expressiveness."),
) -> None:
    """Live translation: speak in one language, hear it back in another using your cloned voice.

    Listens to your microphone, transcribes speech, translates it,
    and speaks the translation using your cloned voice — all fully local.

    Examples:
      voicecraft translate --voice my_voice --from en --to hi
      voicecraft translate --voice my_voice --from hi --to en
      voicecraft translate --voice my_voice --from en --to hi --whisper-model medium
    """
    from voicecraft.translator import LiveTranslator, TranslatorConfig, WHISPER_MODELS

    if from_lang not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Error: Unsupported source language '{from_lang}'. Use: {list(SUPPORTED_LANGUAGES.keys())}[/red]")
        raise typer.Exit(1)

    if to_lang not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Error: Unsupported target language '{to_lang}'. Use: {list(SUPPORTED_LANGUAGES.keys())}[/red]")
        raise typer.Exit(1)

    if from_lang == to_lang:
        console.print("[red]Error: Source and target languages must be different.[/red]")
        raise typer.Exit(1)

    if whisper_model not in WHISPER_MODELS:
        console.print(f"[red]Error: Unknown whisper model '{whisper_model}'. Use: {list(WHISPER_MODELS)}[/red]")
        raise typer.Exit(1)

    config = TranslatorConfig(
        voice_name=voice,
        from_lang=from_lang,
        to_lang=to_lang,
        whisper_model=whisper_model,
        mic_device=device,
        speed=speed,
        temperature=temperature,
    )

    translator = LiveTranslator(config)
    translator.run()


if __name__ == "__main__":
    app()
