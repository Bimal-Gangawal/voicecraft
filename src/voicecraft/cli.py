"""Voicecraft CLI — clone a voice and synthesize speech."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

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
    """Download the XTTS v2 model (run this once before using other commands)."""
    from voicecraft.model_manager import download_model

    console.print("[bold]Voicecraft Setup[/bold]")
    console.print()
    start = time.time()
    download_model()
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
    sample: Optional[Path] = typer.Option(None, "--sample", "-s", help="Path to voice sample audio file (6-60s)."),
    name: str = typer.Option(..., "--name", "-n", help="Name for the voice profile."),
    record_mic: bool = typer.Option(False, "--record", "-r", help="Record from microphone instead of using a file."),
    duration: float = typer.Option(30.0, "--duration", "-d", help="Recording duration in seconds (used with --record)."),
    device: Optional[int] = typer.Option(None, "--device", help="Audio input device index (used with --record)."),
) -> None:
    """Extract a voice profile from an audio sample or a live microphone recording."""
    from voicecraft.extractor import extract_voice_profile

    if record_mic:
        from voicecraft.recorder import record_audio

        console.print(f"[bold]Recording voice sample for profile: {name}[/bold]")
        sample = record_audio(duration=duration, device=device)
    elif sample is None:
        console.print("[red]Error: Provide --sample <file> or use --record to capture from mic.[/red]")
        raise typer.Exit(1)
    elif not sample.exists():
        console.print(f"[red]Error: File not found: {sample}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Cloning voice from: {sample}[/bold]")
    console.print(f"[bold]Profile name: {name}[/bold]")
    console.print()

    start = time.time()
    extract_voice_profile(sample, name)
    elapsed = time.time() - start

    console.print()
    console.print(f"[green]Done in {elapsed:.1f}s. Use [bold]voicecraft speak --voice {name}[/bold] to generate speech.[/green]")


def _build_settings(
    speed: float, temperature: float, repetition_penalty: float
) -> "VoiceSettings":
    """Build a VoiceSettings from CLI flags."""
    from voicecraft.synthesizer import VoiceSettings

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

    settings = _build_settings(speed, temperature, repetition_penalty)

    console.print(f"[bold]Voice:[/bold] {voice}")
    console.print(f"[bold]Language:[/bold] {SUPPORTED_LANGUAGES[lang]}")
    console.print(f"[bold]Speed:[/bold] {speed}x  [bold]Temperature:[/bold] {temperature}")
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

    settings = _build_settings(speed, temperature, repetition_penalty)

    console.print(f"[bold]Sample:[/bold] {sample}")
    console.print(f"[bold]Language:[/bold] {SUPPORTED_LANGUAGES[lang]}")
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
    table.add_column("Duration", justify="right")
    table.add_column("Source File")
    table.add_column("Created")

    for p in profiles:
        table.add_row(
            p["name"],
            f"{p['sample_duration_s']:.1f}s",
            p.get("sample_file", "—"),
            p.get("created_at", "—"),
        )

    console.print(table)


if __name__ == "__main__":
    app()
