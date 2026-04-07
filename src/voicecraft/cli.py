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


@app.command()
def speak(
    voice: str = typer.Option(..., "--voice", "-v", help="Name of a saved voice profile."),
    text: str = typer.Option(..., "--text", "-t", help="Text to synthesize."),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code: 'en' (English) or 'hi' (Hindi)."),
    fmt: str = typer.Option("wav", "--format", "-f", help="Output format: 'wav' or 'mp3'."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path. Auto-generated if omitted."),
    play: bool = typer.Option(False, "--play", "-p", help="Play the audio after generation (macOS only)."),
) -> None:
    """Generate speech in a cloned voice."""
    from voicecraft.export import save_audio
    from voicecraft.synthesizer import synthesize

    if lang not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Error: Unsupported language '{lang}'. Use: {list(SUPPORTED_LANGUAGES.keys())}[/red]")
        raise typer.Exit(1)

    if fmt not in ("wav", "mp3"):
        console.print("[red]Error: Format must be 'wav' or 'mp3'.[/red]")
        raise typer.Exit(1)

    # Auto-generate output path
    if output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = OUTPUT_DIR / f"{voice}_{lang}_{timestamp}.{fmt}"

    console.print(f"[bold]Voice:[/bold] {voice}")
    console.print(f"[bold]Language:[/bold] {SUPPORTED_LANGUAGES[lang]}")
    console.print(f"[bold]Text:[/bold] {text[:80]}{'...' if len(text) > 80 else ''}")
    console.print()

    start = time.time()
    waveform = synthesize(text, voice, lang)
    saved_path = save_audio(waveform, output, fmt)
    elapsed = time.time() - start

    console.print(f"[green]Generated in {elapsed:.1f}s: {saved_path}[/green]")

    if play:
        _play_audio(saved_path)


@app.command()
def say(
    sample: Path = typer.Option(..., "--sample", "-s", help="Path to voice sample audio file."),
    text: str = typer.Option(..., "--text", "-t", help="Text to synthesize."),
    lang: str = typer.Option("en", "--lang", "-l", help="Language code: 'en' or 'hi'."),
    fmt: str = typer.Option("wav", "--format", "-f", help="Output format: 'wav' or 'mp3'."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
    play: bool = typer.Option(False, "--play", "-p", help="Play the audio after generation."),
) -> None:
    """One-shot: clone a voice and synthesize speech in a single command (no saved profile)."""
    from voicecraft.export import save_audio
    from voicecraft.synthesizer import synthesize_oneshot

    if not sample.exists():
        console.print(f"[red]Error: File not found: {sample}[/red]")
        raise typer.Exit(1)

    if lang not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Error: Unsupported language '{lang}'. Use: {list(SUPPORTED_LANGUAGES.keys())}[/red]")
        raise typer.Exit(1)

    if output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = OUTPUT_DIR / f"oneshot_{lang}_{timestamp}.{fmt}"

    console.print(f"[bold]Sample:[/bold] {sample}")
    console.print(f"[bold]Language:[/bold] {SUPPORTED_LANGUAGES[lang]}")
    console.print(f"[bold]Text:[/bold] {text[:80]}{'...' if len(text) > 80 else ''}")
    console.print()

    start = time.time()
    waveform = synthesize_oneshot(text, sample, lang)
    saved_path = save_audio(waveform, output, fmt)
    elapsed = time.time() - start

    console.print(f"[green]Generated in {elapsed:.1f}s: {saved_path}[/green]")

    if play:
        _play_audio(saved_path)


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


def _play_audio(path: Path) -> None:
    """Play audio using macOS afplay."""
    console.print(f"[cyan]Playing: {path}[/cyan]")
    try:
        subprocess.run(["afplay", str(path)], check=True)
    except FileNotFoundError:
        console.print("[yellow]afplay not found — audio playback is only supported on macOS.[/yellow]")
    except subprocess.CalledProcessError:
        console.print("[yellow]Failed to play audio.[/yellow]")


if __name__ == "__main__":
    app()
