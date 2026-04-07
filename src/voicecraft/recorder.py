"""Record audio from the microphone."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.live import Live
from rich.text import Text

from voicecraft.config import SAMPLE_RATE

console = Console()

# Recording sample rate — 22050 is the XTTS target, but 44100 captures
# higher-quality audio that we downsample later during preprocessing.
RECORD_SAMPLE_RATE = 44100


def _build_level_bar(rms: float, width: int = 40) -> Text:
    """Build a colored audio-level bar from an RMS value."""
    # Normalize RMS to 0-1 range (typical speech RMS is 0.01–0.3)
    level = min(rms / 0.3, 1.0)
    filled = int(level * width)

    bar = Text()
    bar.append("[")
    if filled > 0:
        green_end = int(width * 0.6)
        yellow_end = int(width * 0.85)
        for i in range(filled):
            if i < green_end:
                bar.append("\u2588", style="green")
            elif i < yellow_end:
                bar.append("\u2588", style="yellow")
            else:
                bar.append("\u2588", style="red")
    bar.append(" " * (width - filled))
    bar.append("]")
    return bar


def list_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
    return inputs


def record_audio(
    duration: float = 30.0,
    output_path: str | Path | None = None,
    device: int | None = None,
) -> Path:
    """Record audio from the microphone with a live level meter.

    Args:
        duration: Recording duration in seconds.
        output_path: Where to save the WAV file. Auto-generated if None.
        device: Audio input device index. None = system default.

    Returns:
        Path to the saved WAV file.
    """
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"recording_{timestamp}.wav")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = int(duration * RECORD_SAMPLE_RATE)
    recording = np.zeros((total_samples, 1), dtype=np.float32)
    samples_recorded = 0

    console.print()
    console.print(f"[bold cyan]Recording for {duration:.0f} seconds...[/bold cyan]")
    console.print("[dim]Speak clearly into your microphone. Press Ctrl+C to stop early.[/dim]")
    console.print()

    # Callback writes audio into the buffer
    def callback(indata, frames, time_info, status):
        nonlocal samples_recorded
        if status:
            pass  # Ignore xrun warnings in the callback
        end = min(samples_recorded + frames, total_samples)
        count = end - samples_recorded
        recording[samples_recorded:end] = indata[:count]
        samples_recorded = end

    try:
        stream = sd.InputStream(
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        )

        with stream, Live(console=console, refresh_per_second=10) as live:
            start_time = time.time()

            while samples_recorded < total_samples:
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)

                # Compute RMS of the last 0.1s for the level meter
                meter_samples = int(0.1 * RECORD_SAMPLE_RATE)
                start_idx = max(0, samples_recorded - meter_samples)
                chunk = recording[start_idx:samples_recorded, 0]
                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0

                level_bar = _build_level_bar(rms)
                status_text = Text()
                status_text.append(f"  Time remaining: {remaining:.1f}s  ", style="bold")
                status_text.append(level_bar)

                live.update(status_text)
                time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped early.[/yellow]")

    # Trim to actual recorded length
    recording = recording[:samples_recorded]
    actual_duration = samples_recorded / RECORD_SAMPLE_RATE

    if actual_duration < 1.0:
        console.print("[red]Recording too short (< 1s). Discarded.[/red]")
        raise RuntimeError("Recording too short.")

    # Save
    sf.write(str(output_path), recording, RECORD_SAMPLE_RATE)

    console.print()
    console.print(f"[green]Recorded {actual_duration:.1f}s of audio -> {output_path}[/green]")
    return output_path
