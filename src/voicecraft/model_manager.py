"""Download, cache, and load the XTTS v2 model."""

from __future__ import annotations

import typing

from rich.console import Console

from voicecraft.config import DEVICE, MODEL_CACHE_DIR

console = Console()

# Singleton — avoid loading the model multiple times in one process
_tts_instance: typing.Any | None = None


def get_model_path() -> str:
    """Return the expected local model directory path."""
    return str(MODEL_CACHE_DIR / "xtts_v2")


def is_model_downloaded() -> bool:
    """Check whether the XTTS v2 model files already exist in cache."""
    model_dir = MODEL_CACHE_DIR / "xtts_v2"
    if not model_dir.exists():
        return False
    # XTTS v2 requires at least config.json and a checkpoint file
    has_config = any(model_dir.glob("config.json"))
    has_checkpoint = any(model_dir.glob("*.pth")) or any(model_dir.glob("model.*"))
    return has_config and has_checkpoint


def load_model() -> typing.Any:
    """Load (and download if necessary) the XTTS v2 model. Returns a TTS instance."""
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance

    # Import here to avoid slow import at CLI startup
    from TTS.api import TTS

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    console.print(f"[cyan]Loading XTTS v2 model on device=[bold]{DEVICE}[/bold]...[/cyan]")

    tts = TTS(model_name).to(DEVICE)

    console.print("[green]Model loaded successfully.[/green]")
    _tts_instance = tts
    return tts


def download_model() -> None:
    """Explicitly download the XTTS v2 model (used by `setup` command)."""
    from TTS.api import TTS

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    console.print("[cyan]Downloading XTTS v2 model (~1.8 GB)... this only happens once.[/cyan]")

    # Instantiating TTS with the model name triggers the download
    TTS(model_name)

    console.print("[green]Model downloaded and cached successfully.[/green]")
