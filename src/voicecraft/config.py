"""Configuration, paths, and device detection for Voicecraft."""

from pathlib import Path

import torch

# ── Directories ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOICES_DIR = PROJECT_ROOT / "voices"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_CACHE_DIR = Path.home() / ".local" / "share" / "voicecraft" / "models"

for _d in (VOICES_DIR, OUTPUT_DIR, MODEL_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Audio constants ──────────────────────────────────────────────────────────
SAMPLE_RATE = 22050          # XTTS v2 expected sample rate
MIN_SAMPLE_DURATION = 6.0    # seconds — hard minimum
WARN_SAMPLE_DURATION = 10.0  # seconds — warn if below this
MAX_SAMPLE_DURATION = 60.0   # seconds — trim if longer

# ── Supported languages ─────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi"}

# ── Device detection ─────────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available torch device string."""
    if torch.backends.mps.is_available():
        return "cpu"  # XTTS v2 has issues with MPS; use CPU for stability
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
