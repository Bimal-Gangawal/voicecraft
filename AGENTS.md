# Voicecraft — Build & Dev Commands

## Install (editable)
```bash
cd voicecraft && uv pip install -e ".[dev]"
```

## Run CLI
```bash
voicecraft setup          # Download XTTS v2 model
voicecraft clone --sample <path> --name <name>
voicecraft speak --voice <name> --text "..." --lang en
voicecraft voices         # List saved profiles
```

## Run tests
```bash
cd voicecraft && uv run pytest tests/ -v
```

## System dependency
```bash
brew install ffmpeg   # Required for MP3 export
```

## Notes
- Model cache: ~/.local/share/voicecraft/models/
- Voice profiles: ./voices/
- Output: ./output/
- Device auto-detection: MPS (Apple Silicon) > CPU
