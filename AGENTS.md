# Voicecraft — Build & Dev Commands

## Install (editable)
```bash
cd voicecraft && uv pip install -e ".[dev]"
```

### Windows (with GPU)
```powershell
uv venv .venv --python 3.11
uv pip install -e . --python .venv\Scripts\python.exe
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --python .venv\Scripts\python.exe --reinstall-package torch --reinstall-package torchaudio
```

## Run CLI
```bash
voicecraft setup          # Download XTTS v2 model + translation models
voicecraft clone --sample <path> --name <name>
voicecraft speak --voice <name> --text "..." --lang en
voicecraft translate --voice <name> --from en --to hi   # Live translation
voicecraft voices         # List saved profiles
```

## Run tests
```bash
cd voicecraft && uv run pytest tests/ -v
```

## System dependencies

### macOS
```bash
brew install ffmpeg   # Required for MP3 export
```

### Windows
```powershell
winget install --id Gyan.FFmpeg -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

## Notes
- Model cache: ~/.local/share/voicecraft/models/
- Voice profiles: ./voices/
- Output: ./output/
- Device auto-detection: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
