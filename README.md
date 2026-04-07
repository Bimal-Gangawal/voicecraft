# Voicecraft

Clone a voice from a 30-second audio sample and synthesize speech in **English** and **Hindi**.

Fully local — no cloud APIs, no data leaves your machine. Powered by [Coqui XTTS v2](https://github.com/coqui-ai/TTS).

## Requirements

- **Python 3.11** (not 3.12+ — required by TTS library)
- **macOS** (Apple Silicon recommended) or Linux
- **ffmpeg** (for MP3 export)

```bash
brew install python@3.11 ffmpeg
```

## Installation

```bash
cd voicecraft
uv venv .venv --python python3.11
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

### 1. Download the model (one-time, ~1.8 GB)

```bash
voicecraft setup
```

### 2. Clone a voice

**From an existing audio file:**
```bash
voicecraft clone --sample ~/my_voice_sample.wav --name my_voice
```

**Record directly from your microphone:**
```bash
voicecraft clone --record --name my_voice
```

The sample should be **6–60 seconds** of clear speech. 30 seconds is ideal.

### 3. Generate speech

**English:**
```bash
voicecraft speak --voice my_voice --text "Hello, this is my cloned voice." --lang en
```

**Hindi:**
```bash
voicecraft speak --voice my_voice --text "नमस्ते, यह मेरी क्लोन की गई आवाज़ है।" --lang hi
```

### 4. Output formats

```bash
# WAV (default)
voicecraft speak --voice my_voice --text "Hello" --lang en --format wav

# MP3
voicecraft speak --voice my_voice --text "Hello" --lang en --format mp3

# Custom output path
voicecraft speak --voice my_voice --text "Hello" --lang en -o ~/Desktop/output.wav

# Play immediately (macOS)
voicecraft speak --voice my_voice --text "Hello" --lang en --play
```

### One-shot mode (no saved profile)

```bash
voicecraft say --sample ~/voice.wav --text "Quick test" --lang en
```

## Recording

Record audio directly from your microphone without needing an external app.

```bash
# Record a 30-second sample (default) and save as WAV
voicecraft record

# Record with custom duration and output path
voicecraft record --duration 20 --output my_sample.wav

# List available audio input devices
voicecraft record --list-devices

# Use a specific microphone by device index
voicecraft record --device 2

# Record and clone in one step
voicecraft clone --record --name my_voice
voicecraft clone --record --name my_voice --duration 45
```

The recorder shows a live audio level meter and countdown timer while recording. Press `Ctrl+C` to stop early.

## CLI Reference

| Command | Description |
|---------|-------------|
| `voicecraft setup` | Download the XTTS v2 model |
| `voicecraft record` | Record audio from the microphone |
| `voicecraft clone -s <sample> -n <name>` | Extract and save a voice profile from a file |
| `voicecraft clone --record -n <name>` | Record from mic and save a voice profile |
| `voicecraft speak -v <voice> -t <text> -l <lang>` | Synthesize speech with a saved voice |
| `voicecraft say -s <sample> -t <text> -l <lang>` | One-shot clone + synthesize |
| `voicecraft voices` | List all saved voice profiles |

### List saved voices

```bash
voicecraft voices
```

## Supported Languages

| Code | Language |
|------|----------|
| `en` | English |
| `hi` | Hindi |

## Project Structure

```
voicecraft/
├── src/voicecraft/
│   ├── cli.py              # CLI commands (setup, record, clone, speak, say, voices)
│   ├── config.py           # Paths, device detection
│   ├── model_manager.py    # XTTS v2 model management + compatibility patches
│   ├── audio.py            # Audio preprocessing
│   ├── recorder.py         # Microphone recording with live level meter
│   ├── extractor.py        # Voice profile extraction
│   ├── synthesizer.py      # Speech synthesis
│   └── export.py           # WAV/MP3 export
├── voices/                 # Saved voice profiles
├── output/                 # Generated audio
└── tests/                  # Unit tests (32 tests)
```

## Tips for Best Results

- Use a **quiet recording** with minimal background noise
- **30 seconds** of natural speech works best
- The sample should be a **single speaker** only
- Avoid music or sound effects in the sample
- WAV or FLAC samples give the best quality (MP3 works too)
- Speak naturally — don't read too fast or too slow

## Running Tests

```bash
source .venv/bin/activate
uv run pytest tests/ -v
```
