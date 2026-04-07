# Voicecraft

Clone a voice from a 30-second audio sample and synthesize speech in **English** and **Hindi**.

Fully local — no cloud APIs, no data leaves your machine. Powered by [Coqui XTTS v2](https://github.com/coqui-ai/TTS).

---

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

---

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

The sample should be **6-60 seconds** of clear speech. 30 seconds is ideal.

### 3. Generate speech (plays directly through speakers)

**English:**
```bash
voicecraft speak --voice my_voice --text "Hello, this is my cloned voice." --lang en
```

**Hindi:**
```bash
voicecraft speak --voice my_voice --text "नमस्ते, यह मेरी क्लोन की गई आवाज़ है।" --lang hi
```

### 4. Save to file instead of playing

```bash
voicecraft speak --voice my_voice --text "Hello" --lang en --save output.wav
voicecraft speak --voice my_voice --text "Hello" --lang en --save output.mp3 --format mp3
```

---

<details>
<summary><strong>Recording</strong></summary>

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

</details>

<details>
<summary><strong>Voice Tuning (Accent & Prosody)</strong></summary>

Control how the cloned voice sounds with these flags on `speak` and `say`:

| Flag | Default | Description |
|------|---------|-------------|
| `--speed` | `1.0` | Speech rate multiplier. `0.5` = slow, `1.0` = normal, `2.0` = fast |
| `--temperature` | `0.75` | Expressiveness. `0.1` = flat/monotone, `0.75` = natural, `1.0+` = very expressive |
| `--repetition-penalty` | `10.0` | Penalises repeated sounds. Higher = less repetition |

**Examples:**

```bash
# Slower, more expressive speech
voicecraft speak -v my_voice -t "This is dramatic." -l en --speed 0.8 --temperature 0.9

# Fast, flat narration
voicecraft speak -v my_voice -t "Breaking news today." -l en --speed 1.3 --temperature 0.3

# Hindi with natural pacing
voicecraft speak -v my_voice -t "आज का मौसम बहुत अच्छा है।" -l hi --speed 0.9
```

</details>

<details>
<summary><strong>Punctuation Handling</strong></summary>

Voicecraft automatically inserts natural pauses based on punctuation:

| Punctuation | Pause Duration | Example |
|-------------|---------------|---------|
| `.` `!` `?` `।` | 350ms | End of sentence |
| `,` `;` | 180ms | Clause break |
| `:` `—` | 250ms | Introductory pause |
| `...` | 450ms | Trailing thought |
| Paragraph break | 550ms | New paragraph |

**Text normalization** is also applied automatically:
- `&` becomes "and" (English) or "aur" (Hindi)
- `%` becomes "percent" (English) or "pratishat" (Hindi)
- Smart quotes and dashes are normalized
- Missing sentence-ending punctuation is added for better prosody

</details>

<details>
<summary><strong>One-Shot Mode</strong></summary>

Clone and synthesize in a single command without saving a voice profile:

```bash
# Plays directly
voicecraft say --sample ~/voice.wav --text "Quick test" --lang en

# Save to file
voicecraft say --sample ~/voice.wav --text "Quick test" --lang en --save output.wav
```

</details>

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `voicecraft setup` | Download the XTTS v2 model |
| `voicecraft record` | Record audio from the microphone |
| `voicecraft clone -s <sample> -n <name>` | Clone voice from a file |
| `voicecraft clone --record -n <name>` | Record from mic and clone |
| `voicecraft speak -v <voice> -t <text> -l <lang>` | Speak (plays directly) |
| `voicecraft speak ... --save <path>` | Speak and save to file |
| `voicecraft say -s <sample> -t <text> -l <lang>` | One-shot clone + speak |
| `voicecraft voices` | List saved voice profiles |

## Supported Languages

| Code | Language |
|------|----------|
| `en` | English |
| `hi` | Hindi |

---

<details>
<summary><strong>Project Structure</strong></summary>

```
voicecraft/
├── src/voicecraft/
│   ├── cli.py              # CLI commands (setup, record, clone, speak, say, voices)
│   ├── config.py           # Paths, device detection, constants
│   ├── model_manager.py    # XTTS v2 model management + compatibility patches
│   ├── audio.py            # Audio loading, resampling, trimming, normalization
│   ├── recorder.py         # Microphone recording with live level meter
│   ├── extractor.py        # Voice profile extraction (speaker latents)
│   ├── synthesizer.py      # Text processing, chunking, and speech synthesis
│   └── export.py           # WAV/MP3 export and direct playback
├── voices/                 # Saved voice profiles
├── output/                 # Generated audio (when using --save)
└── tests/                  # 51 unit tests
```

</details>

<details>
<summary><strong>Tips for Best Results</strong></summary>

- Use a **quiet recording** with minimal background noise
- **30 seconds** of natural speech works best
- The sample should be a **single speaker** only
- Avoid music or sound effects in the sample
- WAV or FLAC samples give the best quality (MP3 works too)
- Speak naturally — don't read too fast or too slow
- For Hindi, use Devanagari script for best pronunciation

</details>

<details>
<summary><strong>Running Tests</strong></summary>

```bash
source .venv/bin/activate
uv run pytest tests/ -v
```

51 tests covering audio processing, text normalization, punctuation pauses, chunking, export, and voice profile management.

</details>
