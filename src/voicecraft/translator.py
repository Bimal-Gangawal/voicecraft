"""Live translation pipeline: Mic → STT → Translate → TTS → Speaker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.text import Text

from voicecraft.config import DEVICE, SUPPORTED_LANGUAGES

console = Console()

# ── Constants ────────────────────────────────────────────────────────────────
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio
SILENCE_THRESHOLD = 0.015    # RMS below this is considered silence
SPEECH_THRESHOLD = 0.02      # RMS above this triggers speech detection
SILENCE_DURATION = 1.2       # Seconds of silence after speech to trigger processing
MIN_SPEECH_DURATION = 0.5    # Minimum speech duration in seconds to process
MAX_SPEECH_DURATION = 30.0   # Maximum before forced processing

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")


@dataclass
class TranslatorConfig:
    """Configuration for the live translator."""
    voice_name: str
    from_lang: str = "en"
    to_lang: str = "hi"
    whisper_model: str = "small"
    mic_device: int | None = None
    speed: float = 1.0
    temperature: float = 0.75


def _get_whisper_lang_code(lang: str) -> str:
    """Map voicecraft language codes to Whisper language codes."""
    return {"en": "en", "hi": "hi"}[lang]


def ensure_translation_models(from_lang: str, to_lang: str) -> None:
    """Download and install Argos Translate language pair if not present."""
    import argostranslate.package
    import argostranslate.translate

    # Check if the pair is already installed
    installed = argostranslate.translate.get_installed_languages()
    from_installed = None
    to_installed = None
    for lang in installed:
        if lang.code == from_lang:
            from_installed = lang
        if lang.code == to_lang:
            to_installed = lang

    if from_installed and to_installed:
        # Check if the direct translation exists
        translation = from_installed.get_translation(to_installed)
        if translation:
            return

    # Need to download and install
    console.print(f"[cyan]Downloading translation model: {from_lang} → {to_lang}...[/cyan]")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    pkg = next(
        (p for p in available if p.from_code == from_lang and p.to_code == to_lang),
        None,
    )
    if pkg is None:
        raise RuntimeError(
            f"No translation model found for {from_lang} → {to_lang}. "
            f"Available pairs: {[(p.from_code, p.to_code) for p in available]}"
        )

    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)
    console.print(f"[green]Translation model {from_lang} → {to_lang} installed.[/green]")


def translate_text(text: str, from_lang: str, to_lang: str) -> str:
    """Translate text using Argos Translate (fully offline)."""
    import argostranslate.translate

    result = argostranslate.translate.translate(text, from_lang, to_lang)
    return result


class LiveTranslator:
    """Continuously listens, transcribes, translates, and speaks back."""

    def __init__(self, config: TranslatorConfig) -> None:
        self.config = config
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False

        # Pre-validate
        if config.from_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {config.from_lang}")
        if config.to_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {config.to_lang}")
        if config.from_lang == config.to_lang:
            raise ValueError("Source and target languages must be different.")
        if config.whisper_model not in WHISPER_MODELS:
            raise ValueError(
                f"Unknown whisper model: {config.whisper_model}. "
                f"Choose from: {WHISPER_MODELS}"
            )

    def _load_whisper(self):
        """Load the faster-whisper model."""
        from faster_whisper import WhisperModel

        compute_type = "float16" if DEVICE == "cuda" else "int8"
        console.print(
            f"[cyan]Loading Whisper [bold]{self.config.whisper_model}[/bold] "
            f"on [bold]{DEVICE}[/bold] ({compute_type})...[/cyan]"
        )
        model = WhisperModel(
            self.config.whisper_model,
            device=DEVICE,
            compute_type=compute_type,
        )
        console.print("[green]Whisper model loaded.[/green]")
        return model

    def _load_tts(self):
        """Load voice profile and TTS model. Returns (model, gpt_latent, speaker_emb)."""
        from voicecraft.extractor import load_voice_profile
        from voicecraft.model_manager import load_model
        from voicecraft.synthesizer import VoiceSettings

        console.print(f"[cyan]Loading voice profile: [bold]{self.config.voice_name}[/bold][/cyan]")
        gpt_cond_latent, speaker_embedding, metadata = load_voice_profile(self.config.voice_name)
        gpt_cond_latent = gpt_cond_latent.to(DEVICE)
        speaker_embedding = speaker_embedding.to(DEVICE)

        tts = load_model()
        tts_model = tts.synthesizer.tts_model

        settings = VoiceSettings(
            speed=self.config.speed,
            temperature=self.config.temperature,
        )

        return tts_model, gpt_cond_latent, speaker_embedding, settings

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback — pushes audio chunks to queue."""
        self._audio_queue.put(indata[:, 0].copy())

    def _compute_rms(self, audio: np.ndarray) -> float:
        """Compute RMS energy of an audio buffer."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _build_status_bar(self, state: str, rms: float = 0.0, extra: str = "") -> Text:
        """Build a rich status line for the live display."""
        text = Text()

        if state == "listening":
            text.append("  🎤 ", style="bold green")
            text.append("Listening", style="bold green")
            # Audio level bar
            level = min(rms / 0.3, 1.0)
            width = 30
            filled = int(level * width)
            text.append("  [")
            for i in range(filled):
                if i < int(width * 0.6):
                    text.append("█", style="green")
                elif i < int(width * 0.85):
                    text.append("█", style="yellow")
                else:
                    text.append("█", style="red")
            text.append(" " * (width - filled))
            text.append("]")
        elif state == "recording":
            text.append("  🔴 ", style="bold red")
            text.append("Speech detected — recording", style="bold red")
            level = min(rms / 0.3, 1.0)
            width = 30
            filled = int(level * width)
            text.append("  [")
            for i in range(filled):
                text.append("█", style="red")
            text.append(" " * (width - filled))
            text.append("]")
        elif state == "transcribing":
            text.append("  📝 ", style="bold yellow")
            text.append("Transcribing...", style="bold yellow")
        elif state == "translating":
            text.append("  🌐 ", style="bold blue")
            text.append("Translating...", style="bold blue")
        elif state == "speaking":
            text.append("  🔊 ", style="bold magenta")
            text.append("Speaking translation...", style="bold magenta")

        if extra:
            text.append(f"  {extra}", style="dim")

        return text

    def run(self) -> None:
        """Main loop: listen → detect speech → transcribe → translate → speak."""
        import torch
        from voicecraft.export import play_audio
        from voicecraft.synthesizer import (
            VoiceSettings,
            _build_waveform,
            _normalize_text,
            _split_into_chunks,
        )

        # ── Load all models ──────────────────────────────────────────────
        console.print()
        console.print("[bold]━━━ Voicecraft Live Translator ━━━[/bold]")
        console.print()

        # Ensure translation models are downloaded
        ensure_translation_models(self.config.from_lang, self.config.to_lang)

        whisper = self._load_whisper()
        tts_model, gpt_cond_latent, speaker_embedding, settings = self._load_tts()

        from_name = SUPPORTED_LANGUAGES[self.config.from_lang]
        to_name = SUPPORTED_LANGUAGES[self.config.to_lang]

        console.print()
        console.print(f"[bold green]Ready![/bold green] Speak in {from_name} → hear in {to_name}")
        console.print(f"[dim]Voice: {self.config.voice_name} | Whisper: {self.config.whisper_model} | Device: {DEVICE}[/dim]")
        console.print(f"[dim]Press Ctrl+C to stop.[/dim]")
        console.print()

        self._running = True
        stream = sd.InputStream(
            samplerate=WHISPER_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.config.mic_device,
            callback=self._audio_callback,
            blocksize=int(WHISPER_SAMPLE_RATE * 0.1),  # 100ms chunks
        )

        try:
            with stream:
                while self._running:
                    self._process_one_utterance(
                        whisper, tts_model, gpt_cond_latent,
                        speaker_embedding, settings, play_audio,
                        _normalize_text, _split_into_chunks, _build_waveform,
                    )
        except KeyboardInterrupt:
            self._running = False
            console.print()
            console.print("[yellow]Live translation stopped.[/yellow]")

    def _process_one_utterance(
        self,
        whisper,
        tts_model,
        gpt_cond_latent,
        speaker_embedding,
        settings,
        play_audio,
        _normalize_text,
        _split_into_chunks,
        _build_waveform,
    ) -> None:
        """Wait for one speech utterance, process it, and speak the translation."""

        # ── Phase 1: Wait for speech ─────────────────────────────────────
        audio_buffer: list[np.ndarray] = []
        speech_detected = False
        silence_start: float | None = None
        speech_start: float | None = None

        with Live(console=console, refresh_per_second=10, transient=True) as live:
            while self._running:
                try:
                    chunk = self._audio_queue.get(timeout=0.15)
                except queue.Empty:
                    continue

                rms = self._compute_rms(chunk)

                if not speech_detected:
                    # Waiting for speech to start
                    live.update(self._build_status_bar("listening", rms))
                    if rms > SPEECH_THRESHOLD:
                        speech_detected = True
                        speech_start = time.time()
                        silence_start = None
                        audio_buffer.append(chunk)
                else:
                    # Speech in progress — accumulate
                    audio_buffer.append(chunk)
                    live.update(self._build_status_bar("recording", rms))

                    if rms < SILENCE_THRESHOLD:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            # Enough silence — speech utterance complete
                            break
                    else:
                        silence_start = None

                    # Force-break on very long speech
                    if speech_start and (time.time() - speech_start) >= MAX_SPEECH_DURATION:
                        break

        if not audio_buffer or not speech_detected:
            return

        # Combine buffer into a single array
        audio = np.concatenate(audio_buffer)
        duration = len(audio) / WHISPER_SAMPLE_RATE

        if duration < MIN_SPEECH_DURATION:
            return  # Too short, ignore

        # ── Phase 2: Transcribe ──────────────────────────────────────────
        console.print("  📝 [yellow]Transcribing...[/yellow]", end="")
        t0 = time.time()

        segments, info = whisper.transcribe(
            audio,
            language=_get_whisper_lang_code(self.config.from_lang),
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        t_transcribe = time.time() - t0

        if not transcript:
            console.print(" [dim](no speech detected)[/dim]")
            return

        console.print(f" [bold]\"{transcript}\"[/bold] [dim]({t_transcribe:.1f}s)[/dim]")

        # ── Phase 3: Translate ───────────────────────────────────────────
        console.print("  🌐 [blue]Translating...[/blue]", end="")
        t0 = time.time()

        translated = translate_text(transcript, self.config.from_lang, self.config.to_lang)
        t_translate = time.time() - t0

        console.print(f" [bold]\"{translated}\"[/bold] [dim]({t_translate:.1f}s)[/dim]")

        # ── Phase 4: Synthesize & play ───────────────────────────────────
        console.print("  🔊 [magenta]Speaking...[/magenta]", end="")
        t0 = time.time()

        text = _normalize_text(translated, self.config.to_lang)
        chunks = _split_into_chunks(text, self.config.to_lang)
        waveform = _build_waveform(
            chunks, tts_model, self.config.to_lang,
            gpt_cond_latent, speaker_embedding, settings,
        )
        t_synth = time.time() - t0
        console.print(f" [dim]({t_synth:.1f}s)[/dim]")

        # Drain the audio queue while we play to avoid echo/feedback
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        play_audio(waveform)
        console.print()
