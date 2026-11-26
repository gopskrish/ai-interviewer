"""Builds the AI pieces chosen in settings (AI_MODE, TTS_ENGINE)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from interviewer.ai.base import LanguageModel, SpeechToText, TextToSpeech
from interviewer.ai.kokoro_tts import KokoroTTS
from interviewer.ai.ollama import OllamaLLM
from interviewer.ai.piper_tts import PiperTTS
from interviewer.ai.whisper_stt import FasterWhisperSTT
from interviewer.audio.vad import SileroVad, VadModel
from interviewer.config import Settings


class NotBuiltYet(RuntimeError):
    """The requested mode exists in the design but hasn't been built."""


@dataclass
class AIBundle:
    llm: LanguageModel
    stt: SpeechToText
    tts: TextToSpeech
    new_vad: Callable[[], VadModel]


def build_ai(settings: Settings, http: httpx.AsyncClient) -> AIBundle:
    if settings.ai_mode == "openai":
        raise NotBuiltYet(
            "OpenAI mode isn't built yet. Set AI_MODE=local in .env to use the free local models."
        )
    llm = OllamaLLM(settings.ollama_url, settings.ollama_model, http)
    stt = FasterWhisperSTT(
        settings.whisper_model,
        settings.whisper_cpu_threads,
        download_root=settings.models_dir / "whisper",
    )
    if settings.tts_engine == "piper":
        tts: TextToSpeech = PiperTTS(settings.piper_voice)
    else:
        tts = KokoroTTS(settings.kokoro_model, settings.kokoro_voices, voice=settings.kokoro_voice)
    return AIBundle(llm=llm, stt=stt, tts=tts, new_vad=SileroVad)
