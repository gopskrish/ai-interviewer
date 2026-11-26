from pathlib import Path

import httpx
import pytest

from interviewer.ai.factory import AIBundle, NotBuiltYet, build_ai
from interviewer.ai.kokoro_tts import KokoroTTS
from interviewer.ai.ollama import OllamaLLM
from interviewer.ai.piper_tts import PiperTTS
from interviewer.ai.whisper_stt import FasterWhisperSTT
from interviewer.audio.vad import SileroVad
from interviewer.config import Settings


def test_build_ai_openai_mode_fails_fast():
    with pytest.raises(NotBuiltYet, match="AI_MODE=local"):
        build_ai(Settings.from_env({"AI_MODE": "openai"}), httpx.AsyncClient())


def test_build_ai_local_defaults_to_kokoro_and_ollama():
    bundle = build_ai(Settings.from_env({}), httpx.AsyncClient())
    assert isinstance(bundle, AIBundle)
    assert isinstance(bundle.llm, OllamaLLM) and bundle.llm.model == "qwen3:4b-instruct"
    assert bundle.llm.base_url == "http://ollama:11434"
    assert isinstance(bundle.tts, KokoroTTS) and bundle.tts.voice_id == "kokoro:af_heart"
    assert isinstance(bundle.stt, FasterWhisperSTT) and bundle.stt.download_root == Path("/models/whisper")


def test_build_ai_local_picks_piper_when_configured():
    bundle = build_ai(Settings.from_env({"TTS_ENGINE": "piper"}), httpx.AsyncClient())
    assert isinstance(bundle.tts, PiperTTS)
    assert isinstance(bundle.new_vad(), SileroVad)
