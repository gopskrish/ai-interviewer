"""Interfaces every AI backend implements, so local models and OpenAI can be swapped."""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    """The language model couldn't give a usable answer."""


class LLMTimeout(LLMError):
    """The language model took too long."""


class LanguageModel(Protocol):
    async def complete_json(self, system: str, user: str, schema: dict, timeout_s: float) -> dict:
        """Return a JSON object matching ``schema``, or raise ``LLMError``."""
        ...


class SpeechToText(Protocol):
    async def transcribe(self, pcm16: bytes, sample_rate: int, prompt: str = "") -> str: ...


class TextToSpeech(Protocol):
    sample_rate: int
    voice_id: str

    async def synthesize(self, text: str) -> bytes:
        """Return PCM16 mono audio at ``sample_rate``."""
        ...
