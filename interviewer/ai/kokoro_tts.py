"""Text-to-speech with Kokoro (ONNX, CPU), 24 kHz."""

from __future__ import annotations

import asyncio
from pathlib import Path

from interviewer.audio.pcm import float32_to_pcm16, resample_linear


class KokoroTTS:
    sample_rate = 24000

    def __init__(self, model_path: Path, voices_path: Path, voice: str = "af_heart", speed: float = 1.0, engine=None):
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.voice = voice
        self.speed = speed
        self.voice_id = f"kokoro:{voice}"
        self._engine = engine
        self._lock = asyncio.Lock()

    def _load(self):
        from kokoro_onnx import Kokoro

        return Kokoro(str(self.model_path), str(self.voices_path))

    async def _ensure_engine(self):
        if self._engine is None:
            self._engine = await asyncio.to_thread(self._load)
        return self._engine

    async def warm_up(self) -> None:
        async with self._lock:
            await self._ensure_engine()

    async def synthesize(self, text: str) -> bytes:
        async with self._lock:
            engine = await self._ensure_engine()
            samples, rate = await asyncio.to_thread(engine.create, text, voice=self.voice, speed=self.speed, lang="en-us")
        if rate != self.sample_rate:
            samples = resample_linear(samples, rate, self.sample_rate)
        return float32_to_pcm16(samples)
