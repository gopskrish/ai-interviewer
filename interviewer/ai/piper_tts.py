"""Text-to-speech with Piper: faster than Kokoro on slow PCs, a bit more robotic."""

from __future__ import annotations

import asyncio
from pathlib import Path


class PiperTTS:
    def __init__(self, voice_path: Path, voice=None):
        self.voice_path = Path(voice_path)
        self.voice_id = f"piper:{self.voice_path.stem}"
        self._voice = voice
        self._lock = asyncio.Lock()

    def _load(self):
        from piper import PiperVoice

        return PiperVoice.load(str(self.voice_path))

    @property
    def sample_rate(self) -> int:
        if self._voice is None:
            self._voice = self._load()
        return int(self._voice.config.sample_rate)

    async def warm_up(self) -> None:
        async with self._lock:
            if self._voice is None:
                self._voice = await asyncio.to_thread(self._load)

    def _run(self, text: str) -> bytes:
        return b"".join(chunk.audio_int16_bytes for chunk in self._voice.synthesize(text))

    async def synthesize(self, text: str) -> bytes:
        async with self._lock:
            if self._voice is None:
                self._voice = await asyncio.to_thread(self._load)
            return await asyncio.to_thread(self._run, text)
