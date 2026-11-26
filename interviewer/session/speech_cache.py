"""Synthesised speech kept in memory and on disk, so kit lines play instantly.

Lines are keyed by voice and text, so changing the voice re-synthesises.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from interviewer.ai.base import TextToSpeech

log = logging.getLogger(__name__)


class SpeechCache:
    def __init__(self, tts: TextToSpeech, cache_dir: Path):
        self._tts = tts
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, bytes] = {}

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self._tts.voice_id}\n{text}".encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.pcm"

    def has(self, text: str) -> bool:
        key = self._key(text)
        return key in self._memory or self._path(key).exists()

    async def get(self, text: str) -> bytes:
        key = self._key(text)
        if key in self._memory:
            return self._memory[key]
        path = self._path(key)
        if path.exists():
            audio = path.read_bytes()
        else:
            audio = await self._tts.synthesize(text)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(audio)
            os.replace(tmp, path)
        self._memory[key] = audio
        return audio

    async def warm(self, lines: list[str]) -> None:
        for line in dict.fromkeys(lines):
            if self.has(line):
                continue
            try:
                await self.get(line)
            except Exception:  # one bad line shouldn't stop the rest
                log.exception("couldn't synthesise %r", line)
