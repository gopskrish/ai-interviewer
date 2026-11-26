"""Stand-in AI pieces for tests and offline development."""

from __future__ import annotations

import copy

from interviewer.ai.base import LLMError


class FakeLLM:
    """Returns queued responses in order; queued exceptions are raised."""

    def __init__(self, responses: list[dict | Exception]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    async def complete_json(self, system: str, user: str, schema: dict, timeout_s: float) -> dict:
        self.calls.append((system, user, schema))
        if not self._responses:
            raise LLMError("no more fake responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


class FakeSTT:
    """Returns queued transcripts in order, then empty strings."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.calls = 0

    async def transcribe(self, pcm16: bytes, sample_rate: int, prompt: str = "") -> str:
        self.calls += 1
        return self._texts.pop(0) if self._texts else ""


class FakeTTS:
    """Returns silence whose length is proportional to the text."""

    voice_id = "fake"

    def __init__(self, sample_rate: int = 24000, seconds_per_char: float = 0.01):
        self.sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        return b"\x00\x00" * int(round(len(text) * self.seconds_per_char * self.sample_rate))
