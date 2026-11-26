"""Sends speech to the meeting at real-time pace, a little ahead of playback.

Attendee queues whatever it receives and has no command to clear the
queue, so the only way to stop the AI mid-sentence is to stop sending.
Keeping only ``lead_ms`` of audio in flight means a barge-in silences the
bot within about ``lead_ms + chunk_ms``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from interviewer.audio.pcm import chunk_bytes, duration_s

_EPSILON = 1e-9


class OutputPacer:
    def __init__(
        self,
        send: Callable[[bytes, int], Awaitable[None]],
        chunk_ms: int = 200,
        lead_ms: int = 200,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._send = send
        self._chunk_ms = chunk_ms
        self._chunk_s = chunk_ms / 1000
        self._lead_s = lead_ms / 1000
        self._clock = clock
        self._sleep = sleep
        self._stopped = False
        self._end_at: float | None = None

    @property
    def speaking(self) -> bool:
        return self._end_at is not None and self._clock() < self._end_at

    def stop(self) -> None:
        """Stop the current ``play`` before its next chunk."""
        self._stopped = True

    async def _wait_until(self, target: float) -> bool:
        """Sleep until ``target``; False if stopped meanwhile."""
        while (remaining := target - self._clock()) > _EPSILON:
            await self._sleep(min(self._chunk_s, remaining))
            if self._stopped:
                return False
        return not self._stopped

    async def play(self, pcm: bytes, sample_rate: int) -> bool:
        """Send ``pcm`` paced at real time. True if it all played, False if stopped."""
        self._stopped = False
        chunks = chunk_bytes(pcm, sample_rate, self._chunk_ms)
        if not chunks:
            return True
        start = self._clock()
        self._end_at = start + duration_s(pcm, sample_rate)
        try:
            for index, chunk in enumerate(chunks):
                due = start + index * self._chunk_s - self._lead_s
                if not await self._wait_until(due):
                    return False
                await self._send(chunk, sample_rate)
            return await self._wait_until(self._end_at)
        finally:
            self._end_at = None
