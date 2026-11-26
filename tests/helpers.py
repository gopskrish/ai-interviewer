"""Shared test utilities: a controllable clock and synthetic audio."""

from __future__ import annotations

import asyncio

import numpy as np


class FakeClock:
    """A monotonic clock that only moves when ``sleep`` is awaited (or ``now`` is set)."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    async def sleep(self, dt: float) -> None:
        if dt > 0:
            self.now += dt
        await asyncio.sleep(0)


class LoudnessVad:
    """A stand-in VAD: any frame louder than RMS 0.01 counts as speech."""

    def speech_probability(self, frame: np.ndarray) -> float:
        return 1.0 if float(np.sqrt(np.mean(np.square(frame)))) > 0.01 else 0.0

    def reset(self) -> None:
        pass


def tone(seconds: float, rate: int = 16000) -> bytes:
    """A 440 Hz sine wave at amplitude 0.3, as PCM16 mono."""
    n = int(round(seconds * rate))
    t = np.arange(n) / rate
    samples = 0.3 * np.sin(2 * np.pi * 440 * t)
    return (samples * 32767).round().astype("<i2").tobytes()


def silence(seconds: float, rate: int = 16000) -> bytes:
    """Digital silence as PCM16 mono."""
    return b"\x00\x00" * int(round(seconds * rate))
