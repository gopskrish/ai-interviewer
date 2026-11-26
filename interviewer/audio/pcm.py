"""PCM16 mono helpers (little-endian 16-bit samples)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_SCALE = 32767.0


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """PCM16 bytes -> float32 samples in [-1, 1]."""
    usable = len(pcm) - (len(pcm) % 2)
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / _SCALE
    return np.clip(samples, -1.0, 1.0)


def float32_to_pcm16(x: np.ndarray) -> bytes:
    """float32 samples -> PCM16 bytes, clipped to [-1, 1]."""
    clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (clipped * _SCALE).round().astype("<i2").tobytes()


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampling; good enough for speech."""
    x = np.asarray(x, dtype=np.float32)
    if src_rate == dst_rate or len(x) == 0:
        return x.copy()
    n_out = int(round(len(x) * dst_rate / src_rate))
    positions = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    return np.interp(positions, np.arange(len(x)), x).astype(np.float32)


def chunk_bytes(pcm: bytes, sample_rate: int, chunk_ms: int) -> list[bytes]:
    """Split PCM16 into chunks of ``chunk_ms``; the last chunk may be shorter."""
    step = (sample_rate * chunk_ms // 1000) * 2
    usable = len(pcm) - (len(pcm) % 2)
    return [pcm[i : i + step] for i in range(0, usable, step)]


def duration_s(pcm: bytes, sample_rate: int) -> float:
    return (len(pcm) // 2) / sample_rate


class WavAppender:
    """Writes PCM16 mono to a WAV file as it arrives; the header stays valid after each write."""

    def __init__(self, path: Path, sample_rate: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)
        self._closed = False

    def write(self, pcm: bytes) -> None:
        if not self._closed and pcm:
            self._wav.writeframes(pcm[: len(pcm) - (len(pcm) % 2)])

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._wav.close()
