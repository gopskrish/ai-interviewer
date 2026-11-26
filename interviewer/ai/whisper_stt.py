"""Speech-to-text with faster-whisper on the CPU."""

from __future__ import annotations

import asyncio
from pathlib import Path

from interviewer.audio.pcm import pcm16_to_float32, resample_linear

WHISPER_RATE = 16000


class FasterWhisperSTT:
    def __init__(
        self,
        model_size: str = "small",
        cpu_threads: int = 4,
        download_root: Path | None = None,
        model=None,
    ):
        self.model_size = model_size
        self.cpu_threads = cpu_threads
        self.download_root = Path(download_root) if download_root else None
        self._model = model
        self._lock = asyncio.Lock()

    def _load(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.cpu_threads,
            download_root=str(self.download_root) if self.download_root else None,
        )

    async def _ensure_model(self):
        if self._model is None:
            self._model = await asyncio.to_thread(self._load)
        return self._model

    async def warm_up(self) -> None:
        async with self._lock:
            await self._ensure_model()

    def _run(self, model, audio, prompt: str) -> str:
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            initial_prompt=prompt or None,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, pcm16: bytes, sample_rate: int, prompt: str = "") -> str:
        audio = pcm16_to_float32(pcm16)
        if sample_rate != WHISPER_RATE:
            audio = resample_linear(audio, sample_rate, WHISPER_RATE)
        if len(audio) == 0:
            return ""
        async with self._lock:
            model = await self._ensure_model()
            return await asyncio.to_thread(self._run, model, audio, prompt)
