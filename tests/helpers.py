"""Shared test utilities: a controllable clock, synthetic audio, websocket helpers and a wired-up manager."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import numpy as np

from interviewer.ai.factory import AIBundle
from interviewer.ai.fakes import FakeLLM, FakeSTT, FakeTTS
from interviewer.attendee import BotInfo
from interviewer.config import Settings
from interviewer.kit import list_kits

REPO = Path(__file__).resolve().parents[1]
KITS_DIR = REPO / "kits"
TEAMS_LINK = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_abc%40thread.v2/0?context=%7b%22Tid%22%3a%22t%22%7d"


def mixed(pcm: bytes, rate: int = 16000) -> str:
    """An Attendee realtime_audio.mixed message carrying ``pcm``."""
    return json.dumps({"trigger": "realtime_audio.mixed", "bot_id": "b",
                       "data": {"chunk": base64.b64encode(pcm).decode(), "timestamp_ms": 0, "sample_rate": rate}})


def chunks(pcm: bytes, chunk_s: float = 0.1, rate: int = 16000) -> list[bytes]:
    step = int(rate * chunk_s) * 2
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


async def feed(runner, pcm: bytes, chunk_s: float = 0.1, rate: int = 16000, wait: bool = True) -> None:
    """Send ``pcm`` to a runner in 100 ms messages, then (by default) wait for it to go idle."""
    for piece in chunks(pcm, chunk_s, rate):
        await runner.handle_message(mixed(piece, rate))
        await asyncio.sleep(0)
    if wait:
        await runner.idle()


async def speak(runner, seconds: float = 2.5, pause_s: float = 3.5) -> None:
    """One candidate turn: ``seconds`` of tone, then enough silence to end the turn."""
    await feed(runner, tone(seconds) + silence(pause_s))


def collector(into: list):
    async def send(message: dict) -> None:
        into.append(message)
    return send


class FakeAttendee:
    """Records bot calls; ``state`` is what every get_bot returns."""

    def __init__(self, state: str = "joining"):
        self.state = state
        self.created: list[dict] = []
        self.left: list[str] = []
        self.create_error: Exception | None = None

    async def create_bot(self, meeting_url: str, bot_name: str, audio_ws_url: str, sample_rate: int = 16000) -> BotInfo:
        if self.create_error is not None:
            raise self.create_error
        self.created.append({"meeting_url": meeting_url, "bot_name": bot_name, "audio_ws_url": audio_ws_url})
        return BotInfo(id=f"bot_{len(self.created)}", state="joining")

    async def get_bot(self, bot_id: str) -> BotInfo:
        return BotInfo(id=bot_id, state=self.state)

    async def leave(self, bot_id: str) -> None:
        self.left.append(bot_id)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {"data_dir": tmp_path / "data", "kits_dir": KITS_DIR, "models_dir": tmp_path / "models",
              "attendee_api_key": "test-key"}
    values.update(overrides)
    return Settings(**values)


def make_manager(tmp_path: Path, clock, *, attendee=None, llm=None, stt=None, settings=None, http=None):
    from interviewer.session.manager import SessionManager
    from interviewer.session.store import SessionStore

    settings = settings or make_settings(tmp_path)
    ai = AIBundle(llm=llm or FakeLLM([]), stt=stt or FakeSTT([]), tts=FakeTTS(), new_vad=LoudnessVad)
    kits, _ = list_kits(settings.kits_dir)
    http = http or httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    return SessionManager(settings, SessionStore(settings.data_dir), ai, kits, attendee, http,
                          clock=clock, sleep=clock.sleep)


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
