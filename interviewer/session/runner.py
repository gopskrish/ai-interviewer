"""One live interview: meeting audio in, the AI's voice out.

Audio from Attendee (or the practice page) goes through turn detection.
Speech is transcribed segment by segment while the candidate talks. When
their turn ends, a short acknowledgment plays at once while the engine
decides what to say next, then the reply is spoken at real-time pace.

Two rules protect consent: script lines (the intro with its disclosure,
and the closing) can't be talked over, and anything the candidate said
while the intro was still playing doesn't count as the answer to
"Is that OK with you?".
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable

from interviewer.ai.base import SpeechToText
from interviewer.audio.pacer import OutputPacer
from interviewer.audio.pcm import WavAppender, float32_to_pcm16, pcm16_to_float32, resample_linear
from interviewer.audio.vad import (
    SAMPLE_RATE,
    BargeIn,
    SegmentReady,
    SpeechStarted,
    TurnConfig,
    TurnDetector,
    TurnEnded,
    VadModel,
)
from interviewer.engine.engine import Action, End, EngineLike, Say, Stage
from interviewer.session.protocol import ProtocolError, bot_output_message, parse_audio_message
from interviewer.session.speech_cache import SpeechCache
from interviewer.session.store import SessionRecord, SessionStore
from interviewer.session.transcript import Transcript, Turn

log = logging.getLogger(__name__)

ACK_MIN_SPEECH_MS = 1000
AUDIO_GAP_S = 0.2
NUDGE_THRESHOLDS = (10, 30, 90)
NUDGE_RESET_KINDS = {"question", "follow_up", "repeat", "clarify"}
UNINTERRUPTIBLE_KINDS = {"script"}
INTRO_GRACE_S = 1.0
DEFAULT_ACKS = ["Thanks."]

_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")

Send = Callable[[dict], Awaitable[None]]


class SessionRunner:
    def __init__(
        self,
        record: SessionRecord,
        engine: EngineLike,
        stt: SpeechToText,
        cache: SpeechCache,
        vad: VadModel,
        store: SessionStore,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        turn_config: TurnConfig | None = None,
        stt_prompt: str = "",
    ):
        self.record = record
        self.engine = engine
        self._stt = stt
        self._cache = cache
        self._store = store
        self._clock = clock
        self._stt_prompt = stt_prompt
        self._detector = TurnDetector(vad, turn_config)
        self._pacer = OutputPacer(self._send_audio, clock=clock, sleep=sleep)
        self._send: Send | None = None
        self._play_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

        self.transcript = Transcript()
        self.finished = asyncio.Event()
        self.end_reason: str | None = None
        self.begun = False

        self._t0 = clock()
        self._audio_origin: float | None = None
        self._last_audio_at: float | None = None
        self._wav: WavAppender | None = None
        self._segments: list[asyncio.Task] = []
        self._turn_started_at: float | None = None
        self._current_say: Say | None = None
        self._batch_cancelled = False
        self._script_active = False
        self._script_ended_at: float | None = None
        self._quiet_since: float | None = None
        self._nudges_sent: set[int] = set()
        self._ack_index = 0
        self._busy = 0

    # ----- connection ----------------------------------------------------

    @property
    def speaking_text(self) -> str | None:
        return self._current_say.text if self._current_say else None

    def attach(self, send: Send) -> None:
        """Start sending to ``send``. On a reconnect, the time away doesn't count as silence."""
        self._send = send
        self._last_audio_at = None
        if self._quiet_since is not None:
            self._quiet_since = self._clock()

    def detach(self) -> None:
        self._send = None

    async def _send_audio(self, chunk: bytes, sample_rate: int) -> None:
        send = self._send
        if send is None:
            return
        try:
            await send(bot_output_message(chunk, sample_rate))
        except Exception:
            log.info("websocket send failed; dropping audio until it reconnects")
            self._send = None

    # ----- lifecycle -----------------------------------------------------

    async def begin(self) -> None:
        if self.begun:
            return
        self.begun = True
        await self._perform(self.engine.start())

    async def end(self, reason: str) -> None:
        if self.finished.is_set():
            return
        self._batch_cancelled = True
        self._pacer.stop()
        actions = self.engine.end_now(reason)
        if self._send is None:
            await self._finish(reason)
            return
        await self._perform(actions)

    async def idle(self) -> None:
        """Wait until all background work (speech, transcription, decisions) is done.

        Only unfinished tasks are awaited: a finished task stays in ``_tasks``
        until its done-callback runs, and gathering finished tasks doesn't
        yield to the event loop, so waiting on them would spin forever.
        """
        while pending := [task for task in self._tasks if not task.done()]:
            await asyncio.gather(*pending, return_exceptions=True)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _now(self) -> float:
        return self._clock() - self._t0

    # ----- audio in --------------------------------------------------------

    async def handle_message(self, raw: str) -> None:
        if self.finished.is_set():
            return
        try:
            audio = parse_audio_message(raw)
        except ProtocolError as exc:
            log.warning("ignoring bad websocket message: %s", exc)
            return
        if audio is None:
            return
        pcm = audio.pcm
        if audio.sample_rate != SAMPLE_RATE:
            pcm = float32_to_pcm16(resample_linear(pcm16_to_float32(pcm), audio.sample_rate, SAMPLE_RATE))
        self._last_audio_at = self._clock()
        self._take_audio(pcm)

    async def tick(self) -> None:
        """Called every ~100 ms: keeps turn detection moving and nudges after long silences."""
        if self.finished.is_set():
            return
        now = self._clock()
        if self._last_audio_at is not None and now - self._last_audio_at > AUDIO_GAP_S:
            gap = now - self._last_audio_at
            self._last_audio_at = now
            self._take_audio(b"\x00\x00" * int(gap * SAMPLE_RATE))
        if (not self.begun or self._busy or self._play_lock.locked() or self._pacer.speaking
                or self._detector.in_speech or self._quiet_since is None):
            return
        quiet = now - self._quiet_since
        for threshold in NUDGE_THRESHOLDS:
            if quiet >= threshold and threshold not in self._nudges_sent:
                self._nudges_sent.add(threshold)
                try:
                    actions = self.engine.on_silence(threshold)
                except Exception as exc:
                    log.exception("engine failed on silence")
                    self.engine.events.append(f"engine_error:{exc}")
                    return
                if actions:
                    self._spawn(self._perform(actions))
                return

    def _take_audio(self, pcm: bytes) -> None:
        if self._audio_origin is None:
            self._audio_origin = self._now()
        if self._wav is None:
            self._wav = WavAppender(self._store.dir(self.record.id) / "meeting-audio.wav", SAMPLE_RATE)
        self._wav.write(pcm)
        for event in self._detector.feed(pcm, ai_speaking=self._pacer.speaking):
            self._on_event(event)

    def _on_event(self, event) -> None:
        if isinstance(event, SpeechStarted):
            self._turn_started_at = self._clock()
            self._quiet_since = None
        elif isinstance(event, SegmentReady):
            if self.begun:
                self._segments.append(self._spawn(self._transcribe(event.pcm)))
        elif isinstance(event, TurnEnded):
            self._quiet_since = self._clock()
            segments, self._segments = self._segments, []
            if not self.begun:
                self._spawn(self.begin())
                return
            if not self.finished.is_set():
                self._spawn(self._handle_turn(event, segments, self._turn_started_at))
        elif isinstance(event, BargeIn):
            self._barge_in()

    async def _transcribe(self, pcm: bytes) -> str:
        try:
            return await self._stt.transcribe(pcm, SAMPLE_RATE, self._stt_prompt)
        except Exception as exc:
            log.exception("speech-to-text failed")
            self.engine.events.append(f"stt_error:{exc}")
            return ""

    # ----- turns -------------------------------------------------------------

    def _is_intro_overlap(self, started_at: float | None) -> bool:
        # Engines without a consent question (the day-one echo check) keep these turns:
        # there, hearing speech during the greeting is exactly what's being tested.
        if not getattr(self.engine, "asks_consent", True):
            return False
        if self.engine.stage != Stage.INTRO or started_at is None:
            return False
        if self._script_active:
            return True
        return self._script_ended_at is not None and started_at < self._script_ended_at - INTRO_GRACE_S

    def _next_ack(self) -> str:
        kit = getattr(self.engine, "kit", None)
        acks = list(getattr(kit, "acknowledgments", None) or DEFAULT_ACKS)
        ack = acks[self._ack_index % len(acks)]
        self._ack_index += 1
        return ack

    async def _handle_turn(self, event: TurnEnded, segments: list[asyncio.Task], started_at: float | None) -> None:
        self._busy += 1
        try:
            async with self._turn_lock:
                if self.finished.is_set():
                    return
                if self._is_intro_overlap(started_at):
                    return
                origin = self._audio_origin or 0.0
                topic = self.engine.current_topic_id
                turn = Turn("candidate", "", origin + event.start_s, origin + event.end_s, topic_id=topic)
                self.transcript.add(turn)
                ack_task = None
                if event.speech_ms >= ACK_MIN_SPEECH_MS and self.engine.stage != Stage.INTRO:
                    ack_task = self._spawn(self._perform([Say(self._next_ack(), "ack", topic)]))
                texts = await asyncio.gather(*segments, return_exceptions=True)
                text = " ".join(t.strip() for t in texts if isinstance(t, str) and t.strip()).strip()
                if text:
                    turn.text = text
                else:
                    self.transcript.turns.remove(turn)
                try:
                    actions = await self.engine.on_candidate_turn(text)
                except Exception as exc:
                    log.exception("engine failed on a turn")
                    self.engine.events.append(f"engine_error:{exc}")
                    actions = []
                if ack_task is not None:
                    await ack_task
                if actions:
                    await self._perform(actions)
        finally:
            self._busy -= 1

    # ----- speaking ----------------------------------------------------------

    async def _perform(self, actions: list[Action]) -> None:
        async with self._play_lock:
            self._batch_cancelled = False
            for action in actions:
                if isinstance(action, End):
                    await self._finish(action.reason)
                    return
                if self._batch_cancelled or self.finished.is_set():
                    continue
                await self._say(action)

    async def _say(self, say: Say) -> None:
        turn = Turn("ai", say.text, self._now(), self._now(), kind=say.kind, topic_id=say.topic_id)
        self.transcript.add(turn)
        self._current_say = say
        if say.kind in UNINTERRUPTIBLE_KINDS:
            self._script_active = True
        interrupted = False
        try:
            pieces = [say.text] if self._cache.has(say.text) else _SENTENCE_END.split(say.text.strip())
            for piece in pieces:
                if not piece.strip():
                    continue
                if self._batch_cancelled:
                    interrupted = True
                    break
                audio = await self._cache.get(piece)
                if not await self._pacer.play(audio, self._cache.sample_rate):
                    interrupted = True
                    break
        except Exception as exc:
            log.exception("couldn't speak %r", say.text)
            self.engine.events.append(f"tts_error:{exc}")
        finally:
            turn.end_s = self._now()
            turn.interrupted = interrupted
            self._current_say = None
            if say.kind in UNINTERRUPTIBLE_KINDS:
                self._script_active = False
                self._script_ended_at = self._clock()
            if say.kind != "nudge":
                self._quiet_since = self._clock()
            if say.kind in NUDGE_RESET_KINDS:
                self._nudges_sent = set()

    def _barge_in(self) -> None:
        say = self._current_say
        if say is None or say.kind in UNINTERRUPTIBLE_KINDS:
            return
        self._batch_cancelled = True
        self._pacer.stop()

    async def _finish(self, reason: str) -> None:
        if self.finished.is_set():
            return
        self.end_reason = reason
        if self._wav is not None:
            self._wav.close()
        self.transcript.save(self._store.dir(self.record.id))
        self.record.status = "scoring"
        self.record.end_reason = reason
        self.record.flags = sorted(self.engine.flags)
        self._store.save(self.record)
        self.finished.set()
