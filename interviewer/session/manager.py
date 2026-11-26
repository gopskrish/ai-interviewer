"""Live sessions: builds runners, sends bots into meetings, handles reconnects, scores finished interviews.

There is one manager per service. A session has a SessionRunner while it's
live. Background loops tick the connected runners every 100 ms, poll
Attendee for bot status every 5 s and delete old sessions once a day.

What happens when a runner finishes:
- the bot leaves the call (after the last words of the closing line);
- a session that never began isn't scored (an unused practice session is
  deleted; an interview is marked "ended");
- otherwise the evaluator scores it in the background, waiting while another
  interview is live so the two don't fight over the CPU, then the scorecard
  is saved and, for interviews, posted to Teams if a webhook is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable

import httpx

from interviewer.ai.factory import AIBundle
from interviewer.attendee import AttendeeClient, AttendeeError, join_timed_out, normalize_meeting_link, status_label
from interviewer.audio.vad import TurnConfig
from interviewer.config import Settings
from interviewer.engine.engine import EngineLike, InterviewEngine, generate_experience_questions
from interviewer.kit import Kit
from interviewer.scoring.evaluator import evaluate_interview
from interviewer.scoring.teams import post_scorecard
from interviewer.session.runner import Send, SessionRunner
from interviewer.session.speech_cache import SpeechCache
from interviewer.session.store import SessionRecord, SessionStore
from interviewer.session.transcript import Transcript, format_ts, write_atomic

log = logging.getLogger(__name__)

EngineFactory = Callable[[Kit, "list[str] | None"], EngineLike]

LIVE_STATUSES = {"created", "waiting_for_bot", "in_progress"}
BOT_GONE_STATES = {"leaving", "post_processing", "ended", "fatal_error", "data_deleted"}
BOT_KINDS = {"interview", "check"}
JOIN_TIMEOUT_S = 180.0
AUDIO_WAIT_S = 30.0
LEAVE_DELAY_S = 1.5

NO_ATTENDEE = ("Teams interviews need Attendee. Set ATTENDEE_API_KEY in .env (see the README) and restart "
               "the interviewer. Practice works without it.")
RESTARTED = "The interviewer service restarted while this session was running."
NOT_SCORED = "Ended before the interview began, so there's nothing to score."
NO_AUDIO = ("The bot is in the call, but its audio connection to the interviewer hasn't arrived. "
            "Check the certificate steps in the README and `docker compose logs attendee-worker`.")

STATUS_LABELS = {
    "created": "Not started",
    "waiting_for_bot": "Bot joining",
    "in_progress": "In progress",
    "scoring": "Scoring…",
    "done": "Done",
    "failed": "Failed",
    "ended": "Ended early",
}


class SessionError(Exception):
    """A problem the owner can fix, with a message for the page."""


class SessionManager:
    GRACE_S = 300
    TICK_S = 0.1
    POLL_S = 5.0
    CLEANUP_S = 24 * 3600
    SCORING_WAIT_S = 5.0

    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        ai: AIBundle,
        kits: dict[str, Kit],
        attendee: AttendeeClient | None,
        http: httpx.AsyncClient,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        engine_factory: EngineFactory | None = None,
    ):
        self.settings = settings
        self.store = store
        self.ai = ai
        self.kits = kits
        self.attendee = attendee
        self._http = http
        self._clock = clock
        self._sleep = sleep
        self._engine_factory = engine_factory or self._interview_engine
        self.cache = SpeechCache(ai.tts, settings.data_dir / "tts-cache")

        self._runners: dict[str, SessionRunner] = {}
        self._sockets: dict[str, Send] = {}
        self._detached_at: dict[str, float] = {}
        self._bot_started_at: dict[str, float] = {}
        self._joined_at: dict[str, float] = {}
        self._ever_connected: set[str] = set()
        self._left: set[str] = set()
        self._handled: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._loops: list[asyncio.Task] = []

    # ----- creating sessions ---------------------------------------------------

    def _interview_engine(self, kit: Kit, experience_questions: list[str] | None) -> EngineLike:
        return InterviewEngine(kit, self.ai.llm, clock=self._clock, experience_questions=experience_questions,
                               llm_timeout_s=self.settings.llm_timeout_s)

    def _kit(self, kit_id: str) -> Kit:
        kit = self.kits.get(kit_id)
        if kit is None:
            raise SessionError(f"There's no usable interview kit called {kit_id!r}.")
        return kit

    async def create_interview(self, kit_id: str, candidate_name: str, meeting_link: str,
                               notes: str = "") -> SessionRecord:
        kit = self._kit(kit_id)
        name = (candidate_name or "").strip()
        if not name:
            raise SessionError("Enter the candidate's name.")
        meeting_url = normalize_meeting_link(meeting_link)
        if self.attendee is None:
            raise SessionError(NO_ATTENDEE)
        notes = (notes or "").strip()
        record = self.store.create("interview", name, kit_id, meeting_url, notes)
        experience = await generate_experience_questions(kit, notes, self.ai.llm) if notes else None
        self._start_runner(record, self._engine_factory(kit, experience), kit.vocabulary)
        await self._send_bot(record)
        return record

    async def create_practice(self, kit_id: str) -> SessionRecord:
        kit = self._kit(kit_id)
        record = self.store.create("practice", "Practice", kit_id)
        self._start_runner(record, self._engine_factory(kit, None), kit.vocabulary)
        self._detached_at[record.id] = self._clock()  # deleted if nobody connects within the grace period
        return record

    def _start_runner(self, record: SessionRecord, engine: EngineLike, vocabulary=()) -> SessionRunner:
        config = TurnConfig(end_of_turn_silence_ms=int(self.settings.end_of_turn_silence_s * 1000))
        runner = SessionRunner(record, engine, self.ai.stt, self.cache, self.ai.new_vad(), self.store,
                               clock=self._clock, sleep=self._sleep, turn_config=config,
                               stt_prompt=", ".join(vocabulary))
        self._runners[record.id] = runner
        self._spawn(self.cache.warm(engine.planned_lines()))
        return runner

    async def _send_bot(self, record: SessionRecord) -> None:
        audio_url = f"{self.settings.public_ws_base.rstrip('/')}/ws/audio/{record.id}?token={record.token}"
        try:
            bot = await self.attendee.create_bot(record.meeting_url, self.settings.bot_name, audio_url)
        except AttendeeError as exc:
            self._drop(record.id)
            self._fail(record, f"Couldn't send the bot to the meeting. {exc}")
            return
        record.bot_id, record.bot_state, record.status = bot.id, bot.state, "waiting_for_bot"
        self._bot_started_at[record.id] = self._clock()
        self.store.save(record)

    def runner(self, session_id: str) -> SessionRunner | None:
        return self._runners.get(session_id)

    def record(self, session_id: str) -> SessionRecord | None:
        runner = self._runners.get(session_id)
        return runner.record if runner is not None else self.store.get(session_id)

    # ----- websocket connections ------------------------------------------------

    async def on_ws_connect(self, session_id: str, token: str, send: Send) -> SessionRunner:
        runner = self._runners.get(session_id)
        if runner is None or not secrets.compare_digest(str(token or "").encode(), runner.record.token.encode()):
            raise PermissionError("unknown session or wrong token")
        self._sockets[session_id] = send
        self._detached_at.pop(session_id, None)
        self._ever_connected.add(session_id)
        if runner.record.error == NO_AUDIO:
            runner.record.error = None
            self.store.save(runner.record)
        runner.attach(send)
        return runner

    async def on_ws_disconnect(self, session_id: str, send: Send | None = None) -> None:
        """Detach the socket. ``send`` identifies which socket closed; a stale one is ignored."""
        current = self._sockets.get(session_id)
        if current is None or (send is not None and send is not current):
            return
        del self._sockets[session_id]
        runner = self._runners.get(session_id)
        if runner is not None:
            runner.detach()
            if not runner.finished.is_set():
                self._detached_at[session_id] = self._clock()

    # ----- owner actions -----------------------------------------------------------

    async def begin(self, session_id: str) -> None:
        runner = self._runners.get(session_id)
        if runner is None or runner.finished.is_set():
            raise SessionError("This session isn't running any more.")
        if session_id not in self._sockets:
            if runner.record.kind == "practice":
                raise SessionError("The microphone isn't connected yet. Click Start microphone first.")
            raise SessionError("The bot isn't in the call yet. Wait until it shows as in the call, then click Begin.")
        if not runner.begun:
            self._spawn(runner.begin())

    async def end(self, session_id: str) -> None:
        record = self.record(session_id)
        if record is None:
            raise SessionError("There's no such session.")
        runner = self._runners.get(session_id)
        if runner is not None and not runner.finished.is_set():
            await runner.end("ended_by_owner")
        if record.kind in BOT_KINDS:
            await self._leave_bot(record, drain=True)
        self._follow_up_finished()

    # ----- background passes ---------------------------------------------------------

    async def tick_all(self) -> None:
        for session_id, runner in list(self._runners.items()):
            if runner.finished.is_set():
                continue
            if session_id in self._sockets:
                await runner.tick()
                if runner.begun and runner.record.status in ("created", "waiting_for_bot"):
                    runner.record.status = "in_progress"
                    self.store.save(runner.record)
            detached_at = self._detached_at.get(session_id)
            if detached_at is not None and self._clock() - detached_at >= self.GRACE_S:
                del self._detached_at[session_id]
                if not runner.begun and runner.record.kind == "practice":
                    self._discard(session_id)
                else:
                    await runner.end("disconnected")
        self._follow_up_finished()

    async def poll_bots(self) -> None:
        if self.attendee is None:
            return
        for session_id, runner in list(self._runners.items()):
            record = runner.record
            if runner.finished.is_set() or not record.bot_id:
                continue
            try:
                state = (await self.attendee.get_bot(record.bot_id)).state or record.bot_state or ""
            except AttendeeError as exc:
                log.warning("couldn't get the status of bot %s: %s", record.bot_id, exc)
                state = record.bot_state or ""
            now = self._clock()
            if state != record.bot_state:
                record.bot_state = state
                self.store.save(record)
            if state in BOT_GONE_STATES:
                if runner.begun:
                    await runner.end("bot_left")
                else:
                    self._drop(session_id)
                    self._fail(record, f"The bot left the meeting before the interview began. "
                                       f"Last status: {status_label(state)}")
            elif join_timed_out(self._bot_started_at.get(session_id, now), now, state, JOIN_TIMEOUT_S):
                await self._leave_bot(record)
                self._drop(session_id)
                self._fail(record, f"The bot didn't get into the call within 3 minutes. "
                                   f"Last status: {status_label(state)}")
            elif state.startswith("joined_"):
                joined_at = self._joined_at.setdefault(session_id, now)
                if (session_id not in self._ever_connected and now - joined_at >= AUDIO_WAIT_S
                        and not record.error):
                    record.error = NO_AUDIO
                    self.store.save(record)
        self._follow_up_finished()

    async def start_background(self) -> None:
        """Recover from a restart, clean up old sessions, load Whisper, then start the periodic loops."""
        self._recover()
        await self._cleanup()
        warm_up = getattr(self.ai.stt, "warm_up", None)
        if warm_up is not None:
            self._spawn(warm_up())  # load the model now rather than during the first answer
        self._loops = [
            asyncio.create_task(self._every(self.TICK_S, self.tick_all), name="tick-runners"),
            asyncio.create_task(self._every(self.POLL_S, self.poll_bots), name="poll-bots"),
            asyncio.create_task(self._every(self.CLEANUP_S, self._cleanup, first_wait_s=self.CLEANUP_S),
                                name="cleanup"),
        ]

    async def stop_background(self) -> None:
        tasks = [*self._loops, *self._tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._loops = []

    async def idle(self) -> None:
        """Wait until runners, cache warming, follow-ups and scoring have settled (used by tests)."""
        while True:
            for runner in list(self._runners.values()):
                await runner.idle()
            self._follow_up_finished()
            pending = [task for task in self._tasks if not task.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def _every(self, interval_s: float, job: Callable[[], Awaitable[None]], first_wait_s: float = 0.0) -> None:
        # Wall-clock scheduling: these loops always use asyncio.sleep, not the injected sleep.
        if first_wait_s:
            await asyncio.sleep(first_wait_s)
        while True:
            try:
                await job()
            except Exception:
                log.exception("background job %s failed", getattr(job, "__name__", job))
            await asyncio.sleep(interval_s)

    async def _cleanup(self) -> None:
        deleted = self.store.cleanup(self.settings.retention_days)
        if deleted:
            log.info("deleted %d sessions older than %d days", len(deleted), self.settings.retention_days)

    def _recover(self) -> None:
        for record in self.store.list():
            if record.id in self._runners:
                continue
            if record.status in LIVE_STATUSES:
                if record.kind == "practice" and record.status == "created":
                    shutil.rmtree(self.store.dir(record.id), ignore_errors=True)
                    continue
                self._fail(record, RESTARTED)
                if record.kind in BOT_KINDS and record.bot_id:
                    self._spawn(self._leave_bot(record))
            elif record.status == "scoring":
                self._spawn(self._rescore(record))

    # ----- after a session ends ------------------------------------------------------

    def _follow_up_finished(self) -> None:
        for session_id, runner in list(self._runners.items()):
            if runner.finished.is_set() and session_id not in self._handled:
                self._handled.add(session_id)
                self._spawn(self._after_finish(runner))

    async def _after_finish(self, runner: SessionRunner) -> None:
        record = runner.record
        if record.kind in BOT_KINDS:
            await self._leave_bot(record, drain=True)
        await runner.idle()
        self._runners.pop(record.id, None)
        self._detached_at.pop(record.id, None)
        if not runner.begun:
            if record.kind == "practice":
                self._discard(record.id)
                return
            record.status, record.error = "ended", NOT_SCORED
            self.store.save(record)
            return
        if record.kind == "check":
            record.status = "done"
            self.store.save(record)
            return
        kit = getattr(runner.engine, "kit", None) or self.kits.get(record.kit_id)
        await self._score(record, runner.transcript, kit, dict(runner.engine.evidence))

    async def _rescore(self, record: SessionRecord) -> None:
        path = self.store.dir(record.id) / "transcript.json"
        try:
            transcript = Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            self._fail(record, f"{RESTARTED} No transcript was saved, so it can't be scored ({exc}).")
            return
        await self._score(record, transcript, self.kits.get(record.kit_id), {})

    def _live_besides(self, session_id: str) -> bool:
        return any(r.begun and not r.finished.is_set() for sid, r in self._runners.items() if sid != session_id)

    async def _score(self, record: SessionRecord, transcript: Transcript, kit: Kit | None,
                     evidence: dict[str, list[str]]) -> None:
        while self._live_besides(record.id):
            await self._sleep(self.SCORING_WAIT_S)
        if kit is None:
            self._fail(record, f"Couldn't score: the kit {record.kit_id!r} isn't available any more.")
            return
        try:
            card = await evaluate_interview(transcript, kit, list(kit.interview_topics), evidence, self.ai.llm,
                                            record.id, record.candidate_name, record.flags, record.end_reason)
            folder = self.store.dir(record.id)
            write_atomic(folder / "scorecard.json", json.dumps(card.to_dict(), indent=2))
            write_atomic(folder / "scorecard.md", card.to_markdown())
        except Exception as exc:
            log.exception("scoring session %s failed", record.id)
            self._fail(record, f"Scoring failed: {exc}")
            return
        record.status, record.error = "done", None
        if self.settings.teams_webhook_url and record.kind == "interview":
            try:
                await post_scorecard(self.settings.teams_webhook_url, card.to_adaptive_card(), self._http)
            except Exception as exc:
                log.warning("posting the scorecard to Teams failed: %s", exc)
                record.error = f"The scorecard is saved here, but posting it to Teams failed: {exc}"
        self.store.save(record)

    async def _leave_bot(self, record: SessionRecord, drain: bool = False) -> None:
        if self.attendee is None or not record.bot_id or record.id in self._left:
            return
        if (record.bot_state or "") in BOT_GONE_STATES:
            return
        self._left.add(record.id)
        if drain and record.id in self._sockets:
            await self._sleep(LEAVE_DELAY_S)  # let the end of the closing line play
        try:
            await self.attendee.leave(record.bot_id)
        except AttendeeError as exc:
            log.warning("couldn't make bot %s leave: %s", record.bot_id, exc)

    # ----- bookkeeping -------------------------------------------------------------------

    def _fail(self, record: SessionRecord, message: str) -> None:
        record.status, record.error = "failed", message
        self.store.save(record)

    def _drop(self, session_id: str) -> None:
        self._runners.pop(session_id, None)
        self._detached_at.pop(session_id, None)
        self._handled.add(session_id)

    def _discard(self, session_id: str) -> None:
        self._drop(session_id)
        shutil.rmtree(self.store.dir(session_id), ignore_errors=True)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ----- what the pages show -------------------------------------------------------------

    def snapshot(self, session_id: str) -> dict | None:
        runner = self._runners.get(session_id)
        record = runner.record if runner is not None else self.store.get(session_id)
        if record is None:
            return None
        folder = self.store.dir(record.id)
        if runner is not None:
            turns = runner.transcript.turns
        else:
            turns = _load_json(folder / "transcript.json", Transcript.from_dict, Transcript()).turns
        live = runner is not None and not runner.finished.is_set()
        return {
            "id": record.id,
            "kind": record.kind,
            "candidate_name": record.candidate_name,
            "kit_id": record.kit_id,
            "created_at": record.created_at,
            "status": record.status,
            "status_label": STATUS_LABELS.get(record.status, record.status),
            "bot_state": record.bot_state,
            "bot_label": status_label(record.bot_state) if record.bot_state else "",
            "error": record.error,
            "end_reason": record.end_reason,
            "flags": list(record.flags),
            "live": live,
            "connected": session_id in self._sockets,
            "begun": bool(runner and runner.begun),
            "speaking": runner.speaking_text if live else None,
            "transcript": [{"speaker": t.speaker, "text": t.text, "at": format_ts(t.start_s), "kind": t.kind,
                            "interrupted": t.interrupted} for t in turns if t.text],
            "scorecard": _load_json(folder / "scorecard.json", lambda data: data, None),
            "downloads": {"transcript": (folder / "transcript.txt").exists(),
                          "scorecard": (folder / "scorecard.md").exists()},
        }


def _load_json(path, convert, default):
    try:
        return convert(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return default
