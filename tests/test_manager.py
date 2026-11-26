"""SessionManager: runners, bots, reconnects, grace periods and scoring."""

import asyncio
import json

import httpx
import pytest

from interviewer.ai.fakes import FakeLLM, FakeSTT
from interviewer.attendee import AttendeeError, MeetingLinkError
from interviewer.engine.engine import Stage
from interviewer.session.manager import SessionError, SessionManager
from interviewer.session.store import SessionStore
from interviewer.session.transcript import Transcript, Turn
from tests.helpers import TEAMS_LINK, collector, feed, make_manager, make_settings, silence, speak, tone

ANSWERED = {"label": "answered", "evidence_note": "Gave a concrete example.", "follow_up_index": None,
            "custom_follow_up": None}
WEBHOOK = "https://example.webhook.office.com/hook"


def eval_reply(quote: str) -> dict:
    return {"score": 3, "not_enough_signal": False, "quotes": [{"text": quote, "at": "00:00"}],
            "strengths": ["Clear and concrete."], "concerns": []}


class PromptSTT:
    def __init__(self, texts):
        self.texts, self.prompts = list(texts), []

    async def transcribe(self, pcm16, sample_rate, prompt=""):
        self.prompts.append(prompt)
        return self.texts.pop(0) if self.texts else ""


async def start_practice(manager, sent=None):
    record = await manager.create_practice("senior-dotnet")
    runner = await manager.on_ws_connect(record.id, record.token, collector([] if sent is None else sent))
    return record, runner


def ai_kinds(runner):
    return [t.kind for t in runner.transcript.turns if t.speaker == "ai"]


# ----- creating sessions ------------------------------------------------------


async def test_create_interview_sends_the_bot_with_a_tokenised_wss_url(manager, fake_attendee):
    record = await manager.create_interview("senior-dotnet", "Asha", f"  <{TEAMS_LINK}>  ")
    assert record.status == "waiting_for_bot" and record.bot_id == "bot_1" and record.meeting_url == TEAMS_LINK
    assert fake_attendee.created == [{
        "meeting_url": TEAMS_LINK,
        "bot_name": "AI Interviewer",
        "audio_ws_url": f"wss://interviewer:8443/ws/audio/{record.id}?token={record.token}",
    }]
    assert manager.runner(record.id) is not None
    assert manager.store.get(record.id).status == "waiting_for_bot"


async def test_create_interview_warms_the_speech_cache(manager):
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    await manager.idle()
    assert manager.runner(record.id).engine.kit.intro_script in manager.ai.tts.texts


async def test_resume_notes_become_experience_questions(tmp_path, clock, fake_attendee):
    custom = ["Tell me about the payments service on your resume.", "How did you run the move to .NET 8?"]
    manager = make_manager(tmp_path, clock, attendee=fake_attendee, llm=FakeLLM([{"questions": custom}]))
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK, notes="Built payments at Contoso.")
    lines = manager.runner(record.id).engine.planned_lines()
    assert custom[0] in lines and custom[1] in lines


async def test_bad_input_is_rejected_before_anything_is_created(manager, fake_attendee):
    with pytest.raises(SessionError, match="kit"):
        await manager.create_interview("no-such-kit", "Asha", TEAMS_LINK)
    with pytest.raises(SessionError, match="name"):
        await manager.create_interview("senior-dotnet", "   ", TEAMS_LINK)
    with pytest.raises(MeetingLinkError):
        await manager.create_interview("senior-dotnet", "Asha", "https://zoom.us/j/123")
    assert manager.store.list() == [] and fake_attendee.created == []


async def test_create_interview_without_attendee_explains_the_setup(tmp_path, clock):
    manager = make_manager(tmp_path, clock, attendee=None)
    with pytest.raises(SessionError, match="ATTENDEE_API_KEY"):
        await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    assert manager.store.list() == []


async def test_bot_creation_failure_marks_the_session_failed(manager, fake_attendee):
    fake_attendee.create_error = AttendeeError(None, "can't reach Attendee at http://attendee-app:8000: refused")
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    assert record.status == "failed" and "can't reach Attendee" in record.error
    assert manager.store.get(record.id).status == "failed"
    assert manager.runner(record.id) is None


async def test_stt_prompt_is_the_kit_vocabulary(tmp_path, clock):
    stt = PromptSTT(["Yes, that's fine."])
    manager = make_manager(tmp_path, clock, stt=stt)
    _, runner = await start_practice(manager)
    await speak(runner)  # "hello" starts the intro and isn't transcribed
    await speak(runner)
    assert stt.prompts and "ASP.NET Core" in stt.prompts[0]


@pytest.mark.parametrize(("setting_s", "begins"), [(1.0, True), (1.5, False)])
async def test_end_of_turn_silence_setting_reaches_the_runner(tmp_path, clock, setting_s, begins):
    manager = make_manager(tmp_path, clock, settings=make_settings(tmp_path, end_of_turn_silence_s=setting_s))
    _, runner = await start_practice(manager)
    await feed(runner, tone(2.5) + silence(1.2))
    assert runner.begun is begins


# ----- websocket connections ------------------------------------------------------


async def test_wrong_token_or_session_is_refused(manager):
    record = await manager.create_practice("senior-dotnet")
    with pytest.raises(PermissionError):
        await manager.on_ws_connect(record.id, "wrong", collector([]))
    with pytest.raises(PermissionError):
        await manager.on_ws_connect("ffffffffffff", record.token, collector([]))


async def test_reconnect_resumes_same_runner(tmp_path, clock):
    manager = make_manager(tmp_path, clock, stt=FakeSTT(["Yes, that's fine."]))
    first, second = [], []
    record, runner = await start_practice(manager, first)
    await speak(runner)
    assert first and runner.begun
    await manager.on_ws_disconnect(record.id)
    again = await manager.on_ws_connect(record.id, record.token, collector(second))
    assert again is runner and runner.engine.stage == Stage.INTRO
    await speak(again)
    assert second
    assert ai_kinds(runner) == ["script", "question"]  # one intro, then the warm-up question


async def test_stale_disconnect_does_not_detach_the_new_socket(tmp_path, clock):
    manager = make_manager(tmp_path, clock)
    old, new = [], []
    record = await manager.create_practice("senior-dotnet")
    old_send, new_send = collector(old), collector(new)
    runner = await manager.on_ws_connect(record.id, record.token, old_send)
    await manager.on_ws_connect(record.id, record.token, new_send)  # the reconnect arrives first
    await manager.on_ws_disconnect(record.id, old_send)  # then the old socket's close
    await speak(runner)
    assert new and not old


async def test_detached_runner_gets_no_nudges(manager, clock):
    record, runner = await start_practice(manager)
    await speak(runner)
    await manager.on_ws_disconnect(record.id)
    for _ in range(3):
        clock.now += 60
        await manager.tick_all()
        await manager.idle()
    assert ai_kinds(runner) == ["script"]


async def test_no_reconnect_finishes_after_grace(manager, clock):
    record, runner = await start_practice(manager)
    await speak(runner)
    await manager.on_ws_disconnect(record.id)
    clock.now += SessionManager.GRACE_S - 1
    await manager.tick_all()
    assert not runner.finished.is_set()
    clock.now += 2
    await manager.tick_all()
    assert runner.end_reason == "disconnected"
    assert manager.store.get(record.id).status == "scoring"
    await manager.idle()
    assert manager.store.get(record.id).status == "done"


async def test_unused_practice_session_is_discarded_after_grace(manager, clock):
    record = await manager.create_practice("senior-dotnet")
    clock.now += SessionManager.GRACE_S + 1
    await manager.tick_all()
    assert manager.runner(record.id) is None and manager.store.get(record.id) is None
    with pytest.raises(PermissionError):
        await manager.on_ws_connect(record.id, record.token, collector([]))


async def test_status_moves_to_in_progress_when_the_interview_begins(manager):
    record, runner = await start_practice(manager)
    assert record.status == "created"
    await speak(runner)
    await manager.tick_all()
    assert manager.store.get(record.id).status == "in_progress"


# ----- the bot --------------------------------------------------------------------------


async def test_bot_join_timeout_sets_failed(manager, clock, fake_attendee):
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    fake_attendee.state = "waiting_room"
    clock.now += 60
    await manager.poll_bots()
    stored = manager.store.get(record.id)
    assert stored.status == "waiting_for_bot" and stored.bot_state == "waiting_room"
    clock.now += 121
    await manager.poll_bots()
    stored = manager.store.get(record.id)
    assert stored.status == "failed" and "3 minutes" in stored.error and "lobby" in stored.error
    assert fake_attendee.left == ["bot_1"]  # don't leave it waiting in the lobby
    assert manager.runner(record.id) is None


async def test_bot_failing_before_the_interview_marks_failed(manager, fake_attendee):
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    fake_attendee.state = "fatal_error"
    await manager.poll_bots()
    stored = manager.store.get(record.id)
    assert stored.status == "failed" and "Failed to join" in stored.error
    assert manager.runner(record.id) is None


async def test_bot_leaving_the_call_ends_the_session(manager, fake_attendee):
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    runner = await manager.on_ws_connect(record.id, record.token, collector([]))
    fake_attendee.state = "joined_recording"
    await manager.poll_bots()
    await speak(runner)
    fake_attendee.state = "ended"
    await manager.poll_bots()
    assert runner.end_reason == "bot_left"
    await manager.idle()
    assert manager.store.get(record.id).status == "done"
    assert fake_attendee.left == []  # it had already gone


async def test_natural_end_makes_the_bot_leave(tmp_path, clock, fake_attendee):
    stt = FakeSTT(["No, I'm not comfortable with being recorded."])
    manager = make_manager(tmp_path, clock, attendee=fake_attendee, stt=stt)
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    runner = await manager.on_ws_connect(record.id, record.token, collector([]))
    await speak(runner)
    await speak(runner)
    assert runner.end_reason == "consent_declined"
    await manager.idle()
    assert fake_attendee.left == ["bot_1"]


async def test_begin_needs_the_bot_in_the_call(manager):
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    with pytest.raises(SessionError, match="isn't in the call yet"):
        await manager.begin(record.id)
    sent = []
    runner = await manager.on_ws_connect(record.id, record.token, collector(sent))
    await manager.begin(record.id)
    await manager.idle()
    assert runner.begun and sent


async def test_end_before_the_interview_began_is_not_scored(manager, fake_attendee):
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    await manager.end(record.id)
    await manager.idle()
    stored = manager.store.get(record.id)
    assert stored.status == "ended" and stored.end_reason == "ended_by_owner"
    assert fake_attendee.left == ["bot_1"]
    assert not (manager.store.dir(record.id) / "scorecard.json").exists()


# ----- scoring ---------------------------------------------------------------------------


async def test_full_scripted_interview_produces_transcript_and_scorecard(tmp_path, clock):
    answers = [
        "Yes, that's fine.",
        "I'm a backend developer and I've been building .NET services for eight years.",
        "I led the rewrite of our order service from .NET Framework to .NET 8.",
        "The hardest part was moving the shared database without downtime, so we used the strangler pattern.",
        "Value types are copied and live inline, reference types are shared on the heap, and boxing allocates.",
        "I return problem details for errors and version the API in the URL so clients don't break.",
        "I would split reads and writes, put a queue in front of the slow parts, and cache hot product data.",
        "No, that's all, thanks.",
    ]
    replies = [ANSWERED] * 6 + [eval_reply("boxing allocates"), eval_reply("version the API in the URL"),
                                eval_reply("put a queue in front of the slow parts")]
    manager = make_manager(tmp_path, clock, stt=FakeSTT(answers), llm=FakeLLM(replies))
    record, runner = await start_practice(manager)
    await speak(runner)  # "hello" starts the intro
    for _ in answers:
        await speak(runner)
    assert runner.end_reason == "completed"
    await manager.idle()
    folder = manager.store.dir(record.id)
    assert (folder / "transcript.txt").exists() and (folder / "scorecard.md").exists()
    card = json.loads((folder / "scorecard.json").read_text())
    assert [t["topic_id"] for t in card["topics"]] == ["csharp_runtime", "aspnet_api", "system_design"]
    assert [t["score"] for t in card["topics"]] == [3, 3, 3]
    assert manager.store.get(record.id).status == "done"


def _webhook_manager(tmp_path, clock, fake_attendee, status_code):
    posted = []

    def handler(request):
        posted.append(json.loads(request.content))
        return httpx.Response(status_code)

    manager = make_manager(tmp_path, clock, attendee=fake_attendee,
                           http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                           settings=make_settings(tmp_path, teams_webhook_url=WEBHOOK))
    return manager, posted


async def _finished_interview(manager):
    record = await manager.create_interview("senior-dotnet", "Asha", TEAMS_LINK)
    runner = await manager.on_ws_connect(record.id, record.token, collector([]))
    await speak(runner)
    await manager.end(record.id)
    return record


async def test_webhook_gets_the_card_for_interviews_only(tmp_path, clock, fake_attendee):
    manager, posted = _webhook_manager(tmp_path, clock, fake_attendee, 202)
    interview = await _finished_interview(manager)
    practice, runner = await start_practice(manager)
    await speak(runner)
    await manager.end(practice.id)
    await manager.idle()
    assert len(posted) == 1 and "Asha" in json.dumps(posted[0])
    assert manager.store.get(interview.id).status == "done"
    assert manager.store.get(practice.id).status == "done"


async def test_webhook_failure_keeps_the_scorecard(tmp_path, clock, fake_attendee):
    manager, posted = _webhook_manager(tmp_path, clock, fake_attendee, 500)
    record = await _finished_interview(manager)
    await manager.idle()
    stored = manager.store.get(record.id)
    assert posted and stored.status == "done" and "Teams" in stored.error
    assert (manager.store.dir(record.id) / "scorecard.md").exists()


async def test_scoring_waits_while_another_interview_is_live(manager):
    first, first_runner = await start_practice(manager)
    await speak(first_runner)
    second, second_runner = await start_practice(manager)
    await speak(second_runner)
    await manager.end(first.id)
    for _ in range(20):
        await asyncio.sleep(0)
    assert manager.store.get(first.id).status == "scoring"
    await manager.end(second.id)
    await manager.idle()
    assert manager.store.get(first.id).status == "done" and manager.store.get(second.id).status == "done"


# ----- starting up ---------------------------------------------------------------------------


async def test_restart_marks_running_sessions_failed_and_rescores(tmp_path, clock):
    settings = make_settings(tmp_path)
    store = SessionStore(settings.data_dir)
    running = store.create("interview", "Asha", "senior-dotnet", TEAMS_LINK)
    running.status = "in_progress"
    store.save(running)
    scoring = store.create("practice", "Practice", "senior-dotnet")
    scoring.status = "scoring"
    store.save(scoring)
    Transcript([Turn("candidate", "Boxing allocates on the heap.", 1.0, 3.0, topic_id="csharp_runtime")]).save(
        store.dir(scoring.id))
    manager = make_manager(tmp_path, clock, llm=FakeLLM([eval_reply("Boxing allocates")]), settings=settings)
    await manager.start_background()
    await manager.idle()
    await manager.stop_background()
    assert store.get(running.id).status == "failed" and "restarted" in store.get(running.id).error
    card = json.loads((store.dir(scoring.id) / "scorecard.json").read_text())
    assert card["topics"][0]["score"] == 3
    assert store.get(scoring.id).status == "done"


async def test_start_background_runs_cleanup(manager):
    old = manager.store.create("practice", "Old", "senior-dotnet")
    old.created_at, old.status = "2020-01-01T00:00:00+00:00", "done"
    manager.store.save(old)
    await manager.start_background()
    await manager.stop_background()
    assert manager.store.get(old.id) is None
