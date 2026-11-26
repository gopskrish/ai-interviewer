import asyncio
import base64
import json
import wave

from interviewer.ai.fakes import FakeLLM, FakeSTT, FakeTTS
from interviewer.engine.engine import NUDGE_GENTLE, NUDGE_TAKE_TIME, End, InterviewEngine, Say, Stage
from interviewer.session.runner import SessionRunner
from interviewer.session.speech_cache import SpeechCache
from interviewer.session.store import SessionStore
from tests.helpers import FakeClock, LoudnessVad, silence, tone

INTRO = "Intro line. Is that OK with you?"


class FakeEngine:
    def __init__(self, replies=None, stage=Stage.INTRO, raise_on_turn=None):
        self.stage = stage
        self.flags, self.evidence, self.events = set(), {}, []
        self.current_topic_id = None
        self.replies = list(replies or [])
        self.turns, self.silences = [], []
        self.raise_on_turn = raise_on_turn

    def start(self):
        return [Say(INTRO, "script")]

    async def on_candidate_turn(self, text):
        self.turns.append(text)
        if self.raise_on_turn:
            raise self.raise_on_turn
        return self.replies.pop(0) if self.replies else []

    def on_silence(self, seconds):
        self.silences.append(seconds)
        return [Say({10: NUDGE_GENTLE, 30: NUDGE_TAKE_TIME}.get(seconds, "Moving on."), "nudge")]

    def end_now(self, reason):
        self.stage = Stage.ENDED
        return [Say("Bye.", "script"), End(reason)]

    def planned_lines(self):
        return []


def mixed(pcm, rate=16000):
    return json.dumps({"trigger": "realtime_audio.mixed", "bot_id": "b",
                       "data": {"chunk": base64.b64encode(pcm).decode(), "timestamp_ms": 0, "sample_rate": rate}})


def make_runner(tmp_path, engine, stt=None, tts=None, attach=True):
    clock, sent = FakeClock(), []
    store = SessionStore(tmp_path)
    record = store.create("practice", "Asha", "senior-dotnet")
    cache = SpeechCache(tts or FakeTTS(), tmp_path / "tts")
    runner = SessionRunner(record, engine, stt or FakeSTT([]), cache, LoudnessVad(), store, clock=clock, sleep=clock.sleep)

    async def send(message):
        sent.append(message)

    if attach:
        runner.attach(send)
    return runner, sent, clock, store


async def feed(runner, pcm, chunk_s=0.1, rate=16000, wait=True):
    step = int(rate * chunk_s) * 2
    for i in range(0, len(pcm), step):
        await runner.handle_message(mixed(pcm[i:i + step], rate))
        await asyncio.sleep(0)
    if wait:
        await runner.idle()


def sent_seconds(sent):
    return sum(len(base64.b64decode(m["data"]["chunk"])) / 2 / m["data"]["sample_rate"] for m in sent)


def ai_turns(runner):
    return [t for t in runner.transcript.turns if t.speaker == "ai"]


async def test_idle_returns_when_finished_tasks_are_not_yet_discarded(tmp_path, monkeypatch):
    runner, *_ = make_runner(tmp_path, FakeEngine())
    finished = asyncio.get_running_loop().create_future()
    finished.set_result(None)
    runner._tasks.add(finished)
    calls = 0
    real_gather = asyncio.gather

    def counting_gather(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 50:
            raise RuntimeError("idle() is spinning")
        return real_gather(*args, **kwargs)

    monkeypatch.setattr(asyncio, "gather", counting_gather)
    await runner.idle()


async def test_first_speech_triggers_intro(tmp_path):
    engine = FakeEngine()
    runner, sent, _, _ = make_runner(tmp_path, engine)
    await feed(runner, tone(0.5) + silence(3.1))
    assert runner.begun and engine.turns == []
    assert runner.transcript.turns[0].kind == "script" and runner.transcript.turns[0].text == INTRO
    # Not cached yet, so it's synthesised sentence by sentence ("Intro line." + "Is that OK with you?").
    assert abs(sent_seconds(sent) - (len("Intro line.") + len("Is that OK with you?")) * 0.01) < 0.005


async def test_answer_flow_plays_ack_then_next_line(tmp_path):
    engine = FakeEngine(stage=Stage.WARMUP, replies=[[Say("Next question?", "question")]])
    runner, _, _, _ = make_runner(tmp_path, engine, stt=FakeSTT(["My answer."]))
    await runner.begin()
    await feed(runner, tone(2.0) + silence(1.6))
    assert [(t.speaker, t.kind) for t in runner.transcript.turns] == [
        ("ai", "script"), ("candidate", None), ("ai", "ack"), ("ai", "question"),
    ]
    assert runner.transcript.turns[1].text == "My answer." and engine.turns == ["My answer."]
    assert ai_turns(runner)[1].text == "Thanks."


async def test_barge_in_stops_ai_and_marks_interrupted(tmp_path):
    long_line = ("long " * 200).strip()
    engine = FakeEngine(stage=Stage.WARMUP, replies=[[Say(long_line, "question"), Say("next", "question")]])
    runner, sent, _, _ = make_runner(tmp_path, engine, stt=FakeSTT(["My answer."]))
    await runner.begin()
    await feed(runner, tone(2.0) + silence(1.6), wait=False)
    while runner.speaking_text != long_line:
        await asyncio.sleep(0)
    await feed(runner, tone(0.4), wait=False)
    await runner.idle()
    ai = ai_turns(runner)
    assert ai[-1].text == long_line and ai[-1].interrupted and all(t.text != "next" for t in ai)
    played_before = (len(INTRO) + len("Thanks.")) * 0.01
    assert sent_seconds(sent) - played_before < len(long_line) * 0.01 / 2


async def test_noise_only_turn_does_not_advance(tmp_path, kit):
    engine = InterviewEngine(kit, FakeLLM([]))
    runner, _, _, _ = make_runner(tmp_path, engine, stt=FakeSTT(["Yes, that's fine", "um"]))
    await runner.begin()
    await feed(runner, tone(1.2) + silence(3.2))
    assert engine.stage == Stage.WARMUP
    await feed(runner, tone(0.3) + silence(3.2))
    assert engine.stage == Stage.WARMUP and "ack" not in [t.kind for t in ai_turns(runner)]


async def test_turn_during_intro_is_not_consent(tmp_path, kit):
    engine = InterviewEngine(kit, FakeLLM([]))
    runner, _, _, _ = make_runner(tmp_path, engine, stt=FakeSTT(["Sorry, I can't hear you well"]))
    begin = asyncio.create_task(runner.begin())
    while runner.speaking_text != kit.intro_script:
        await asyncio.sleep(0)
    await feed(runner, tone(1.0) + silence(3.1), wait=False)
    await begin
    await runner.idle()
    assert engine.stage == Stage.INTRO
    assert ai_turns(runner)[0].interrupted is False


async def test_turn_ends_when_audio_stops_arriving(tmp_path):
    engine = FakeEngine(stage=Stage.WARMUP)
    runner, _, clock, _ = make_runner(tmp_path, engine, stt=FakeSTT(["Long answer here."]))
    await runner.begin()
    await feed(runner, tone(2.0))
    assert engine.turns == []
    clock.now += 3.5
    await runner.tick()
    await runner.idle()
    assert engine.turns == ["Long answer here."]


async def test_silence_nudges_after_question(tmp_path):
    engine = FakeEngine(stage=Stage.WARMUP)
    runner, _, clock, _ = make_runner(tmp_path, engine)
    await runner.begin()
    clock.now += 10.5
    await runner.tick()
    await runner.idle()
    await runner.tick()
    await runner.idle()
    assert engine.silences == [10]
    clock.now += 20.0
    await runner.tick()
    await runner.idle()
    assert engine.silences == [10, 30]
    assert [t.text for t in ai_turns(runner)][-2:] == [NUDGE_GENTLE, NUDGE_TAKE_TIME]


async def test_end_saves_transcript_and_sets_scoring(tmp_path):
    engine = FakeEngine(stage=Stage.WARMUP, replies=[[Say("Bye now.", "script"), End("completed")]])
    runner, _, _, store = make_runner(tmp_path, engine, stt=FakeSTT(["That's all."]))
    await runner.begin()
    await feed(runner, tone(2.0) + silence(1.6))
    assert runner.finished.is_set() and runner.end_reason == "completed"
    assert (store.dir(runner.record.id) / "transcript.json").exists()
    assert store.get(runner.record.id).status == "scoring"


async def test_end_called_by_owner_plays_closing_and_finishes(tmp_path):
    runner, _, _, _ = make_runner(tmp_path, FakeEngine(stage=Stage.WARMUP))
    await runner.begin()
    await runner.end("ended_by_owner")
    assert runner.finished.is_set() and runner.end_reason == "ended_by_owner" and ai_turns(runner)[-1].text == "Bye."


async def test_engine_error_keeps_listening(tmp_path):
    engine = FakeEngine(stage=Stage.WARMUP, raise_on_turn=RuntimeError("boom"))
    runner, _, _, _ = make_runner(tmp_path, engine, stt=FakeSTT(["Answer."]))
    await runner.begin()
    await feed(runner, tone(2.0) + silence(1.6))
    assert not runner.finished.is_set() and any("boom" in e for e in engine.events)


async def test_audio_is_dropped_when_detached(tmp_path):
    runner, sent, _, _ = make_runner(tmp_path, FakeEngine(), attach=False)
    await runner.begin()
    assert sent == [] and ai_turns(runner)[0].text == INTRO


async def test_meeting_audio_is_recorded_at_16k(tmp_path):
    runner, _, _, store = make_runner(tmp_path, FakeEngine(stage=Stage.WARMUP))
    await feed(runner, silence(0.5, 48000), rate=48000)
    await runner.end("ended_by_owner")
    with wave.open(str(store.dir(runner.record.id) / "meeting-audio.wav"), "rb") as w:
        assert w.getframerate() == 16000 and abs(w.getnframes() - 8000) <= 16
