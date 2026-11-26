"""Day-one check: the echo bot, the self-hearing warning and the speed test."""

import time

import pytest

from interviewer.ai.base import LLMError
from interviewer.ai.factory import AIBundle
from interviewer.daycheck import (
    ECHO_GREETING,
    SELF_HEARING,
    BenchReport,
    BenchResult,
    EchoEngine,
    assess_echo,
    main,
    run_bench,
)
from interviewer.engine.engine import End, Say, Stage
from interviewer.session.transcript import Transcript, Turn
from tests.helpers import TEAMS_LINK, LoudnessVad, collector, make_manager, speak

ANSWERED = {"label": "answered", "evidence_note": "", "follow_up_index": None, "custom_follow_up": None}


class StepTimer:
    """A clock that moves only when a fake step says how long it took."""

    def __init__(self, steps):
        self.steps, self.now = steps, 0.0

    def __call__(self):
        return self.now

    def step(self, key):
        self.now += self.steps[key]


class TimedTTS:
    sample_rate = 24000
    voice_id = "timed"

    def __init__(self, timer):
        self.timer, self.texts = timer, []

    async def synthesize(self, text):
        self.timer.step("tts")
        self.texts.append(text)
        return b"\x00\x00" * 24000 * 5


class TimedSTT:
    def __init__(self, timer):
        self.timer, self.seconds = timer, []

    async def transcribe(self, pcm16, sample_rate, prompt=""):
        self.timer.step("stt")
        self.seconds.append(len(pcm16) / 2 / sample_rate)
        return "a sample answer"


class TimedFakeLLM:
    def __init__(self, model, timer, error=None):
        self.model, self.timer, self.error, self.calls = model, timer, error, 0

    async def complete_json(self, system, user, schema, timeout_s):
        self.calls += 1
        if self.error is not None:
            raise self.error
        self.timer.step(f"llm:{self.model}")
        return dict(ANSWERED)


def fake_bundle(timer):
    return AIBundle(llm=None, stt=TimedSTT(timer), tts=TimedTTS(timer), new_vad=LoudnessVad)


async def test_echo_engine_repeats_text():
    assert await EchoEngine().on_candidate_turn("testing one two") == [Say("I heard: testing one two", "echo")]


async def test_echo_engine_contract():
    engine = EchoEngine()
    assert engine.start() == [Say(ECHO_GREETING, "script")]
    assert await engine.on_candidate_turn("um") == []
    assert engine.on_silence(30) == []
    assert engine.end_now("ended_by_owner") == [Say("Check finished.", "script"), End("ended_by_owner")]
    assert engine.stage == Stage.INTRO and engine.asks_consent is False
    assert ECHO_GREETING in engine.planned_lines()


def test_assess_echo_detects_self_hearing():
    heard_itself = Transcript([Turn("candidate", "hello this is the AI interviewer check say a sentence", 0, 3)])
    assert assess_echo(heard_itself) == [SELF_HEARING] == ["The bot seems to hear its own voice."]
    assert assess_echo(Transcript([Turn("candidate", "testing one two three", 0, 2)])) == []
    assert assess_echo(Transcript([Turn("ai", ECHO_GREETING, 0, 5, kind="script")])) == []


async def test_run_bench_pass_fail(kit):
    timer = StepTimer({"tts": 0.8, "stt": 1.5, "llm:fast": 5.0, "llm:slow": 9.0})
    bundle = fake_bundle(timer)
    report = await run_bench(bundle, ["fast", "slow"], lambda m: TimedFakeLLM(m, timer), kit, timer=timer)
    assert [r.model for r in report.results] == ["fast", "slow"]
    assert [r.ok for r in report.results] == [True, False]
    fast = report.results[0]
    assert (fast.tts_s, fast.stt_s, fast.llm_s, fast.reply_s) == pytest.approx((0.8, 1.5, 5.0, 6.5))
    assert bundle.tts.texts[-1] == kit.topic(kit.interview_topics[0]).questions[0].ask
    assert bundle.stt.seconds[-1] == pytest.approx(3.0)
    text = report.to_text()
    assert "PASS" in text and "FAIL" in text and "8.0 s" in text


async def test_bench_warms_each_model_before_timing_it(kit):
    timer = StepTimer({"tts": 0.1, "stt": 0.1, "llm:m": 2.0})
    llm = TimedFakeLLM("m", timer)
    report = await run_bench(fake_bundle(timer), ["m"], lambda m: llm, kit, timer=timer)
    assert llm.calls == 2 and report.results[0].llm_s == pytest.approx(2.0)


async def test_bench_reports_model_not_pulled(kit):
    timer = StepTimer({"tts": 0.1, "stt": 0.1})
    missing = TimedFakeLLM("x", timer, error=LLMError("Ollama returned 404: {\"error\":\"model 'x' not found\"}"))
    report = await run_bench(fake_bundle(timer), ["x"], lambda m: missing, kit, timer=timer)
    result = report.results[0]
    assert result.error == "model not pulled" and result.ok is False and result.llm_s is None
    assert "model not pulled" in report.to_text()


def test_report_text_lists_every_model():
    report = BenchReport([BenchResult("qwen3:4b-instruct", 0.7, 1.2, 4.1, 5.3, True, None),
                          BenchResult("gemma3:4b", 0.7, 1.2, None, None, False, "model not pulled")])
    lines = report.to_text().splitlines()
    assert any("qwen3:4b-instruct" in line and "PASS" in line for line in lines)
    assert any("gemma3:4b" in line and "FAIL" in line for line in lines)


def test_main_rejects_unknown_commands(capsys):
    with pytest.raises(SystemExit):
        main(["dance"])


async def test_check_session_uses_echo_engine(manager, fake_attendee):
    record = await manager.create_check(TEAMS_LINK)
    assert record.kind == "check" and record.status == "waiting_for_bot"
    runner = manager.runner(record.id)
    assert isinstance(runner.engine, EchoEngine)
    assert fake_attendee.created[0]["meeting_url"] == TEAMS_LINK
    await manager.on_ws_connect(record.id, record.token, collector([]))
    await speak(runner)  # "hello" plays the greeting
    await manager.end(record.id)
    await manager.idle()
    assert manager.store.get(record.id).status == "done"
    assert manager.ai.llm.calls == []  # no evaluator
    assert not (manager.store.dir(record.id) / "scorecard.json").exists()
    assert (manager.store.dir(record.id) / "transcript.txt").exists()
