"""Day-one check: is this PC and this Teams setup good enough before any real use?

Two parts:
- An echo bot (EchoEngine) joins a Teams meeting, says a greeting and repeats
  what it hears, so the owner can confirm the bot gets admitted, he hears it,
  his speech turns into text, and the bot doesn't hear its own voice.
- A speed test (run_bench) times text-to-speech, the last bit of
  speech-to-text and one interview decision per Ollama model, against the
  8-second reply target.

Command line: ``python -m interviewer.daycheck bench --models qwen3:4b-instruct,llama3.2:3b,gemma3:4b``
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from interviewer.ai.base import LanguageModel, LLMError
from interviewer.ai.factory import AIBundle
from interviewer.audio.pcm import float32_to_pcm16, pcm16_to_float32, resample_linear
from interviewer.engine.engine import Action, End, Say, Stage
from interviewer.engine.guardrails import is_meaningful
from interviewer.engine.prompts import DECISION_SCHEMA, build_decision_prompt, build_system_prompt
from interviewer.kit import Kit
from interviewer.session.transcript import Transcript

ECHO_GREETING = "Hello, this is the AI interviewer check. Say a sentence, and I'll repeat what I heard."
CHECK_FINISHED = "Check finished."
SELF_HEARING = "The bot seems to hear its own voice."
SELF_HEARING_SHARE = 0.6

REPLY_TARGET_S = 8.0
STT_TAIL_S = 3.0
NOT_PULLED = "model not pulled"
# About 15 seconds of speech: a realistic answer, of which only the last few seconds are
# still waiting to be transcribed when the candidate stops (the rest was done while they talked).
SAMPLE_ANSWER = (
    "In my last project I moved our order service from the old dot net framework to dot net eight. "
    "We ran both versions side by side behind a feature flag, compared their responses for two weeks, "
    "and then switched traffic over gradually, which kept the risk low and let us roll back quickly."
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", text.replace("’", "'").lower()))


class EchoEngine:
    """Repeats what it hears. The stage stays INTRO, so the runner plays no acknowledgments."""

    asks_consent = False

    def __init__(self) -> None:
        self.stage = Stage.INTRO
        self.flags: set[str] = set()
        self.evidence: dict[str, list[str]] = {}
        self.events: list[str] = []
        self.current_topic_id = None

    def start(self) -> list[Say]:
        return [Say(ECHO_GREETING, "script")]

    async def on_candidate_turn(self, text: str) -> list[Action]:
        if not is_meaningful(text):
            return []
        return [Say(f"I heard: {text.strip()}", "echo")]

    def on_silence(self, seconds: float) -> list[Action]:
        return []

    def end_now(self, reason: str) -> list[Action]:
        return [Say(CHECK_FINISHED, "script"), End(reason)]

    def planned_lines(self) -> list[str]:
        return [ECHO_GREETING, CHECK_FINISHED]


def assess_echo(transcript: Transcript) -> list[str]:
    """Warnings for the check page: a 'candidate' turn that repeats most of the greeting is the bot itself."""
    greeting = _words(ECHO_GREETING)
    for turn in transcript.candidate_turns():
        if len(greeting & _words(turn.text)) / len(greeting) >= SELF_HEARING_SHARE:
            return [SELF_HEARING]
    return []


@dataclass
class BenchResult:
    model: str
    tts_s: float
    stt_s: float
    llm_s: float | None
    reply_s: float | None
    ok: bool
    error: str | None = None


@dataclass
class BenchReport:
    results: list[BenchResult]
    target_s: float = REPLY_TARGET_S

    def to_text(self) -> str:
        lines = [f"Speed test. Target: the next line within {self.target_s:.1f} s of the candidate stopping."]
        if self.results:
            first = self.results[0]
            lines += [
                f"Text-to-speech, one interview question: {first.tts_s:.2f} s "
                "(kit lines are cached, so this only matters for custom follow-ups)",
                f"Speech-to-text, last {STT_TAIL_S:.0f} s of an answer: {first.stt_s:.2f} s",
            ]
        lines += ["", f"{'Model':<28}{'LLM turn':>10}{'Reply':>10}  Result"]
        for r in self.results:
            llm = f"{r.llm_s:.2f} s" if r.llm_s is not None else "—"
            reply = f"{r.reply_s:.2f} s" if r.reply_s is not None else "—"
            verdict = "PASS" if r.ok else "FAIL"
            if r.error:
                verdict += f" ({r.error})"
            lines.append(f"{r.model:<28}{llm:>10}{reply:>10}  {verdict}")
        lines += ["", "Reply = speech-to-text + LLM turn. Pick the fastest model that passes and set OLLAMA_MODEL in .env."]
        return "\n".join(lines)


async def _maybe_warm(part) -> None:
    warm_up = getattr(part, "warm_up", None)
    if warm_up is not None:
        await warm_up()


async def run_bench(ai: AIBundle, models: list[str], llm_factory: Callable[[str], LanguageModel], kit: Kit,
                    timer: Callable[[], float] = time.perf_counter, llm_timeout_s: float = 20.0) -> BenchReport:
    topic = kit.topic(kit.interview_topics[0])
    question = topic.questions[0]

    # Load the speech models first, so their start-up isn't counted.
    await _maybe_warm(ai.tts)
    await _maybe_warm(ai.stt)
    sample = await ai.tts.synthesize(SAMPLE_ANSWER)

    start = timer()
    await ai.tts.synthesize(question.ask)
    tts_s = timer() - start

    pcm = sample
    if ai.tts.sample_rate != 16000:
        pcm = float32_to_pcm16(resample_linear(pcm16_to_float32(sample), ai.tts.sample_rate, 16000))
    tail = pcm[-int(STT_TAIL_S * 16000) * 2:]
    start = timer()
    await ai.stt.transcribe(tail, 16000, ", ".join(kit.vocabulary))
    stt_s = timer() - start

    system = build_system_prompt(kit)
    user = build_decision_prompt(question.ask, question.follow_ups, set(), question.look_for, SAMPLE_ANSWER)
    results = []
    for model in models:
        llm = llm_factory(model)
        try:
            await llm.complete_json(system, user, DECISION_SCHEMA, 120.0)  # loads the model; not timed
            start = timer()
            await llm.complete_json(system, user, DECISION_SCHEMA, llm_timeout_s)
            llm_s = timer() - start
        except LLMError as exc:
            error = NOT_PULLED if "not found" in str(exc).lower() else str(exc)
            results.append(BenchResult(model, tts_s, stt_s, None, None, False, error))
            continue
        reply_s = stt_s + llm_s
        results.append(BenchResult(model, tts_s, stt_s, llm_s, reply_s, reply_s <= REPLY_TARGET_S))
    return BenchReport(results)


def _first_kit(settings) -> Kit:
    from interviewer.kit import list_kits

    kits, errors = list_kits(settings.kits_dir)
    if not kits:
        raise SystemExit(f"No usable kits in {settings.kits_dir}: {errors}")
    return kits[sorted(kits)[0]]


async def _bench_command(models: list[str]) -> int:
    import httpx

    from interviewer.ai.factory import build_ai
    from interviewer.ai.ollama import OllamaLLM
    from interviewer.config import Settings

    settings = Settings.from_env()
    async with httpx.AsyncClient() as http:
        ai = build_ai(settings, http)
        report = await run_bench(ai, models, lambda model: OllamaLLM(settings.ollama_url, model, http),
                                 _first_kit(settings), llm_timeout_s=settings.llm_timeout_s)
    print(report.to_text())
    return 0 if report.results and all(r.ok for r in report.results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m interviewer.daycheck", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("bench", help="time speech and LLM replies on this PC")
    bench.add_argument("--models", default="qwen3:4b-instruct",
                       help="comma-separated Ollama models (default: qwen3:4b-instruct)")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    return asyncio.run(_bench_command(models))


if __name__ == "__main__":
    sys.exit(main())
