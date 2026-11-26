"""The interview itself: a timed sequence of stages run by plain code.

Code owns the stage, the clock and which question comes next. The
language model only makes one narrow decision per answer (was it enough,
which follow-up fits), returned as JSON. If the model is slow or returns
something unusable, a simple rule keeps the interview moving.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from interviewer.ai.base import LanguageModel, LLMError, LLMTimeout
from interviewer.engine.guardrails import (
    consent_declined,
    is_meaningful,
    is_safe_to_say,
    wants_human,
    wants_repeat,
)
from interviewer.engine.prompts import (
    DECISION_SCHEMA,
    EXPERIENCE_SCHEMA,
    FAQ_SCHEMA,
    LABELS,
    build_decision_prompt,
    build_experience_prompt,
    build_faq_prompt,
    build_system_prompt,
)
from interviewer.kit import Kit

NUDGE_GENTLE = "Whenever you're ready."
NUDGE_TAKE_TIME = "Take your time, there's no rush."
MOVE_ON = "No problem, let's move on to the next one."
DIDNT_CATCH = "Sorry, I didn't catch that. Could you say it again?"
CLARIFY_ASSUMPTIONS = "Good question. Answer it the way you'd interpret it, and tell me your assumptions."
STEER_BACK = "Let's bring it back to the question."
CANDIDATE_QUESTIONS_PROMPT = "We're nearly done. What questions do you have for me?"
ANYTHING_ELSE = "Anything else you'd like to ask?"
DEFER_TO_HR = "That's a good one for the HR team. I'll note it, and they'll follow up with you."
CONSENT_DECLINED_LINE = (
    "No problem at all. I'll let the team know, and someone will be in touch about other options. "
    "Thanks for your time."
)
WANTS_HUMAN_LINE = (
    "Of course. I'll let the team know you'd like to speak with a person, and they'll be in touch. "
    "Thanks for your time."
)

FIXED_LINES = [
    NUDGE_GENTLE, NUDGE_TAKE_TIME, MOVE_ON, DIDNT_CATCH, CLARIFY_ASSUMPTIONS, STEER_BACK,
    CANDIDATE_QUESTIONS_PROMPT, ANYTHING_ELSE, DEFER_TO_HR, CONSENT_DECLINED_LINE, WANTS_HUMAN_LINE,
]

MAX_FOLLOW_UPS = 2
FALLBACK_SHORT_ANSWER_WORDS = 12
EXPERIENCE_KEY = "experience"
CANDIDATE_QUESTIONS_KEY = "candidate_questions"

_NO_MORE_QUESTIONS = re.compile(r"\b(no|nope|nothing|that's all|that is all|i'm good|i am good|all good|no more)\b")
_QUESTION_WORDS = {"what", "how", "when", "who", "where", "why", "which", "can", "could", "will", "would",
                   "is", "are", "do", "does"}


class Stage(StrEnum):
    INTRO = "intro"
    WARMUP = "warmup"
    EXPERIENCE = "experience"
    TECHNICAL = "technical"
    CANDIDATE_QUESTIONS = "candidate_questions"
    CLOSE = "close"
    ENDED = "ended"


@dataclass(frozen=True)
class Say:
    text: str
    kind: str
    topic_id: str | None = None


@dataclass(frozen=True)
class End:
    reason: str


Action = Say | End


class EngineLike(Protocol):
    stage: Stage
    flags: set[str]
    evidence: dict[str, list[str]]
    events: list[str]

    @property
    def current_topic_id(self) -> str | None: ...

    def start(self) -> list[Say]: ...

    async def on_candidate_turn(self, text: str) -> list[Action]: ...

    def on_silence(self, seconds: float) -> list[Action]: ...

    def end_now(self, reason: str) -> list[Action]: ...

    def planned_lines(self) -> list[str]: ...


@dataclass
class _Question:
    text: str
    follow_ups: list[str]
    look_for: list[str]
    topic_id: str | None


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.replace("’", "'").lower())


class InterviewEngine:
    def __init__(
        self,
        kit: Kit,
        llm: LanguageModel,
        clock: Callable[[], float] = time.monotonic,
        experience_questions: list[str] | None = None,
        llm_timeout_s: float = 20.0,
    ):
        self.kit = kit
        self._llm = llm
        self._clock = clock
        self._timeout = llm_timeout_s
        self._warmup = list(kit.warmup_questions[:1])
        self._experience = list((experience_questions or kit.experience_questions)[:2])
        self._topics = [kit.topic(topic_id) for topic_id in kit.interview_topics]
        self._system = build_system_prompt(kit)

        self.stage = Stage.INTRO
        self.flags: set[str] = set()
        self.evidence: dict[str, list[str]] = {}
        self.events: list[str] = []

        self._index = 0
        self._question: _Question | None = None
        self._asked = kit.intro_script
        self._used: set[int] = set()
        self._follow_ups_used = 0
        self._off_topic = 0
        self._meaningless = 0
        now = clock()
        self._started_at = self._stage_started = self._topic_started = now

    # ----- public interface ---------------------------------------------

    @property
    def current_topic_id(self) -> str | None:
        if self.stage == Stage.TECHNICAL and self._question is not None:
            return self._question.topic_id
        return None

    def start(self) -> list[Say]:
        now = self._clock()
        self._started_at = self._stage_started = now
        self.stage = Stage.INTRO
        self._asked = self.kit.intro_script
        return [Say(self.kit.intro_script, "script")]

    async def on_candidate_turn(self, text: str) -> list[Action]:
        if self.stage in (Stage.ENDED, Stage.CLOSE):
            return []
        topic = self.current_topic_id
        if wants_repeat(text):
            return [Say(self._asked, "repeat", topic)]
        if wants_human(text):
            return self._end_for_human()
        if not is_meaningful(text):
            self._meaningless += 1
            if self._meaningless >= 2:
                self._meaningless = 0
                return [Say(DIDNT_CATCH, "clarify", topic)]
            return []
        self._meaningless = 0

        if self.stage == Stage.INTRO:
            if consent_declined(text):
                self.flags.add("consent_declined")
                self.stage = Stage.ENDED
                return [Say(CONSENT_DECLINED_LINE, "script"), End("consent_declined")]
            return self._enter(Stage.WARMUP)

        if self._time_up():
            return self._close("time_up")
        if self.stage == Stage.CANDIDATE_QUESTIONS:
            return await self._candidate_question(text)
        decision = await self._decide(text)
        return self._apply(decision)

    def on_silence(self, seconds: float) -> list[Action]:
        if self.stage in (Stage.ENDED, Stage.CLOSE):
            return []
        topic = self.current_topic_id
        if seconds >= 90:
            if self.stage == Stage.INTRO:
                return []
            return [Say(MOVE_ON, "nudge", topic), *self._next()]
        if seconds >= 30:
            return [Say(NUDGE_TAKE_TIME, "nudge", topic)]
        if seconds >= 10:
            return [Say(NUDGE_GENTLE, "nudge", topic)]
        return []

    def end_now(self, reason: str) -> list[Action]:
        return self._close(reason)

    def planned_lines(self) -> list[str]:
        lines = [self.kit.intro_script, self.kit.closing_script, *self._warmup, *self._experience]
        for topic in self._topics:
            question = topic.questions[0]
            lines += [question.ask, *question.follow_ups]
        lines += [entry.a for entry in self.kit.candidate_faq]
        lines += list(self.kit.acknowledgments)
        lines += FIXED_LINES
        return list(dict.fromkeys(lines))

    # ----- decisions -----------------------------------------------------

    async def _decide(self, text: str) -> dict:
        question = self._question
        user = build_decision_prompt(self._asked, question.follow_ups, self._used, question.look_for, text)
        try:
            raw = await self._llm.complete_json(self._system, user, DECISION_SCHEMA, self._timeout)
        except LLMTimeout:
            return self._fallback(text, "timeout")
        except LLMError:
            return self._fallback(text, "error")
        if not isinstance(raw, dict) or raw.get("label") not in LABELS:
            return self._fallback(text, "invalid")
        return raw

    def _fallback(self, text: str, reason: str) -> dict:
        self.events.append(f"llm_fallback:{reason}")
        label = "shallow" if len(text.split()) < FALLBACK_SHORT_ANSWER_WORDS else "answered"
        return {"label": label, "evidence_note": "", "follow_up_index": None, "custom_follow_up": None}

    def _apply(self, decision: dict) -> list[Action]:
        note = decision.get("evidence_note")
        if isinstance(note, str) and note.strip():
            self.evidence.setdefault(self._evidence_key(), []).append(note.strip())
        label = decision["label"]
        topic = self.current_topic_id
        if label == "wants_human":
            return self._end_for_human()
        if label == "unclear":
            return [Say(DIDNT_CATCH, "clarify", topic)]
        if label == "asked_question":
            return [Say(CLARIFY_ASSUMPTIONS, "clarify", topic)]
        if label == "off_topic":
            self._off_topic += 1
            if self._off_topic < 2:
                return [Say(STEER_BACK, "clarify", topic), Say(self._asked, "repeat", topic)]
            return self._next()
        if label == "shallow":
            follow_up = self._follow_up(decision)
            if follow_up is not None:
                return [follow_up]
        return self._next()

    def _follow_up(self, decision: dict) -> Say | None:
        if self._follow_ups_used >= MAX_FOLLOW_UPS or self._follow_up_budget_spent():
            return None
        question = self._question
        index = decision.get("follow_up_index")
        text = None
        if (isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < len(question.follow_ups) and index not in self._used):
            self._used.add(index)
            text = question.follow_ups[index]
        else:
            custom = decision.get("custom_follow_up")
            if isinstance(custom, str) and custom.strip() and is_safe_to_say(custom, self.kit.forbidden_topics):
                text = custom.strip()
            else:
                for i, candidate in enumerate(question.follow_ups):
                    if i not in self._used:
                        self._used.add(i)
                        text = candidate
                        break
        if text is None:
            return None
        self._follow_ups_used += 1
        self._off_topic = 0
        self._asked = text
        return Say(text, "follow_up", self.current_topic_id)

    async def _candidate_question(self, text: str) -> list[Action]:
        words = _normalized_words(text)
        lowered = " ".join(words)
        if ("?" not in text and len(words) <= 6 and _NO_MORE_QUESTIONS.search(lowered)
                and not set(words) & _QUESTION_WORDS):
            return self._close("completed")
        try:
            raw = await self._llm.complete_json(
                self._system, build_faq_prompt(self.kit.candidate_faq, text), FAQ_SCHEMA, self._timeout
            )
        except LLMTimeout:
            self.events.append("llm_fallback:timeout")
            raw = {}
        except LLMError:
            self.events.append("llm_fallback:error")
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        if raw.get("no_more_questions") is True:
            return self._close("completed")
        index = raw.get("faq_index")
        faq = self.kit.candidate_faq
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(faq):
            answer = Say(faq[index].a, "faq")
        else:
            answer = Say(DEFER_TO_HR, "defer")
            self.evidence.setdefault(CANDIDATE_QUESTIONS_KEY, []).append(text.strip())
        self._asked = answer.text
        if self._stage_over() or self._time_up():
            return [answer, *self._close("completed")]
        return [answer, Say(ANYTHING_ELSE, "script")]

    # ----- moving through the interview ---------------------------------

    def _next(self) -> list[Action]:
        if self._time_up():
            return self._close("time_up")
        sequences = {
            Stage.WARMUP: (self._warmup, Stage.EXPERIENCE),
            Stage.EXPERIENCE: (self._experience, Stage.TECHNICAL),
            Stage.TECHNICAL: (self._topics, Stage.CANDIDATE_QUESTIONS),
        }
        if self.stage in sequences:
            items, following = sequences[self.stage]
            self._index += 1
            if self._index < len(items) and not self._stage_over():
                return self._ask_current()
            return self._enter(following)
        if self.stage == Stage.CANDIDATE_QUESTIONS:
            return self._close("completed")
        return []

    def _enter(self, stage: Stage) -> list[Action]:
        self.stage = stage
        self._index = 0
        self._stage_started = self._clock()
        if stage == Stage.CANDIDATE_QUESTIONS:
            self._question = None
            self._asked = CANDIDATE_QUESTIONS_PROMPT
            return [Say(CANDIDATE_QUESTIONS_PROMPT, "script")]
        return self._ask_current()

    def _ask_current(self) -> list[Action]:
        if self.stage == Stage.TECHNICAL:
            topic = self._topics[self._index]
            question = topic.questions[0]
            self._question = _Question(question.ask, list(question.follow_ups), list(question.look_for), topic.id)
            self._topic_started = self._clock()
        else:
            items = self._warmup if self.stage == Stage.WARMUP else self._experience
            self._question = _Question(items[self._index], [], [], None)
        self._used = set()
        self._follow_ups_used = 0
        self._off_topic = 0
        self._asked = self._question.text
        return [Say(self._question.text, "question", self._question.topic_id)]

    def _close(self, reason: str) -> list[Action]:
        self.stage = Stage.ENDED
        return [Say(self.kit.closing_script, "script"), End(reason)]

    def _end_for_human(self) -> list[Action]:
        self.flags.add("wants_human")
        self.stage = Stage.ENDED
        return [Say(WANTS_HUMAN_LINE, "script"), End("wants_human")]

    # ----- clocks ----------------------------------------------------------

    def _evidence_key(self) -> str:
        return self.current_topic_id or EXPERIENCE_KEY

    def _time_up(self) -> bool:
        return self._clock() - self._started_at >= self.kit.duration_minutes * 60

    def _stage_over(self) -> bool:
        budget_minutes = getattr(self.kit.stage_minutes, self.stage.value, None)
        if budget_minutes is None:
            return False
        return self._clock() - self._stage_started >= budget_minutes * 60

    def _follow_up_budget_spent(self) -> bool:
        if self.stage == Stage.TECHNICAL:
            per_topic_s = self.kit.stage_minutes.technical / len(self._topics) * 60
            return self._clock() - self._topic_started >= per_topic_s
        return self._stage_over()


async def generate_experience_questions(kit: Kit, notes: str, llm: LanguageModel, timeout_s: float = 60.0) -> list[str]:
    """Up to 2 experience questions tailored to resume notes, made before the call.

    Any question that's missing or unsafe to say is replaced by the kit's
    question in the same position.
    """
    defaults = list(kit.experience_questions[:2])
    if not notes.strip():
        return defaults
    try:
        raw = await llm.complete_json(build_system_prompt(kit), build_experience_prompt(kit, notes), EXPERIENCE_SCHEMA, timeout_s)
    except LLMError:
        return defaults
    proposed = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(proposed, list):
        return defaults
    result = []
    for position, default in enumerate(defaults):
        candidate = proposed[position] if position < len(proposed) else None
        if isinstance(candidate, str) and candidate.strip() and is_safe_to_say(candidate, kit.forbidden_topics):
            result.append(candidate.strip())
        else:
            result.append(default)
    return result
