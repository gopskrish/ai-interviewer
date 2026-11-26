"""Interview kits: what the AI asks, in which order, and how answers are scored.

A kit is a YAML file in the kits directory. Its file name without the
extension is the kit id (``kits/senior-dotnet.yaml`` is ``senior-dotnet``).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

CONSENT_QUESTION = "Is that OK with you?"


class KitError(Exception):
    """A kit file that can't be used, with one message per problem."""

    def __init__(self, path: Path | str, problems: list[str]):
        self.path = Path(path)
        self.problems = problems
        super().__init__(f"{self.path.name}: " + "; ".join(problems))


class Question(BaseModel):
    ask: str
    follow_ups: list[str] = Field(default_factory=list)
    look_for: list[str] = Field(default_factory=list)


class Topic(BaseModel):
    id: str
    name: str
    weight: float = 1.0
    questions: list[Question]
    rubric: dict[int, str]

    @field_validator("questions")
    @classmethod
    def _at_least_one_question(cls, value: list[Question]) -> list[Question]:
        if not value:
            raise ValueError("each topic needs at least 1 question")
        return value

    @field_validator("rubric")
    @classmethod
    def _four_levels(cls, value: dict[int, str]) -> dict[int, str]:
        if set(value) != {1, 2, 3, 4}:
            raise ValueError("the rubric must have exactly the levels 1, 2, 3 and 4")
        return value


class FaqEntry(BaseModel):
    q: str
    a: str


class StageMinutes(BaseModel):
    intro: float = 1
    warmup: float = 2
    experience: float = 5
    technical: float = 18
    candidate_questions: float = 3
    close: float = 1

    def total(self) -> float:
        return sum(self.model_dump().values())


class Kit(BaseModel):
    role: str
    duration_minutes: int = 30
    intro_script: str
    closing_script: str
    warmup_questions: list[str] = Field(min_length=1)
    experience_questions: list[str] = Field(min_length=1)
    topics: list[Topic]
    interview_topics: list[str]
    stage_minutes: StageMinutes = Field(default_factory=StageMinutes)
    candidate_faq: list[FaqEntry] = Field(default_factory=list)
    forbidden_topics: list[str] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    acknowledgments: list[str] = Field(default_factory=lambda: ["Thanks.", "Okay, got it.", "Thank you."])

    @field_validator("intro_script")
    @classmethod
    def _discloses_ai_and_recording(cls, value: str) -> str:
        missing = []
        if not re.search(r"\bAI\b", value):
            missing.append("say it's an AI (the word 'AI')")
        if "record" not in value.lower():
            missing.append("say the session is recorded")
        if CONSENT_QUESTION not in value:
            missing.append(f"ask '{CONSENT_QUESTION}'")
        if missing:
            raise ValueError("the intro must " + ", ".join(missing))
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> "Kit":
        problems = []
        ids = [t.id for t in self.topics]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            problems.append(f"topics: duplicate topic ids {duplicates}")
        if not self.interview_topics:
            problems.append("interview_topics: list at least one topic id")
        if len(set(self.interview_topics)) != len(self.interview_topics):
            problems.append("interview_topics: each topic may appear only once")
        for topic_id in self.interview_topics:
            if topic_id not in ids:
                problems.append(f"interview_topics: unknown topic id '{topic_id}'")
        total = self.stage_minutes.total()
        if total > self.duration_minutes:
            problems.append(
                f"stage_minutes: the stages add up to {total:g} minutes, "
                f"more than duration_minutes ({self.duration_minutes})"
            )
        if problems:
            raise ValueError("\n".join(problems))
        return self

    def topic(self, topic_id: str) -> Topic:
        for topic in self.topics:
            if topic.id == topic_id:
                return topic
        raise KeyError(topic_id)


def _problems_from(error: ValidationError) -> list[str]:
    problems = []
    for err in error.errors():
        where = ".".join(str(part) for part in err["loc"])
        cause = err.get("ctx", {}).get("error")
        messages = str(cause).split("\n") if cause is not None else [err["msg"]]
        for message in messages:
            problems.append(f"{where}: {message}" if where else message)
    return problems


def load_kit(path: Path) -> Kit:
    """Load and validate a kit file, raising ``KitError`` with readable problems."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KitError(path, [f"can't read the file: {exc}"]) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else "an unknown line"
        reason = getattr(exc, "problem", None) or str(exc)
        raise KitError(path, [f"YAML syntax error at {where}: {reason}"]) from exc
    if not isinstance(data, dict):
        raise KitError(path, ["the file must contain YAML 'key: value' pairs"])
    try:
        return Kit.model_validate(data)
    except ValidationError as exc:
        raise KitError(path, _problems_from(exc)) from exc


def list_kits(kits_dir: Path) -> tuple[dict[str, Kit], dict[str, str]]:
    """Load every kit in ``kits_dir``: (kits by id, error text by id)."""
    kits: dict[str, Kit] = {}
    errors: dict[str, str] = {}
    for path in sorted(Path(kits_dir).glob("*.y*ml")):
        try:
            kits[path.stem] = load_kit(path)
        except KitError as exc:
            errors[path.stem] = "; ".join(exc.problems)
    return kits, errors
