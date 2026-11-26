"""The scorecard a human reviews. Evidence and scores per topic; never a hire/reject call."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Quote:
    text: str
    at: str


@dataclass
class TopicScore:
    topic_id: str
    name: str
    score: int | None
    not_enough_signal: bool
    quotes: list[Quote] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    evidence_missing: bool = False

    @property
    def score_label(self) -> str:
        return "Not enough signal" if self.not_enough_signal or self.score is None else f"{self.score}/4"


@dataclass
class Scorecard:
    session_id: str
    candidate_name: str
    role: str
    created_at: str
    topics: list[TopicScore]
    flags: list[str] = field(default_factory=list)
    end_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scorecard":
        topics = [
            TopicScore(**{**t, "quotes": [Quote(**q) for q in t.get("quotes", [])]})
            for t in data.get("topics", [])
        ]
        return cls(**{**data, "topics": topics})

    def to_markdown(self) -> str:
        lines = [
            f"# Interview scorecard: {self.candidate_name}",
            "",
            f"Role: {self.role}  ",
            f"Date: {self.created_at[:10]}  ",
            f"Ended: {self.end_reason or 'unknown'}  ",
            f"Flags: {', '.join(self.flags) if self.flags else 'none'}",
            "",
            "Evidence for a human reviewer. Read the transcript before deciding anything.",
        ]
        for topic in self.topics:
            lines += ["", f"## {topic.name}: {topic.score_label}"]
            if topic.strengths:
                lines += ["", "Strengths:", *[f"- {s}" for s in topic.strengths]]
            if topic.concerns:
                lines += ["", "Concerns:", *[f"- {c}" for c in topic.concerns]]
            if topic.quotes:
                lines += ["", "Evidence:", *[f'- [{q.at}] "{q.text}"' for q in topic.quotes]]
            elif topic.evidence_missing:
                lines += ["", "No verified quotes for this score. Check the transcript."]
        return "\n".join(lines) + "\n"

    def to_adaptive_card(self) -> dict:
        body: list[dict] = [
            {"type": "TextBlock", "text": f"Interview scorecard: {self.candidate_name}",
             "weight": "Bolder", "size": "Medium", "wrap": True},
            {"type": "TextBlock", "text": f"{self.role} · {self.created_at[:10]}", "isSubtle": True, "wrap": True},
            {"type": "FactSet", "facts": [{"title": t.name, "value": t.score_label} for t in self.topics]},
        ]
        if self.flags:
            body.append({"type": "TextBlock", "text": f"Flags: {', '.join(self.flags)}", "color": "Attention", "wrap": True})
        body.append({"type": "TextBlock", "text": "Evidence for a human reviewer. Read the transcript before deciding anything.",
                     "isSubtle": True, "wrap": True})
        return {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
        }
