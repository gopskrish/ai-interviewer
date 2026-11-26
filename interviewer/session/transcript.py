"""What was said, by whom and when (seconds from the start of the session)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


def format_ts(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


@dataclass
class Turn:
    speaker: Literal["ai", "candidate"]
    text: str
    start_s: float
    end_s: float
    kind: str | None = None
    topic_id: str | None = None
    interrupted: bool = False


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Transcript:
    def __init__(self, turns: list[Turn] | None = None):
        self.turns: list[Turn] = list(turns or [])

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def candidate_turns(self, topic_id: str | None = None) -> list[Turn]:
        return [t for t in self.turns if t.speaker == "candidate" and (topic_id is None or t.topic_id == topic_id)]

    def to_dict(self) -> dict:
        return {"turns": [asdict(t) for t in self.turns]}

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls([Turn(**t) for t in data.get("turns", [])])

    def to_text(self) -> str:
        lines = []
        for t in self.turns:
            who = "AI" if t.speaker == "ai" else "Candidate"
            suffix = " (interrupted)" if t.interrupted else ""
            lines.append(f"[{format_ts(t.start_s)}] {who}: {t.text}{suffix}")
        return "\n".join(lines)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        write_atomic(directory / "transcript.json", json.dumps(self.to_dict(), indent=2))
        write_atomic(directory / "transcript.txt", self.to_text())
