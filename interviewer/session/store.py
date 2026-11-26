"""Session records on disk: one folder per session under ``<data>/sessions/``."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from interviewer.session.transcript import write_atomic

_ID = re.compile(r"^[0-9a-f]{12}$")


@dataclass
class SessionRecord:
    id: str
    token: str
    kind: Literal["interview", "practice", "check"]
    candidate_name: str
    kit_id: str
    meeting_url: str | None = None
    bot_id: str | None = None
    bot_state: str | None = None
    status: str = "created"
    end_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = ""
    notes: str = ""


class SessionStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def create(self, kind, candidate_name: str, kit_id: str, meeting_url: str | None = None, notes: str = "") -> SessionRecord:
        record = SessionRecord(
            id=uuid.uuid4().hex[:12],
            token=secrets.token_urlsafe(24),
            kind=kind,
            candidate_name=candidate_name,
            kit_id=kit_id,
            meeting_url=meeting_url,
            notes=notes,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.save(record)
        return record

    def save(self, record: SessionRecord) -> None:
        write_atomic(self.dir(record.id) / "session.json", json.dumps(asdict(record), indent=2))

    def get(self, session_id: str) -> SessionRecord | None:
        if not _ID.match(session_id or ""):
            return None
        path = self.dir(session_id) / "session.json"
        if not path.exists():
            return None
        return SessionRecord(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[SessionRecord]:
        records = []
        for path in self.sessions_dir.glob("*/session.json"):
            try:
                records.append(SessionRecord(**json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, TypeError):
                continue
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def cleanup(self, older_than_days: int, now: datetime | None = None) -> list[str]:
        """Delete sessions older than ``older_than_days``; returns the deleted ids."""
        now = now or datetime.now(UTC)
        cutoff = timedelta(days=older_than_days)
        deleted = []
        for record in self.list():
            try:
                created = datetime.fromisoformat(record.created_at)
            except ValueError:
                continue
            if now - created > cutoff:
                shutil.rmtree(self.dir(record.id), ignore_errors=True)
                deleted.append(record.id)
        return deleted
