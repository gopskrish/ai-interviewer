"""Talks to the self-hosted Attendee API, which sends a bot into the Teams meeting."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import httpx

TEAMS_HOSTS = {"teams.microsoft.com", "teams.live.com", "gov.teams.microsoft.us", "dod.teams.microsoft.us"}
SAFELINKS_SUFFIX = "safelinks.protection.outlook.com"
LINK_ERROR = "That doesn't look like a Teams meeting link. Copy the link from the meeting invite in Teams or Outlook."

_LABELS = {
    "joining": "Joining the meeting…",
    "waiting_room": 'In the lobby. Admit "AI Interviewer" in Teams.',
    "joined_recording": "In the call",
    "joined_not_recording": "In the call",
    "joined_recording_paused": "In the call",
    "joined_recording_permission_denied": "In the call",
    "leaving": "Left the meeting",
    "post_processing": "Left the meeting",
    "ended": "Left the meeting",
    "fatal_error": "Failed to join. Check the Attendee logs.",
    "ready": "Starting…",
    "staged": "Starting…",
    "scheduled": "Starting…",
    "connecting": "Starting…",
}
_LEFT_STATES = {"leaving", "post_processing", "ended"}


class MeetingLinkError(ValueError):
    def __init__(self) -> None:
        super().__init__(LINK_ERROR)


class AttendeeError(Exception):
    def __init__(self, status: int | None, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"Attendee error{f' {status}' if status else ''}: {detail}")


@dataclass(frozen=True)
class BotInfo:
    id: str
    state: str


def normalize_meeting_link(raw: str) -> str:
    """The Teams link to send the bot to, unwrapping Outlook Safe Links."""
    link = (raw or "").strip().strip("<>").strip()
    parsed = urlparse(link)
    host = (parsed.hostname or "").lower()
    if host.endswith(SAFELINKS_SUFFIX):
        inner = parse_qs(parsed.query).get("url", [""])[0]
        link = inner.strip()
        parsed = urlparse(link)
        host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in TEAMS_HOSTS:
        raise MeetingLinkError()
    return link


def status_label(state: str) -> str:
    return _LABELS.get(state, state)


def join_timed_out(created_at: float, now: float, state: str, timeout_s: float = 180.0) -> bool:
    in_call_or_done = state.startswith("joined_") or state in _LEFT_STATES
    return not in_call_or_done and now - created_at >= timeout_s


class AttendeeClient:
    def __init__(self, base_url: str, api_key: str, http: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
        self._http = http

    async def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        try:
            response = await self._http.request(method, f"{self.base_url}{path}", json=json, headers=self._headers, timeout=30)
        except httpx.HTTPError as exc:
            raise AttendeeError(None, f"can't reach Attendee at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise AttendeeError(response.status_code, response.text[:1000])
        try:
            return response.json() if response.content else {}
        except ValueError:
            return {}

    async def create_bot(self, meeting_url: str, bot_name: str, audio_ws_url: str, sample_rate: int = 16000) -> BotInfo:
        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "recording_settings": {"format": "none"},
            "websocket_settings": {"audio": {"url": audio_ws_url, "sample_rate": sample_rate}},
        }
        data = await self._request("POST", "/api/v1/bots", payload)
        return BotInfo(id=str(data.get("id", "")), state=str(data.get("state", "")))

    async def get_bot(self, bot_id: str) -> BotInfo:
        data = await self._request("GET", f"/api/v1/bots/{bot_id}")
        return BotInfo(id=str(data.get("id", bot_id)), state=str(data.get("state", "")))

    async def leave(self, bot_id: str) -> None:
        await self._request("POST", f"/api/v1/bots/{bot_id}/leave")
