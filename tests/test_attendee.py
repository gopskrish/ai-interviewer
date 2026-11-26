import json
import urllib.parse

import httpx
import pytest

from interviewer.attendee import (
    AttendeeClient,
    AttendeeError,
    BotInfo,
    MeetingLinkError,
    join_timed_out,
    normalize_meeting_link,
    status_label,
)

TEAMS = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_abc%40thread.v2/0?context=%7b%22Tid%22%3a%22t1%22%2c%22Oid%22%3a%22o1%22%7d"
LINK_ERROR = "That doesn't look like a Teams meeting link. Copy the link from the meeting invite in Teams or Outlook."


def mock_http(captured, json_body=None, status=200):
    def handler(request):
        captured.append(request)
        return httpx.Response(status, json=json_body if json_body is not None else {})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_create_bot_request_shape():
    captured = []
    c = AttendeeClient("http://attendee-app:8000", "abc", mock_http(captured, {"id": "bot_1", "state": "joining"}, 201))
    info = await c.create_bot(TEAMS, "AI Interviewer", "wss://interviewer:8443/ws/audio/s1?token=t")
    r = captured[0]
    assert info == BotInfo("bot_1", "joining") and r.method == "POST" and r.url.path == "/api/v1/bots"
    assert r.headers["Authorization"] == "Token abc"
    assert json.loads(r.content) == {
        "meeting_url": TEAMS, "bot_name": "AI Interviewer", "recording_settings": {"format": "none"},
        "websocket_settings": {"audio": {"url": "wss://interviewer:8443/ws/audio/s1?token=t", "sample_rate": 16000}},
    }


async def test_create_bot_error_raises_with_detail():
    c = AttendeeClient("http://attendee-app:8000", "abc", mock_http([], {"meeting_url": ["bad"]}, 400))
    with pytest.raises(AttendeeError) as e:
        await c.create_bot(TEAMS, "AI Interviewer", "wss://x")
    assert e.value.status == 400 and "bad" in e.value.detail


async def test_unreachable_attendee_raises_attendee_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    c = AttendeeClient("http://attendee-app:8000", "abc", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(AttendeeError, match="Attendee"):
        await c.get_bot("bot_1")


async def test_get_bot_and_leave_paths():
    captured = []
    c = AttendeeClient("http://attendee-app:8000/", "abc", mock_http(captured, {"id": "bot_1", "state": "waiting_room"}))
    assert await c.get_bot("bot_1") == BotInfo("bot_1", "waiting_room")
    await c.leave("bot_1")
    assert [(r.method, r.url.path) for r in captured] == [("GET", "/api/v1/bots/bot_1"), ("POST", "/api/v1/bots/bot_1/leave")]


def test_normalize_unwraps_outlook_safelinks():
    wrapped = ("  <https://nam02.safelinks.protection.outlook.com/?url=" + urllib.parse.quote(TEAMS, safe="")
               + "&data=05%7C02&reserved=0>  ")
    assert normalize_meeting_link(wrapped) == TEAMS


def test_normalize_accepts_teams_live_and_trims():
    link = "https://teams.live.com/meet/9876543210?p=abc"
    assert normalize_meeting_link(f"\n {link} \t") == link
    assert normalize_meeting_link(TEAMS) == TEAMS


@pytest.mark.parametrize("raw", ["https://zoom.us/j/123", "http://teams.microsoft.com/l/meetup-join/x", "", "not a link",
                                 "https://teams.microsoft.com.evil.example/l/meetup-join/x"])
def test_normalize_rejects_non_teams(raw):
    with pytest.raises(MeetingLinkError) as e:
        normalize_meeting_link(raw)
    assert str(e.value) == LINK_ERROR


def test_status_labels_and_join_timeout():
    assert status_label("waiting_room") == 'In the lobby. Admit "AI Interviewer" in Teams.'
    assert status_label("joining") == "Joining the meeting…"
    assert status_label("joined_recording") == status_label("joined_not_recording") == "In the call"
    assert status_label("ended") == "Left the meeting" and status_label("fatal_error") == "Failed to join. Check the Attendee logs."
    assert status_label("staged") == "Starting…" and status_label("weird_state") == "weird_state"
    assert join_timed_out(0, 181, "waiting_room") is True
    assert join_timed_out(0, 181, "joined_recording") is False
    assert join_timed_out(0, 179, "joining") is False
    assert join_timed_out(0, 500, "ended") is False
