"""The dashboard, forms, JSON API, downloads and the audio websocket."""

import html
import re

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.helpers import TEAMS_LINK, chunks, mixed, silence, tone

FORM = {"kit_id": "senior-dotnet", "candidate_name": "Asha", "meeting_link": TEAMS_LINK, "notes": ""}


def start_interview(client) -> str:
    r = client.post("/sessions", data=FORM, follow_redirects=False)
    return r.headers["location"].rsplit("/", 1)[1]


def open_practice(client) -> tuple[str, str]:
    page = client.get("/practice")
    assert page.status_code == 200
    session_id = re.search(r'data-session-id="([0-9a-f]{12})"', page.text).group(1)
    ws_url = html.unescape(re.search(r'data-ws-url="([^"]+)"', page.text).group(1))
    return session_id, ws_url


def test_post_sessions_creates_bot_and_redirects(client, fake_attendee):
    r = client.post("/sessions", data=FORM, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/sessions/")
    url = fake_attendee.created[0]["audio_ws_url"]
    assert url.startswith("wss://interviewer:8443/ws/audio/") and "?token=" in url
    page = client.get(r.headers["location"])
    assert page.status_code == 200 and "Asha" in page.text


def test_start_form_rejects_bad_link(client, fake_attendee):
    r = client.post("/sessions", data={**FORM, "meeting_link": "https://zoom.us/j/123"})
    assert r.status_code == 400
    assert "doesn't look like a Teams meeting link" in html.unescape(r.text)
    assert 'value="Asha"' in r.text  # what was typed is kept
    assert fake_attendee.created == []


def test_start_form_lists_kits(client):
    page = client.get("/start")
    assert page.status_code == 200 and 'value="senior-dotnet"' in page.text


def test_ws_rejects_bad_token(client):
    session_id, _ = open_practice(client)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/audio/{session_id}?token=wrong") as ws:
            ws.receive_text()
    assert exc.value.code == 4403


def test_ws_practice_roundtrip(client):
    _, ws_url = open_practice(client)
    assert ws_url.startswith("ws://testserver/ws/audio/") and "?token=" in ws_url
    with client.websocket_connect(ws_url.removeprefix("ws://testserver")) as ws:
        for piece in chunks(tone(1.0) + silence(3.5)):
            ws.send_text(mixed(piece))
        message = ws.receive_json()
    assert message["trigger"] == "realtime_audio.bot_output"
    assert message["data"]["sample_rate"] == 24000 and message["data"]["chunk"]


def test_api_session_json(client):
    session_id, _ = open_practice(client)
    data = client.get(f"/api/sessions/{session_id}").json()
    assert data["status"] == "created" and data["transcript"] == [] and data["scorecard"] is None
    assert {"bot_state", "bot_label", "error", "end_reason"} <= data.keys()
    assert client.get("/api/sessions/ffffffffffff").status_code == 404
    assert client.get("/sessions/ffffffffffff").status_code == 404


def test_end_button_ends_and_leaves(client, fake_attendee):
    session_id = start_interview(client)
    assert client.post(f"/api/sessions/{session_id}/end").status_code == 200
    assert fake_attendee.left == ["bot_1"]
    assert client.get(f"/api/sessions/{session_id}").json()["end_reason"] == "ended_by_owner"


def test_begin_before_the_bot_is_in_the_call_returns_409(client):
    session_id = start_interview(client)
    r = client.post(f"/api/sessions/{session_id}/begin")
    assert r.status_code == 409 and "isn't in the call yet" in r.json()["error"]


def test_dashboard_lists_sessions(client):
    start_interview(client)
    page = client.get("/")
    assert page.status_code == 200 and "Asha" in page.text


def test_dashboard_shows_kit_errors(client_with_broken_kit):
    body = html.unescape(client_with_broken_kit.get("/").text)
    assert "broken" in body and "line 3, column 1" in body


def test_scorecard_download_404_then_200(client):
    session_id, _ = open_practice(client)
    assert client.get(f"/sessions/{session_id}/scorecard.md").status_code == 404
    assert client.get(f"/sessions/{session_id}/transcript.txt").status_code == 404
    folder = client.app.state.manager.store.dir(session_id)
    (folder / "scorecard.md").write_text("# Scorecard\n", encoding="utf-8")
    r = client.get(f"/sessions/{session_id}/scorecard.md")
    assert r.status_code == 200 and r.text == "# Scorecard\n"
    assert "attachment" in r.headers["content-disposition"]


def test_cross_site_post_is_rejected(client, fake_attendee):
    r = client.post("/sessions", data=FORM, headers={"origin": "https://evil.example"})
    assert r.status_code == 403 and fake_attendee.created == []
    same_site = client.post("/sessions", data=FORM, headers={"origin": "http://testserver"}, follow_redirects=False)
    assert same_site.status_code == 303


def test_scripts_are_served_as_javascript(client):
    for name in ("audio-utils.mjs", "practice.js", "session.js", "mic-worklet.js"):
        r = client.get(f"/static/{name}")
        assert r.status_code == 200 and "javascript" in r.headers["content-type"]


def test_check_page_starts_an_echo_check(client, fake_attendee):
    assert client.get("/check").status_code == 200
    r = client.post("/check", data={"meeting_link": TEAMS_LINK}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/check?session=")
    session_id = r.headers["location"].rsplit("=", 1)[1]
    page = client.get(r.headers["location"])
    assert page.status_code == 200 and f'data-session-id="{session_id}"' in page.text
    data = client.get(f"/api/sessions/{session_id}").json()
    assert data["kind"] == "check" and data["warnings"] == []
    assert fake_attendee.created[0]["meeting_url"] == TEAMS_LINK


def test_check_form_rejects_bad_link(client):
    r = client.post("/check", data={"meeting_link": "https://zoom.us/j/1"})
    assert r.status_code == 400 and "Teams meeting link" in html.unescape(r.text)


def test_speed_test_runs_in_the_background(client):
    import time

    from interviewer.ai.fakes import FakeLLM

    answered = {"label": "answered", "evidence_note": "", "follow_up_index": None, "custom_follow_up": None}
    client.app.state.llm_factory = lambda model: FakeLLM([answered, answered])
    assert client.get("/api/bench").json() == {"running": False, "report": None, "error": None}
    assert client.post("/api/bench", data={"models": "fake-model"}).status_code == 202
    for _ in range(200):
        state = client.get("/api/bench").json()
        if not state["running"]:
            break
        time.sleep(0.01)
    assert state["error"] is None and "fake-model" in state["report"]
