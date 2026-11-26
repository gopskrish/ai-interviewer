import base64
import json

import pytest

from interviewer.session.protocol import ProtocolError, bot_output_message, parse_audio_message
from tests.helpers import tone


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_protocol_roundtrip():
    msg = {"trigger": "realtime_audio.mixed", "bot_id": "b", "data": {"chunk": b64(tone(0.1)), "timestamp_ms": 1, "sample_rate": 16000}}
    a = parse_audio_message(json.dumps(msg))
    assert a.sample_rate == 16000 and len(a.pcm) == 3200 and a.timestamp_ms == 1
    out = bot_output_message(b"\x00\x00" * 10, 24000)
    assert out == {"trigger": "realtime_audio.bot_output", "data": {"chunk": b64(b"\x00\x00" * 10), "sample_rate": 24000}}


def test_protocol_accepts_dicts():
    msg = {"trigger": "realtime_audio.mixed", "data": {"chunk": b64(b"\x01\x00"), "sample_rate": 16000}}
    assert parse_audio_message(msg).timestamp_ms is None


def test_protocol_ignores_other_triggers():
    assert parse_audio_message(json.dumps({"trigger": "transcript.update"})) is None


@pytest.mark.parametrize("raw", [
    json.dumps({"trigger": "realtime_audio.mixed", "data": {"sample_rate": 16000}}),
    json.dumps({"trigger": "realtime_audio.mixed", "data": {"chunk": "%%%", "sample_rate": 16000}}),
    "not json",
])
def test_protocol_malformed_raises(raw):
    with pytest.raises(ProtocolError):
        parse_audio_message(raw)
