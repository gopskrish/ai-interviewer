"""Attendee's realtime-audio websocket messages (the practice page uses the same format).

In:  {"trigger": "realtime_audio.mixed", "bot_id": ..., "data": {"chunk": <b64 PCM16>, "timestamp_ms": ..., "sample_rate": 16000}}
Out: {"trigger": "realtime_audio.bot_output", "data": {"chunk": <b64 PCM16>, "sample_rate": 24000}}
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass

MIXED_AUDIO = "realtime_audio.mixed"
BOT_OUTPUT = "realtime_audio.bot_output"


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class AudioIn:
    pcm: bytes
    sample_rate: int
    timestamp_ms: int | None


def parse_audio_message(raw: str | bytes | dict) -> AudioIn | None:
    """Meeting audio from a websocket message; None for messages that aren't audio."""
    if isinstance(raw, dict):
        message = raw
    else:
        try:
            message = json.loads(raw)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            raise ProtocolError("message isn't JSON") from exc
    if not isinstance(message, dict):
        raise ProtocolError("message isn't a JSON object")
    if message.get("trigger") != MIXED_AUDIO:
        return None
    data = message.get("data")
    try:
        pcm = base64.b64decode(data["chunk"], validate=True)
        sample_rate = int(data["sample_rate"])
    except (TypeError, KeyError, ValueError, binascii.Error) as exc:
        raise ProtocolError("malformed realtime_audio.mixed message") from exc
    timestamp = data.get("timestamp_ms")
    return AudioIn(pcm=pcm, sample_rate=sample_rate, timestamp_ms=timestamp if isinstance(timestamp, int) else None)


def bot_output_message(pcm: bytes, sample_rate: int) -> dict:
    return {"trigger": BOT_OUTPUT, "data": {"chunk": base64.b64encode(pcm).decode("ascii"), "sample_rate": sample_rate}}
