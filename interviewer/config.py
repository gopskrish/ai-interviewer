"""Service settings, read from environment variables.

Each field's environment variable is the field name in upper case,
for example ``ollama_model`` is read from ``OLLAMA_MODEL``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("/data")
    kits_dir: Path = Path("/app/kits")
    models_dir: Path = Path("/models")

    ai_mode: Literal["local", "openai"] = "local"

    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:4b-instruct"
    llm_timeout_s: float = 20.0

    whisper_model: str = "small"
    whisper_cpu_threads: int = 4

    tts_engine: Literal["kokoro", "piper"] = "kokoro"
    kokoro_model: Path = Path("/models/kokoro/kokoro-v1.0.int8.onnx")
    kokoro_voices: Path = Path("/models/kokoro/voices-v1.0.bin")
    kokoro_voice: str = "af_heart"
    piper_voice: Path = Path("/models/piper/en_US-lessac-medium.onnx")

    attendee_base_url: str = "http://attendee-app:8000"
    attendee_api_key: str = ""
    public_ws_base: str = "wss://interviewer:8443"

    teams_webhook_url: str = ""
    retention_days: int = 30
    bot_name: str = "AI Interviewer"
    end_of_turn_silence_s: float = 1.5

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from ``env`` (default: ``os.environ``).

        Empty values count as unset. Invalid values raise ``ValueError``
        (pydantic's ``ValidationError`` is a ``ValueError``).
        """
        source = os.environ if env is None else env
        values: dict[str, str] = {}
        for field in cls.model_fields:
            raw = source.get(field.upper())
            if raw is not None and raw.strip() != "":
                values[field] = raw.strip()
        return cls.model_validate(values)
