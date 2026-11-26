import pytest

from interviewer.config import Settings


def test_settings_defaults():
    s = Settings.from_env({})
    assert (s.ai_mode, s.ollama_model, s.llm_timeout_s) == ("local", "qwen3:4b-instruct", 20.0)
    assert (s.retention_days, s.bot_name, s.end_of_turn_silence_s) == (30, "AI Interviewer", 1.5)
    assert s.public_ws_base == "wss://interviewer:8443"


def test_settings_rejects_unknown_ai_mode():
    with pytest.raises(ValueError):
        Settings.from_env({"AI_MODE": "cloud"})
