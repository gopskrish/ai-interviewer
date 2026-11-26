import asyncio
import json

import httpx
import pytest

from interviewer.ai.base import LLMError, LLMTimeout
from interviewer.ai.ollama import OllamaLLM

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}


def make_llm(reply=None, status=200, text=None, exc=None, delay_s=0.0):
    captured = []

    async def handler(request):
        captured.append(request)
        if delay_s:
            await asyncio.sleep(delay_s)
        if exc is not None:
            raise exc
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=reply)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaLLM("http://ollama:11434", "qwen3:4b-instruct", http), captured


async def test_ollama_sends_schema_and_options():
    llm, captured = make_llm(reply={"message": {"content": '{"label":"answered"}'}})
    assert await llm.complete_json("SYS", "USER", SCHEMA, timeout_s=20) == {"label": "answered"}
    body = json.loads(captured[0].content)
    assert captured[0].url.path == "/api/chat"
    assert body["model"] == "qwen3:4b-instruct" and body["stream"] is False and body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0.2, "num_ctx": 8192} and body["keep_alive"] == "30m"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert [m["content"] for m in body["messages"]] == ["SYS", "USER"]
    assert "think" not in body


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
async def test_ollama_invalid_json_raises_llm_error(content):
    llm, _ = make_llm(reply={"message": {"content": content}})
    with pytest.raises(LLMError):
        await llm.complete_json("SYS", "USER", SCHEMA, timeout_s=20)


async def test_ollama_http_error_raises_llm_error():
    llm, _ = make_llm(status=500, text="model 'x' not found")
    with pytest.raises(LLMError, match="not found"):
        await llm.complete_json("SYS", "USER", SCHEMA, timeout_s=20)


async def test_ollama_timeout_raises_llm_timeout():
    llm, _ = make_llm(exc=httpx.ReadTimeout("timed out"))
    with pytest.raises(LLMTimeout):
        await llm.complete_json("SYS", "USER", SCHEMA, timeout_s=20)


async def test_ollama_slow_reply_raises_llm_timeout():
    llm, _ = make_llm(reply={"message": {"content": "{}"}}, delay_s=1.0)
    with pytest.raises(LLMTimeout):
        await llm.complete_json("SYS", "USER", SCHEMA, timeout_s=0.05)


async def test_ollama_unreachable_raises_llm_error():
    llm, _ = make_llm(exc=httpx.ConnectError("refused"))
    with pytest.raises(LLMError, match="Ollama"):
        await llm.complete_json("SYS", "USER", SCHEMA, timeout_s=20)
