"""Language model served by Ollama, asked for JSON that matches a schema."""

from __future__ import annotations

import asyncio
import json

import httpx

from interviewer.ai.base import LLMError, LLMTimeout


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        http: httpx.AsyncClient,
        num_ctx: int = 8192,
        temperature: float = 0.2,
        keep_alive: str = "30m",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = http
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.keep_alive = keep_alive

    async def complete_json(self, system: str, user: str, schema: dict, timeout_s: float) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            "keep_alive": self.keep_alive,
        }
        url = f"{self.base_url}/api/chat"
        try:
            response = await asyncio.wait_for(self._http.post(url, json=payload, timeout=None), timeout_s)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise LLMTimeout(f"{self.model} didn't answer within {timeout_s:g} s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"can't reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"Ollama returned {response.status_code}: {response.text[:500]}")
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"unexpected reply from Ollama: {response.text[:200]}") from exc
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"{self.model} returned invalid JSON: {str(content)[:200]}") from exc
        if not isinstance(data, dict):
            raise LLMError(f"{self.model} returned JSON that isn't an object")
        return data
