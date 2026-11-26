"""Posts a scorecard to a Teams channel through a webhook (for example a Teams Workflows URL)."""

from __future__ import annotations

import httpx


async def post_scorecard(webhook_url: str, card: dict, http: httpx.AsyncClient) -> None:
    payload = {
        "type": "message",
        "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}],
    }
    response = await http.post(webhook_url, json=payload, timeout=30)
    response.raise_for_status()
