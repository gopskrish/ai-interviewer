import json

import httpx
import pytest

from interviewer.scoring.teams import post_scorecard


def mock_http(captured, status=202):
    def handler(request):
        captured.append(request)
        return httpx.Response(status, text="ok")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_post_scorecard_payload():
    captured = []
    await post_scorecard("https://example.webhook/x", {"type": "AdaptiveCard"}, mock_http(captured))
    assert json.loads(captured[0].content) == {"type": "message", "attachments": [
        {"contentType": "application/vnd.microsoft.card.adaptive", "content": {"type": "AdaptiveCard"}}]}
    assert str(captured[0].url) == "https://example.webhook/x"


async def test_post_scorecard_raises_on_error():
    with pytest.raises(httpx.HTTPStatusError):
        await post_scorecard("https://example.webhook/x", {}, mock_http([], status=400))
