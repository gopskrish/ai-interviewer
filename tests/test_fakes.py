import pytest

from interviewer.ai.base import LLMError, LLMTimeout
from interviewer.ai.fakes import FakeLLM, FakeSTT, FakeTTS


async def test_fake_llm_raises_queued_exception():
    llm = FakeLLM([LLMTimeout("x")])
    with pytest.raises(LLMTimeout):
        await llm.complete_json("S", "U", {}, timeout_s=1)


async def test_fake_llm_returns_in_order_and_records_calls():
    llm = FakeLLM([{"a": 1}, {"b": 2}])
    assert await llm.complete_json("S1", "U1", {"x": 1}, timeout_s=1) == {"a": 1}
    assert await llm.complete_json("S2", "U2", {}, timeout_s=1) == {"b": 2}
    assert llm.calls == [("S1", "U1", {"x": 1}), ("S2", "U2", {})]


async def test_fake_llm_exhausted_raises_llm_error():
    with pytest.raises(LLMError, match="no more fake responses"):
        await FakeLLM([]).complete_json("S", "U", {}, timeout_s=1)


async def test_fake_stt_returns_texts_in_order():
    stt = FakeSTT(["one", "two"])
    assert [await stt.transcribe(b"", 16000), await stt.transcribe(b"", 16000), await stt.transcribe(b"", 16000)] == ["one", "two", ""]
    assert stt.calls == 3


async def test_fake_tts_returns_silence_proportional_to_text():
    tts = FakeTTS(sample_rate=24000, seconds_per_char=0.01)
    pcm = await tts.synthesize("hello")
    assert len(pcm) == 2 * 1200 and set(pcm) == {0} and tts.texts == ["hello"] and tts.sample_rate == 24000
