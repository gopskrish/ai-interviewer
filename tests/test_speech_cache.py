from interviewer.ai.fakes import FakeTTS
from interviewer.session.speech_cache import SpeechCache


async def test_speech_cache_hits_disk_on_second_instance(tmp_path):
    first, second = FakeTTS(), FakeTTS()
    audio = await SpeechCache(first, tmp_path).get("Hello there.")
    cache = SpeechCache(second, tmp_path)
    assert cache.has("Hello there.") and await cache.get("Hello there.") == audio
    assert first.texts == ["Hello there."] and second.texts == []


async def test_warm_synthesises_only_missing_lines(tmp_path):
    tts = FakeTTS()
    cache = SpeechCache(tts, tmp_path)
    await cache.get("One.")
    await cache.warm(["One.", "Two.", "Two."])
    assert tts.texts == ["One.", "Two."] and cache.sample_rate == 24000
