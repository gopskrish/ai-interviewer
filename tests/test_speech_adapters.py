from pathlib import Path
from types import SimpleNamespace

import numpy as np

from interviewer.ai.kokoro_tts import KokoroTTS
from interviewer.ai.piper_tts import PiperTTS
from interviewer.ai.whisper_stt import FasterWhisperSTT
from tests.helpers import silence


class FakeWhisperModel:
    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return (SimpleNamespace(text=t) for t in self.texts), SimpleNamespace(language="en")


class FakeKokoroEngine:
    def __init__(self):
        self.calls = []

    def create(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.zeros(2400, dtype=np.float32), 24000


class FakePiperVoice:
    def __init__(self):
        self.config = SimpleNamespace(sample_rate=22050)

    def synthesize(self, text):
        yield SimpleNamespace(audio_int16_bytes=b"\x01\x00" * 3)
        yield SimpleNamespace(audio_int16_bytes=b"\x02\x00" * 2)


async def test_whisper_passes_english_prompt_and_resamples():
    model = FakeWhisperModel(texts=[" Hello", " world. "])
    stt = FasterWhisperSTT(model=model)
    assert await stt.transcribe(silence(1.0, 24000), 24000, prompt="C#, EF Core") == "Hello world."
    audio, kw = model.calls[0]
    assert audio.dtype == np.float32 and len(audio) == 16000
    assert kw == {"language": "en", "beam_size": 1, "initial_prompt": "C#, EF Core",
                  "condition_on_previous_text": False, "vad_filter": False, "without_timestamps": True}


async def test_whisper_empty_audio_returns_empty_text():
    model = FakeWhisperModel(texts=["never"])
    assert await FasterWhisperSTT(model=model).transcribe(b"", 16000) == "" and model.calls == []


async def test_whisper_loads_lazily_on_cpu_int8(monkeypatch, tmp_path):
    created = []
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **k: created.append((a, k)) or FakeWhisperModel([]))
    stt = FasterWhisperSTT(model_size="small", cpu_threads=4, download_root=tmp_path)
    assert created == []
    await stt.warm_up()
    assert created == [(("small",), {"device": "cpu", "compute_type": "int8", "cpu_threads": 4, "download_root": str(tmp_path)})]


async def test_kokoro_returns_pcm16_at_24k():
    engine = FakeKokoroEngine()
    tts = KokoroTTS(Path("m.onnx"), Path("v.bin"), engine=engine)
    pcm = await tts.synthesize("Hello there.")
    assert len(pcm) == 4800 and tts.sample_rate == 24000 and tts.voice_id == "kokoro:af_heart"
    assert engine.calls == [("Hello there.", {"voice": "af_heart", "speed": 1.0, "lang": "en-us"})]


async def test_kokoro_loads_lazily(monkeypatch):
    created = []
    import kokoro_onnx

    monkeypatch.setattr(kokoro_onnx, "Kokoro", lambda *a: created.append(a) or FakeKokoroEngine())
    tts = KokoroTTS(Path("/m/k.onnx"), Path("/m/v.bin"), voice="bf_emma")
    assert created == []
    await tts.warm_up()
    assert created == [("/m/k.onnx", "/m/v.bin")] and tts.voice_id == "kokoro:bf_emma"


async def test_piper_concatenates_chunks():
    tts = PiperTTS(Path("/models/piper/en_US-lessac-medium.onnx"), voice=FakePiperVoice())
    assert await tts.synthesize("Hi") == b"\x01\x00" * 3 + b"\x02\x00" * 2
    assert tts.sample_rate == 22050 and tts.voice_id == "piper:en_US-lessac-medium"
