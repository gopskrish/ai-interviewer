import wave

import numpy as np

from interviewer.audio.pcm import (
    WavAppender,
    chunk_bytes,
    duration_s,
    float32_to_pcm16,
    pcm16_to_float32,
    resample_linear,
)
from tests.helpers import silence


def test_pcm_roundtrip():
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    assert np.allclose(pcm16_to_float32(float32_to_pcm16(x)), x, atol=1 / 32767)


def test_float32_to_pcm16_clips():
    out = np.frombuffer(float32_to_pcm16(np.array([2.0, -2.0], np.float32)), "<i2")
    assert out.tolist() == [32767, -32767]


def test_resample_16k_to_24k_length():
    assert len(resample_linear(np.zeros(1600, np.float32), 16000, 24000)) == 2400


def test_chunk_bytes_200ms_at_24k():
    chunks = chunk_bytes(silence(1.0, 24000), 24000, 200)
    assert [len(c) for c in chunks] == [9600] * 5


def test_duration_s():
    assert duration_s(silence(1.5, 24000), 24000) == 1.5


def test_wav_appender_writes_valid_wav(tmp_path):
    path = tmp_path / "out.wav"
    wav = WavAppender(path, 16000)
    wav.write(silence(0.5))
    wav.write(silence(0.5))
    wav.close()
    with wave.open(str(path), "rb") as r:
        assert (r.getnframes(), r.getframerate(), r.getnchannels()) == (16000, 16000, 1)
