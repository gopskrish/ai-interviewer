import numpy as np

from interviewer.audio.vad import BargeIn, SegmentReady, SileroVad, SpeechStarted, TurnDetector, TurnEnded
from tests.helpers import LoudnessVad, silence, tone


def of(kind, events):
    return [e for e in events if isinstance(e, kind)]


def ended(events):
    return of(TurnEnded, events)


def test_turn_ends_after_1500ms_silence_for_long_answer():
    td = TurnDetector(LoudnessVad())
    assert not ended(td.feed(tone(3.0) + silence(1.4), ai_speaking=False))
    ev = ended(td.feed(silence(0.2), ai_speaking=False))
    assert len(ev) == 1 and 2900 <= ev[0].speech_ms <= 3100


def test_short_answer_waits_3000ms():
    td = TurnDetector(LoudnessVad())
    assert not ended(td.feed(tone(1.0) + silence(2.9), ai_speaking=False))
    assert len(ended(td.feed(silence(0.2), ai_speaking=False))) == 1


def test_segment_ready_after_400ms_pause():
    events = TurnDetector(LoudnessVad()).feed(tone(2.0) + silence(0.45), ai_speaking=False)
    assert len(of(SegmentReady, events)) == 1 and not ended(events)


def test_segments_cover_the_turn_before_it_ends():
    events = TurnDetector(LoudnessVad()).feed(
        tone(1.0) + silence(0.5) + tone(1.0) + silence(1.6), ai_speaking=False
    )
    kinds = [type(e).__name__ for e in events if not isinstance(e, SpeechStarted)]
    assert kinds == ["SegmentReady", "SegmentReady", "TurnEnded"]


def test_barge_in_after_300ms_while_ai_speaking():
    td = TurnDetector(LoudnessVad())
    assert not of(BargeIn, td.feed(tone(0.25), ai_speaking=True))
    assert len(of(BargeIn, td.feed(tone(0.1), ai_speaking=True))) == 1


def test_barge_in_fires_once_per_ai_utterance():
    td = TurnDetector(LoudnessVad())
    first = of(BargeIn, td.feed(tone(1.0), ai_speaking=True))
    assert len(first) == 1


def test_no_barge_in_when_ai_silent():
    assert not of(BargeIn, TurnDetector(LoudnessVad()).feed(tone(1.0), ai_speaking=False))


def test_max_turn_forces_end():
    events = TurnDetector(LoudnessVad()).feed(tone(120.1), ai_speaking=False)
    assert len(ended(events)) == 1
    assert isinstance(events[events.index(ended(events)[0]) - 1], SegmentReady)


def test_short_blip_ignored():
    events = TurnDetector(LoudnessVad()).feed(tone(0.06) + silence(4.0), ai_speaking=False)
    assert not of(SpeechStarted, events) and not ended(events)


def test_now_s_tracks_audio_timeline():
    td = TurnDetector(LoudnessVad())
    td.feed(silence(1.0), ai_speaking=False)
    assert abs(td.now_s - 1.0) < 0.033


def test_silero_vad_silence_is_low():
    vad = SileroVad()
    assert vad.speech_probability(np.zeros(512, np.float32)) < 0.2


def test_silero_streaming_matches_faster_whisper_batch():
    import importlib.resources

    from faster_whisper.vad import SileroVADModel

    path = str(importlib.resources.files("faster_whisper") / "assets" / "silero_vad_v6.onnx")
    rng = np.random.default_rng(0)
    t = np.arange(512 * 40) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
             + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    batch = np.asarray(SileroVADModel(path)(audio)).reshape(-1)
    vad = SileroVad()
    streamed = [vad.speech_probability(audio[i * 512:(i + 1) * 512]) for i in range(40)]
    assert np.allclose(batch, streamed, atol=1e-5)
