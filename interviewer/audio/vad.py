"""Voice activity detection and turn-taking.

Audio arrives as 16 kHz PCM16 in pieces of any size. It is cut into
512-sample frames (32 ms). Each frame gets a speech probability from the
VAD, and the detector turns that into conversation events:

- ``SpeechStarted``: someone started talking (after ``min_speech_ms``).
- ``SegmentReady``: a short pause; the audio since the last pause can be
  transcribed now, so little is left to do when the turn ends. Every turn's
  audio is delivered in ``SegmentReady`` events before its ``TurnEnded``.
- ``TurnEnded``: the speaker has finished. People who have only just
  started answering often pause to think, so a turn with little speech so
  far waits longer before it counts as finished.
- ``BargeIn``: the candidate has talked over the AI for ``barge_in_ms``.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from interviewer.audio.pcm import pcm16_to_float32

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE
FRAME_MS = FRAME_S * 1000


class VadModel(Protocol):
    def speech_probability(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class SileroVad:
    """Silero VAD v6 (the ONNX file shipped with faster-whisper), run frame by frame."""

    FRAME_SAMPLES = FRAME_SAMPLES
    CONTEXT_SAMPLES = 64

    def __init__(self, model_path: Path | None = None):
        import onnxruntime

        if model_path is None:
            model_path = importlib.resources.files("faster_whisper") / "assets" / "silero_vad_v6.onnx"
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=options
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)

    def speech_probability(self, frame: np.ndarray) -> float:
        frame = np.asarray(frame, dtype=np.float32).reshape(1, FRAME_SAMPLES)
        batch = np.concatenate([self._context, frame], axis=1)
        probs, self._h, self._c = self._session.run(None, {"input": batch, "h": self._h, "c": self._c})
        self._context = frame[:, -self.CONTEXT_SAMPLES :]
        return float(np.asarray(probs).reshape(-1)[0])


@dataclass(frozen=True)
class TurnConfig:
    speech_threshold: float = 0.5
    min_speech_ms: int = 96
    segment_silence_ms: int = 400
    end_of_turn_silence_ms: int = 1500
    short_answer_speech_ms: int = 2000
    short_answer_end_of_turn_ms: int = 3000
    barge_in_ms: int = 300
    max_turn_s: float = 120.0


@dataclass(frozen=True)
class SpeechStarted:
    at_s: float


@dataclass(frozen=True)
class SegmentReady:
    pcm: bytes
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TurnEnded:
    pcm: bytes
    speech_ms: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class BargeIn:
    at_s: float


Event = SpeechStarted | SegmentReady | TurnEnded | BargeIn


class TurnDetector:
    def __init__(self, vad: VadModel, config: TurnConfig | None = None):
        self._vad = vad
        self._config = config or TurnConfig()
        self._pending = b""
        self._frames = 0
        self._ai_was_speaking = False
        self._barge_frames = 0
        self._barge_fired = False
        self._reset_turn()

    @property
    def now_s(self) -> float:
        return self._frames * FRAME_S

    @property
    def in_speech(self) -> bool:
        return self._in_turn

    def reset(self) -> None:
        self._pending = b""
        self._vad.reset()
        self._barge_frames = 0
        self._barge_fired = False
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._in_turn = False
        self._run_frames: list[bytes] = []
        self._turn_audio: list[bytes] = []
        self._turn_start_s = 0.0
        self._speech_frames = 0
        self._silence_frames = 0
        self._segment_audio: list[bytes] = []
        self._segment_start_s = 0.0
        self._segment_speech_frames = 0

    def feed(self, pcm16: bytes, ai_speaking: bool) -> list[Event]:
        events: list[Event] = []
        data = self._pending + pcm16
        usable = len(data) - (len(data) % FRAME_BYTES)
        self._pending = data[usable:]
        for offset in range(0, usable, FRAME_BYTES):
            frame = data[offset : offset + FRAME_BYTES]
            is_speech = self._vad.speech_probability(pcm16_to_float32(frame)) >= self._config.speech_threshold
            self._frames += 1
            self._track_barge_in(is_speech, ai_speaking, events)
            self._track_turn(frame, is_speech, events)
        return events

    def _track_barge_in(self, is_speech: bool, ai_speaking: bool, events: list[Event]) -> None:
        if not ai_speaking:
            self._barge_frames = 0
            self._barge_fired = False
            return
        self._barge_frames = self._barge_frames + 1 if is_speech else 0
        if not self._barge_fired and self._barge_frames * FRAME_MS >= self._config.barge_in_ms:
            self._barge_fired = True
            events.append(BargeIn(at_s=self.now_s))

    def _track_turn(self, frame: bytes, is_speech: bool, events: list[Event]) -> None:
        cfg = self._config
        if not self._in_turn:
            if not is_speech:
                self._run_frames = []
                return
            self._run_frames.append(frame)
            if len(self._run_frames) * FRAME_MS < cfg.min_speech_ms:
                return
            self._in_turn = True
            start = self.now_s - len(self._run_frames) * FRAME_S
            self._turn_start_s = self._segment_start_s = start
            self._turn_audio = list(self._run_frames)
            self._segment_audio = list(self._run_frames)
            self._speech_frames = self._segment_speech_frames = len(self._run_frames)
            self._silence_frames = 0
            self._run_frames = []
            events.append(SpeechStarted(at_s=start))
            return

        self._turn_audio.append(frame)
        self._segment_audio.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._segment_speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        silence_ms = self._silence_frames * FRAME_MS
        if self._segment_speech_frames and silence_ms >= cfg.segment_silence_ms:
            self._flush_segment(events)

        speech_ms = self._speech_frames * FRAME_MS
        needed_ms = cfg.short_answer_end_of_turn_ms if speech_ms < cfg.short_answer_speech_ms else cfg.end_of_turn_silence_ms
        if silence_ms >= needed_ms:
            self._end_turn(events)
        elif self.now_s - self._turn_start_s >= cfg.max_turn_s:
            self._flush_segment(events)
            self._end_turn(events)

    def _flush_segment(self, events: list[Event]) -> None:
        if not self._segment_speech_frames:
            return
        events.append(SegmentReady(pcm=b"".join(self._segment_audio), start_s=self._segment_start_s, end_s=self.now_s))
        self._segment_audio = []
        self._segment_start_s = self.now_s
        self._segment_speech_frames = 0

    def _end_turn(self, events: list[Event]) -> None:
        events.append(
            TurnEnded(
                pcm=b"".join(self._turn_audio),
                speech_ms=int(round(self._speech_frames * FRAME_MS)),
                start_s=self._turn_start_s,
                end_s=self.now_s,
            )
        )
        self._reset_turn()
