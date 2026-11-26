from interviewer.audio.pacer import OutputPacer
from tests.helpers import FakeClock, silence


async def _record(sent, clock, chunk):
    sent.append((clock.now, chunk))


async def test_pacer_sends_ahead_by_lead():
    clock, sent = FakeClock(), []
    pacer = OutputPacer(lambda b, sr: _record(sent, clock, b), clock=clock, sleep=clock.sleep)
    assert await pacer.play(silence(1.0, 24000), 24000) is True
    assert len(sent) == 5 and [t for t, _ in sent][:2] == [0.0, 0.0]
    assert clock.now >= 1.0 and pacer.speaking is False


async def test_pacer_stop_halts_sending():
    clock, sent = FakeClock(), []
    pacer = None

    async def send(chunk, sample_rate):
        sent.append(chunk)
        if len(sent) == 2:
            pacer.stop()

    pacer = OutputPacer(send, clock=clock, sleep=clock.sleep)
    assert await pacer.play(silence(1.0, 24000), 24000) is False
    assert len(sent) <= 3


async def test_pacer_speaking_flag_during_play():
    clock, seen = FakeClock(), []
    pacer = None

    async def send(chunk, sample_rate):
        seen.append(pacer.speaking)

    pacer = OutputPacer(send, clock=clock, sleep=clock.sleep)
    assert pacer.speaking is False
    await pacer.play(silence(0.6, 24000), 24000)
    assert seen and all(seen) and pacer.speaking is False


async def test_pacer_stale_stop_does_not_cancel_next_play():
    clock, sent = FakeClock(), []
    pacer = OutputPacer(lambda b, sr: _record(sent, clock, b), clock=clock, sleep=clock.sleep)
    pacer.stop()
    assert await pacer.play(silence(0.4, 24000), 24000) is True
    assert len(sent) == 2
