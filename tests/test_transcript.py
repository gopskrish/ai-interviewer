import json

from interviewer.session.transcript import Transcript, Turn, format_ts


def test_format_ts():
    assert (format_ts(65.2), format_ts(3600), format_ts(0), format_ts(-1)) == ("01:05", "60:00", "00:00", "00:00")


def test_transcript_text_format():
    t = Transcript()
    t.add(Turn("ai", "Hello", 5.0, 6.0, kind="script", interrupted=True))
    t.add(Turn("candidate", "Hi", 65.2, 66.0))
    assert t.to_text() == "[00:05] AI: Hello (interrupted)\n[01:05] Candidate: Hi"


def test_transcript_roundtrip_json(tmp_path):
    t = Transcript()
    t.add(Turn("ai", "Q?", 1.0, 2.0, kind="question", topic_id="aspnet_api"))
    t.add(Turn("candidate", "A.", 3.0, 9.5, topic_id="aspnet_api"))
    t.save(tmp_path)
    loaded = Transcript.from_dict(json.loads((tmp_path / "transcript.json").read_text()))
    assert loaded.turns == t.turns
    assert (tmp_path / "transcript.txt").read_text() == t.to_text()


def test_candidate_turns_filtered_by_topic():
    t = Transcript()
    t.add(Turn("candidate", "warm-up", 1, 2))
    t.add(Turn("ai", "Q", 2, 3, topic_id="x"))
    t.add(Turn("candidate", "about x", 3, 4, topic_id="x"))
    assert [u.text for u in t.candidate_turns()] == ["warm-up", "about x"]
    assert [u.text for u in t.candidate_turns("x")] == ["about x"]
