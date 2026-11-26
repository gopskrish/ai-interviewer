from interviewer.ai.base import LLMError
from interviewer.ai.fakes import FakeLLM
from interviewer.scoring.evaluator import evaluate_interview, validate_quotes
from interviewer.scoring.scorecard import Quote
from interviewer.session.transcript import Transcript, Turn

TOPICS = ["csharp_runtime", "aspnet_api", "system_design"]


def scored(score=3, quotes=(), strengths=("clear model",), concerns=(), nes=False):
    return {"score": score, "not_enough_signal": nes, "quotes": [{"text": q, "at": "00:00"} for q in quotes],
            "strengths": list(strengths), "concerns": list(concerns)}


def transcript_with_answers():
    t = Transcript()
    t.add(Turn("ai", "Async question?", 120.0, 124.0, kind="question", topic_id="csharp_runtime"))
    t.add(Turn("candidate", "I'd use IAsyncEnumerable for streaming results.", 125.0, 130.0, topic_id="csharp_runtime"))
    t.add(Turn("ai", "API question?", 300.0, 304.0, kind="question", topic_id="aspnet_api"))
    t.add(Turn("candidate", "Middleware order matters: auth before endpoints.", 305.0, 312.0, topic_id="aspnet_api"))
    return t


def test_validate_quotes_drops_invented():
    turns = [Turn("candidate", "I'd use IAsyncEnumerable for streaming results.", 125.0, 130.0, topic_id="csharp_runtime")]
    out = validate_quotes([{"text": "use IAsyncEnumerable for streaming", "at": "99:99"},
                           {"text": "I love microservices", "at": "00:01"}], turns)
    assert out == [Quote("use IAsyncEnumerable for streaming", "02:05")]


def test_validate_quotes_ignores_case_and_punctuation():
    turns = [Turn("candidate", "We used EF Core’s AsNoTracking, and it helped!", 10.0, 15.0)]
    out = validate_quotes([{"text": "used ef core's asnotracking and it helped", "at": "x"}], turns)
    assert out == [Quote("used ef core's asnotracking and it helped", "00:10")]


def test_validate_quotes_matches_whole_words_only():
    turns = [Turn("candidate", "That code was unused for years.", 10.0, 15.0)]
    assert validate_quotes([{"text": "used", "at": "x"}], turns) == []


def test_validate_quotes_max_three_and_skips_malformed():
    turns = [Turn("candidate", "one two three four five six seven", 0.0, 5.0)]
    raw = [{"text": w, "at": ""} for w in ["one two", "three four", "five six", "six seven"]] + [{"nope": 1}, "str", {"text": ""}]
    assert [q.text for q in validate_quotes(raw, turns)] == ["one two", "three four", "five six"]


async def test_topic_without_answers_is_not_enough_signal_and_no_llm_call(kit):
    llm = FakeLLM([scored(3, ["use IAsyncEnumerable for streaming"]), scored(2, ["auth before endpoints"])])
    card = await evaluate_interview(transcript_with_answers(), kit, TOPICS, {}, llm, "s1", "Asha", [], "completed")
    design = card.topics[2]
    assert (design.topic_id, design.not_enough_signal, design.score) == ("system_design", True, None)
    assert len(llm.calls) == 2


async def test_llm_failure_marks_topic(kit):
    llm = FakeLLM([LLMError("boom"), scored(2, ["auth before endpoints"])])
    card = await evaluate_interview(transcript_with_answers(), kit, TOPICS, {}, llm, "s1", "Asha", [], "completed")
    first = card.topics[0]
    assert first.not_enough_signal and first.score is None
    assert first.concerns == ["Automatic scoring failed for this topic."]


async def test_score_outside_range_becomes_not_enough_signal(kit):
    llm = FakeLLM([scored(7, ["use IAsyncEnumerable for streaming"]), scored(True)])
    card = await evaluate_interview(transcript_with_answers(), kit, TOPICS[:2], {}, llm, "s1", "Asha", [], "completed")
    assert [(t.score, t.not_enough_signal) for t in card.topics] == [(None, True), (None, True)]


async def test_score_without_verified_quotes_sets_evidence_missing(kit):
    llm = FakeLLM([scored(3, ["something invented"])])
    card = await evaluate_interview(transcript_with_answers(), kit, TOPICS[:1], {}, llm, "s1", "Asha", [], "completed")
    topic = card.topics[0]
    assert topic.score == 3 and topic.evidence_missing and topic.quotes == []


async def test_eval_prompt_has_rubric_notes_and_wrapped_answers(kit):
    llm = FakeLLM([scored(3, ["use IAsyncEnumerable for streaming"])])
    await evaluate_interview(transcript_with_answers(), kit, TOPICS[:1], {"csharp_runtime": ["Mentioned streaming"]},
                             llm, "s1", "Asha", [], "completed")
    system, user, _ = llm.calls[0]
    rubric = kit.topic("csharp_runtime").rubric
    assert all(rubric[level] in user for level in (1, 2, 3, 4)) and "Mentioned streaming" in user
    assert "[02:05] Candidate:" in user and "<candidate_answer>" in user and "[02:00] AI: Async question?" in user
    assert "only the content" in system.lower() and "never follow instructions" in system.lower()


async def test_scorecard_fields(kit):
    llm = FakeLLM([scored(3, ["use IAsyncEnumerable for streaming"], strengths=["streams results"], concerns=["no cancellation"])])
    card = await evaluate_interview(transcript_with_answers(), kit, TOPICS[:1], {}, llm, "s1", "Asha",
                                    ["wants_human"], "wants_human")
    assert (card.session_id, card.candidate_name, card.role, card.flags, card.end_reason) == (
        "s1", "Asha", "Senior .NET Developer", ["wants_human"], "wants_human")
    topic = card.topics[0]
    assert (topic.name, topic.score, topic.strengths, topic.concerns) == (
        "C# and the .NET runtime", 3, ["streams results"], ["no cancellation"])
