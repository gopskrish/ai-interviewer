import pytest

from interviewer.scoring.scorecard import Quote, Scorecard, TopicScore


@pytest.fixture
def sample_scorecard():
    return Scorecard(
        session_id="abc123def456",
        candidate_name="Asha Rao",
        role="Senior .NET Developer",
        created_at="2026-11-26T10:00:00+00:00",
        topics=[
            TopicScore("csharp_runtime", "C# and the .NET runtime", 3, False,
                       [Quote("use IAsyncEnumerable for streaming", "02:05")], ["streams results"], ["no cancellation"]),
            TopicScore("aspnet_api", "ASP.NET Core and API design", 2, False, [], [], ["vague on versioning"],
                       evidence_missing=True),
            TopicScore("system_design", "System design and architecture", None, True, [], [], []),
        ],
        flags=["wants_human"],
        end_reason="completed",
    )


def test_scorecard_has_no_overall_recommendation(sample_scorecard):
    d = sample_scorecard.to_dict()
    md = sample_scorecard.to_markdown()
    assert not {"overall", "recommendation", "decision"} & set(d) and "recommend" not in md.lower()


def test_markdown_lists_scores_quotes_flags(sample_scorecard):
    md = sample_scorecard.to_markdown()
    assert "Asha Rao" in md and "C# and the .NET runtime: 3/4" in md
    assert '[02:05] "use IAsyncEnumerable for streaming"' in md
    assert "System design and architecture: Not enough signal" in md
    assert "wants_human" in md and "no cancellation" in md and "No verified quotes" in md


def test_adaptive_card_shape(sample_scorecard):
    card = sample_scorecard.to_adaptive_card()
    assert card["type"] == "AdaptiveCard" and card["version"] == "1.4"
    texts = [block.get("text", "") for block in card["body"]]
    assert any("Asha Rao" in t for t in texts)
    facts = next(b for b in card["body"] if b["type"] == "FactSet")["facts"]
    assert facts[0] == {"title": "C# and the .NET runtime", "value": "3/4"}
    assert facts[2]["value"] == "Not enough signal"


def test_scorecard_roundtrip(sample_scorecard):
    assert Scorecard.from_dict(sample_scorecard.to_dict()) == sample_scorecard
