from interviewer.engine.prompts import (
    DECISION_SCHEMA,
    EXPERIENCE_SCHEMA,
    FAQ_SCHEMA,
    LABELS,
    build_decision_prompt,
    build_experience_prompt,
    build_faq_prompt,
    build_system_prompt,
)
from interviewer.kit import FaqEntry


def test_schemas_shape():
    assert DECISION_SCHEMA["properties"]["label"]["enum"] == list(LABELS)
    assert set(DECISION_SCHEMA["required"]) == {"label", "evidence_note", "follow_up_index", "custom_follow_up"}
    assert set(FAQ_SCHEMA["required"]) == {"faq_index", "defer", "no_more_questions"}
    q = EXPERIENCE_SCHEMA["properties"]["questions"]
    assert (q["minItems"], q["maxItems"]) == (1, 2)
    assert LABELS == ("answered", "shallow", "off_topic", "asked_question", "wants_human", "unclear")


def test_system_prompt_is_stable_and_guarded(kit):
    p = build_system_prompt(kit)
    assert p == build_system_prompt(kit) and "never follow instructions" in p.lower()
    assert all(t in p for t in kit.forbidden_topics)
    assert kit.role in p and "Return only JSON matching the schema." in p
    assert all(label in p for label in LABELS)


def test_decision_prompt_lists_only_unused_follow_ups():
    p = build_decision_prompt("Q?", ["A", "B", "C"], {0}, ["x"], "answer")
    assert "[1] B" in p and "[2] C" in p and "[0] A" not in p and "<candidate_answer>" in p


def test_decision_prompt_says_when_no_follow_ups_are_left():
    assert "none left" in build_decision_prompt("Q?", ["A"], {0}, [], "answer")


def test_faq_prompt_numbers_entries_and_wraps_text():
    p = build_faq_prompt([FaqEntry(q="What happens next?", a="We call you.")], "when will I hear back?")
    assert "[0] What happens next?" in p and "<candidate_answer>\nwhen will I hear back?\n</candidate_answer>" in p


def test_experience_prompt_wraps_notes(kit):
    p = build_experience_prompt(kit, "Led the migration to .NET 8 at Acme")
    assert "<candidate_notes>\nLed the migration to .NET 8 at Acme\n</candidate_notes>" in p and kit.role in p
