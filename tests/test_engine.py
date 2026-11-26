from interviewer.ai.base import LLMError, LLMTimeout
from interviewer.ai.fakes import FakeLLM
from interviewer.engine.engine import (
    ANYTHING_ELSE,
    CANDIDATE_QUESTIONS_PROMPT,
    CLARIFY_ASSUMPTIONS,
    CONSENT_DECLINED_LINE,
    DEFER_TO_HR,
    DIDNT_CATCH,
    MOVE_ON,
    NUDGE_GENTLE,
    NUDGE_TAKE_TIME,
    STEER_BACK,
    WANTS_HUMAN_LINE,
    End,
    InterviewEngine,
    Say,
    Stage,
    generate_experience_questions,
)
from tests.helpers import FakeClock

ANSWER = "I build payment APIs in .NET for a fintech team"


def decision(label, idx=None, custom=None, note="n"):
    return {"label": label, "evidence_note": note, "follow_up_index": idx, "custom_follow_up": custom}


async def on_technical(kit, responses, clock=None):
    """Engine on csharp_runtime's main question, with ``responses`` queued for the LLM."""
    llm = FakeLLM([decision("answered")] * 3 + list(responses))
    engine = InterviewEngine(kit, llm, clock=clock or FakeClock())
    engine.start()
    await engine.on_candidate_turn("Yes, that's fine")
    for _ in range(3):
        await engine.on_candidate_turn(ANSWER)
    assert engine.stage == Stage.TECHNICAL and engine.current_topic_id == "csharp_runtime"
    return engine, llm


async def on_candidate_questions(kit, responses):
    engine, llm = await on_technical(kit, [decision("answered")] * 3 + list(responses))
    for _ in range(2):
        await engine.on_candidate_turn(ANSWER)
    reply = await engine.on_candidate_turn(ANSWER)
    assert reply == [Say(CANDIDATE_QUESTIONS_PROMPT, "script")] and engine.stage == Stage.CANDIDATE_QUESTIONS
    return engine, llm


def follow_ups(kit, topic_id="csharp_runtime"):
    return kit.topic(topic_id).questions[0].follow_ups


def main_question(kit, topic_id):
    return kit.topic(topic_id).questions[0].ask


def test_start_says_intro(kit):
    assert InterviewEngine(kit, FakeLLM([])).start() == [Say(kit.intro_script, "script")]


async def test_consent_accepted_moves_to_warmup(kit):
    e = InterviewEngine(kit, llm := FakeLLM([]))
    e.start()
    assert await e.on_candidate_turn("Yes, that's fine") == [Say(kit.warmup_questions[0], "question")]
    assert e.stage == Stage.WARMUP and llm.calls == []


async def test_consent_declined_ends_and_flags(kit):
    e = InterviewEngine(kit, FakeLLM([]))
    e.start()
    assert await e.on_candidate_turn("No, I'm not comfortable with recording") == [
        Say(CONSENT_DECLINED_LINE, "script"), End("consent_declined"),
    ]
    assert "consent_declined" in e.flags and e.stage == Stage.ENDED


async def test_intro_noise_waits_for_a_real_answer(kit):
    e = InterviewEngine(kit, FakeLLM([]))
    e.start()
    assert await e.on_candidate_turn("um") == [] and e.stage == Stage.INTRO


async def test_shallow_uses_llm_chosen_follow_up(kit):
    e, _ = await on_technical(kit, [decision("shallow", 1)])
    assert await e.on_candidate_turn(ANSWER) == [Say(follow_ups(kit)[1], "follow_up", "csharp_runtime")]


async def test_max_two_follow_ups_then_next_topic(kit):
    e, _ = await on_technical(kit, [decision("shallow", 0), decision("shallow", 1), decision("shallow", 2)])
    await e.on_candidate_turn(ANSWER)
    await e.on_candidate_turn(ANSWER)
    assert await e.on_candidate_turn(ANSWER) == [Say(main_question(kit, "aspnet_api"), "question", "aspnet_api")]


async def test_custom_follow_up_used_when_safe(kit):
    e, _ = await on_technical(kit, [decision("shallow", None, "How would you cancel that work cleanly?")])
    assert await e.on_candidate_turn(ANSWER) == [
        Say("How would you cancel that work cleanly?", "follow_up", "csharp_runtime"),
    ]


async def test_custom_follow_up_rejected_when_unsafe(kit):
    e, _ = await on_technical(kit, [decision("shallow", None, "Are you married?")])
    assert await e.on_candidate_turn(ANSWER) == [Say(follow_ups(kit)[0], "follow_up", "csharp_runtime")]


async def test_repeat_replays_last_question_without_llm(kit):
    e, llm = await on_technical(kit, [])
    calls = len(llm.calls)
    assert await e.on_candidate_turn("Sorry, can you repeat that?") == [
        Say(main_question(kit, "csharp_runtime"), "repeat", "csharp_runtime"),
    ]
    assert len(llm.calls) == calls


async def test_wants_human_ends_and_flags(kit):
    e, llm = await on_technical(kit, [])
    calls = len(llm.calls)
    assert await e.on_candidate_turn("Can I talk to a real person?") == [Say(WANTS_HUMAN_LINE, "script"), End("wants_human")]
    assert "wants_human" in e.flags and len(llm.calls) == calls


async def test_wants_human_label_from_llm_ends(kit):
    e, _ = await on_technical(kit, [decision("wants_human")])
    long_request = "Honestly I would feel much more comfortable doing this conversation with somebody from your team"
    assert (await e.on_candidate_turn(long_request))[-1] == End("wants_human")


async def test_asked_question_stays_on_question(kit):
    e, _ = await on_technical(kit, [decision("asked_question"), decision("shallow", 0)])
    assert await e.on_candidate_turn("Do you mean in ASP.NET Core or a console app?") == [
        Say(CLARIFY_ASSUMPTIONS, "clarify", "csharp_runtime"),
    ]
    assert e.current_topic_id == "csharp_runtime"
    assert await e.on_candidate_turn(ANSWER) == [Say(follow_ups(kit)[0], "follow_up", "csharp_runtime")]


async def test_off_topic_twice_moves_on(kit):
    e, _ = await on_technical(kit, [decision("off_topic"), decision("off_topic")])
    assert await e.on_candidate_turn(ANSWER) == [
        Say(STEER_BACK, "clarify", "csharp_runtime"),
        Say(main_question(kit, "csharp_runtime"), "repeat", "csharp_runtime"),
    ]
    assert await e.on_candidate_turn(ANSWER) == [Say(main_question(kit, "aspnet_api"), "question", "aspnet_api")]


async def test_not_meaningful_twice_asks_to_repeat(kit):
    e, llm = await on_technical(kit, [])
    calls = len(llm.calls)
    assert await e.on_candidate_turn("um") == []
    assert await e.on_candidate_turn("uh") == [Say(DIDNT_CATCH, "clarify", "csharp_runtime")]
    assert len(llm.calls) == calls


async def test_llm_timeout_falls_back(kit):
    e, _ = await on_technical(kit, [LLMTimeout("t"), LLMTimeout("t")])
    assert (await e.on_candidate_turn("I would use a cache"))[0].kind == "follow_up"
    long = "I would put a distributed cache in front of the database and invalidate on writes using events from the outbox"
    assert (await e.on_candidate_turn(long))[0].topic_id == "aspnet_api"
    assert e.events.count("llm_fallback:timeout") == 2


async def test_llm_malformed_label_falls_back(kit):
    e, _ = await on_technical(kit, [{"label": "great"}])
    reply = await e.on_candidate_turn("I would use a cache")
    assert reply[0].kind == "follow_up" and "llm_fallback:invalid" in e.events


async def test_llm_error_falls_back(kit):
    e, _ = await on_technical(kit, [LLMError("boom")])
    await e.on_candidate_turn("I would use a cache")
    assert "llm_fallback:error" in e.events


async def test_topic_budget_skips_follow_ups(kit):
    clock = FakeClock()
    e, _ = await on_technical(kit, [decision("shallow", 0)], clock=clock)
    clock.now += 6.1 * 60
    assert await e.on_candidate_turn(ANSWER) == [Say(main_question(kit, "aspnet_api"), "question", "aspnet_api")]


async def test_technical_budget_skips_remaining_topics(kit):
    clock = FakeClock()
    e, _ = await on_technical(kit, [decision("answered")], clock=clock)
    clock.now += 18.1 * 60
    assert await e.on_candidate_turn(ANSWER) == [Say(CANDIDATE_QUESTIONS_PROMPT, "script")]


async def test_overall_time_up_closes(kit):
    clock = FakeClock()
    e, _ = await on_technical(kit, [], clock=clock)
    clock.now += 30.1 * 60
    assert await e.on_candidate_turn(ANSWER) == [Say(kit.closing_script, "script"), End("time_up")]
    assert e.stage == Stage.ENDED


async def test_candidate_questions_faq_defer_and_done(kit):
    e, _ = await on_candidate_questions(kit, [
        {"faq_index": 0, "defer": False, "no_more_questions": False},
        {"faq_index": None, "defer": True, "no_more_questions": False},
        {"faq_index": None, "defer": False, "no_more_questions": True},
    ])
    assert await e.on_candidate_turn("What happens after this?") == [
        Say(kit.candidate_faq[0].a, "faq"), Say(ANYTHING_ELSE, "script"),
    ]
    assert await e.on_candidate_turn("What's the salary range for this role?") == [
        Say(DEFER_TO_HR, "defer"), Say(ANYTHING_ELSE, "script"),
    ]
    assert "What's the salary range for this role?" in e.evidence["candidate_questions"]
    assert await e.on_candidate_turn("Honestly I believe you have covered everything I wanted to know today") == [
        Say(kit.closing_script, "script"), End("completed"),
    ]


async def test_short_no_more_questions_closes_without_llm(kit):
    e, llm = await on_candidate_questions(kit, [])
    calls = len(llm.calls)
    assert await e.on_candidate_turn("No, that's all, thanks") == [Say(kit.closing_script, "script"), End("completed")]
    assert len(llm.calls) == calls


async def test_silence_nudges(kit):
    e, _ = await on_technical(kit, [])
    assert e.on_silence(10) == [Say(NUDGE_GENTLE, "nudge", "csharp_runtime")]
    assert e.on_silence(30) == [Say(NUDGE_TAKE_TIME, "nudge", "csharp_runtime")]
    assert e.on_silence(90) == [Say(MOVE_ON, "nudge", "csharp_runtime"), Say(main_question(kit, "aspnet_api"), "question", "aspnet_api")]


def test_no_move_on_during_intro(kit):
    e = InterviewEngine(kit, FakeLLM([]))
    e.start()
    assert e.on_silence(90) == []


async def test_evidence_recorded_per_topic(kit):
    e, _ = await on_technical(kit, [decision("answered", note="Explained continuations clearly")])
    await e.on_candidate_turn(ANSWER)
    assert e.evidence["csharp_runtime"] == ["Explained continuations clearly"]
    assert e.evidence["experience"] == ["n", "n", "n"]


def test_planned_lines_cover_kit(kit):
    lines = set(InterviewEngine(kit, FakeLLM([])).planned_lines())
    expected = {kit.intro_script, kit.closing_script, kit.warmup_questions[0], *kit.experience_questions[:2],
                *kit.acknowledgments, *(f.a for f in kit.candidate_faq),
                NUDGE_GENTLE, NUDGE_TAKE_TIME, MOVE_ON, DIDNT_CATCH, CLARIFY_ASSUMPTIONS, STEER_BACK,
                CANDIDATE_QUESTIONS_PROMPT, ANYTHING_ELSE, DEFER_TO_HR, CONSENT_DECLINED_LINE, WANTS_HUMAN_LINE}
    for topic_id in kit.interview_topics:
        expected |= {main_question(kit, topic_id), *follow_ups(kit, topic_id)}
    assert expected <= lines


def test_end_now(kit):
    e = InterviewEngine(kit, FakeLLM([]))
    e.start()
    assert e.end_now("ended_by_owner") == [Say(kit.closing_script, "script"), End("ended_by_owner")]
    assert e.stage == Stage.ENDED


async def test_tailored_experience_questions_are_asked(kit):
    e = InterviewEngine(kit, FakeLLM([decision("answered")]), experience_questions=["Tell me about Acme.", "And Beta?"])
    e.start()
    await e.on_candidate_turn("Yes")
    assert await e.on_candidate_turn(ANSWER) == [Say("Tell me about Acme.", "question")]
    assert "Tell me about Acme." in e.planned_lines()


async def test_generate_experience_questions_validates(kit):
    llm = FakeLLM([{"questions": ["Tell me about your .NET 8 migration at Acme.", "Are you married?"]}])
    assert await generate_experience_questions(kit, "Migrated Acme to .NET 8", llm) == [
        "Tell me about your .NET 8 migration at Acme.", kit.experience_questions[1],
    ]
    assert await generate_experience_questions(kit, "notes", FakeLLM([LLMError("x")])) == kit.experience_questions[:2]
