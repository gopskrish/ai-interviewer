"""Fast, deterministic checks on what the candidate said and what the AI may say.

The quick checks for "repeat that" and "talk to a person" only fire on
short replies: in a long answer those words are usually part of the story
("we repeat the call with backoff"), and the language model's label covers
the rest.
"""

from __future__ import annotations

import re

FILLERS = {"um", "uh", "umm", "uhh", "hmm", "hm", "mm", "er", "erm", "ah"}

OUTCOME_WORDS = {
    "score", "scored", "rating", "hire", "hired", "reject", "rejected", "pass", "passed",
    "fail", "failed", "offer", "salary", "compensation", "ctc",
}

SHORT_REQUEST_WORDS = 12
MAX_SPOKEN_WORDS = 30

_REPEAT = re.compile(
    r"\b(repeat|say (that|it) again|come again|pardon|didn'?t (catch|get|hear)|one more time)\b"
)
_HUMAN = re.compile(
    r"\b(talk|speak|chat)\s+(to|with)\s+(a\s+|an\s+|some\s+)?(real\s+|actual\s+)?(human|person|interviewer)\b"
    r"|\b(real|actual)\s+(person|human)\b"
    r"|\bhuman interviewer\b"
)
_LEADING_NO = re.compile(r"^\s*(no|nope|nah)\b(?!\s*,?\s*(problems?|worries|worry|issues?)\b)")
_NEGATIVE_PHRASES = (
    "not ok", "not okay", "rather not", "don't want", "do not want",
    "not comfortable", "don't agree", "do not agree",
)
_POSITIVE_MARKERS = ("fine", "go ahead", "sure", "that's ok", "that's okay", "it's ok", "it's okay")


def _normalize(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").lower().strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[\w#+']+", _normalize(text))


def is_meaningful(text: str) -> bool:
    """False for silence, fillers ("um", "hmm") and single characters."""
    real = [w for w in _words(text) if w not in FILLERS]
    return len("".join(real)) >= 2


def wants_repeat(text: str) -> bool:
    return len(_words(text)) <= SHORT_REQUEST_WORDS and bool(_REPEAT.search(_normalize(text)))


def wants_human(text: str) -> bool:
    return len(_words(text)) <= SHORT_REQUEST_WORDS and bool(_HUMAN.search(_normalize(text)))


def consent_declined(text: str) -> bool:
    """Did the reply to "Is that OK with you?" say no?"""
    t = _normalize(text)
    if any(phrase in t for phrase in _NEGATIVE_PHRASES):
        return True
    if _LEADING_NO.search(t):
        return not any(marker in t for marker in _POSITIVE_MARKERS)
    return False


def is_safe_to_say(text: str, forbidden_topics: list[str]) -> bool:
    """May the AI say this model-written line out loud?"""
    if len(text.split()) > MAX_SPOKEN_WORDS:
        return False
    lowered = _normalize(text)
    if any(word in OUTCOME_WORDS for word in _words(text)):
        return False
    for term in forbidden_topics:
        if re.search(rf"\b{re.escape(term.lower())}\b", lowered):
            return False
    return True


def wrap_untrusted(text: str, tag: str) -> str:
    """Put untrusted text inside <tag> markers, removing any copies of the markers from the text."""
    clean = re.sub(rf"</?\s*{re.escape(tag)}\s*>", "", text, flags=re.IGNORECASE).strip()
    return f"<{tag}>\n{clean}\n</{tag}>"


def wrap_candidate_text(text: str) -> str:
    return wrap_untrusted(text, "candidate_answer")
