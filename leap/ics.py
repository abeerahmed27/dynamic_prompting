"""
Input completeness scoring for the dynamic prompting pipeline.

The scorer is still heuristic, but it now:
1. Uses prior conversation context to avoid asking repeat questions.
2. Makes S2 overrides rarer by requiring multiple genuinely missing dimensions.
3. Gives partial credit when the user is clearly answering a prior clarify turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


_EMOTION_WORDS = re.compile(
    r"\b(angry|upset|hurt|sad|scared|nervous|anxious|confused|frustrated|"
    r"annoyed|embarrassed|ignored|betrayed|lonely|hate|love|miss|afraid|"
    r"terrible|awful|horrible|devastated|lost|broken|fine|okay|whatever|"
    r"feel\w*|felt|feeling|emotion|cry|cried|crying)\b",
    re.IGNORECASE,
)

_CONTEXT_SIGNALS = re.compile(
    r"\b(she|he|they|her|him|them|my|our|friend|best\s*friend|sister|brother|"
    r"mum|mom|dad|teacher|boss|partner|boyfriend|girlfriend|group|chat|school|"
    r"work|class|team|today|yesterday|last\s*\w+|this\s*\w+|week|month)\b",
    re.IGNORECASE,
)

_CONFLICT_SIGNALS = re.compile(
    r"\b(said|told|did|happened|ignored|blocked|posted|sent|texted|called|"
    r"yelled|shouted|left|ghosted|excluded|laughed|made\s+fun|argument|fight|"
    r"disagreement|drama|problem|issue|situation|conflict)\b",
    re.IGNORECASE,
)

_OUTCOME_SIGNALS = re.compile(
    r"\b(what\s+should|how\s+(do|can|should)\s+i|help\s+me|don'?t\s+know\s+what|"
    r"want\s+to|need\s+to|trying\s+to|should\s+i|can\s+i|advice|tip|suggestion|"
    r"fix|solve|deal\s+with|handle|approach|respond|reply|say)\b",
    re.IGNORECASE,
)

_SCAFFOLD_QUESTIONS: dict[str, str] = {
    "emotional_signal": "How are you feeling about this right now?",
    "context_richness": "Who's involved in this? A quick bit of context helps.",
    "conflict_detail": "What actually happened? Even a short version helps.",
    "desired_outcome": "What would feel like the best outcome here for you?",
    "message_substance": "Tell me a bit more so I can respond properly.",
}


@dataclass
class ICSResult:
    score: int
    tier: str
    breakdown: dict[str, int]
    missing_dims: list[str]
    scaffold_question: str
    should_override_stage: bool
    answered_by_context: list[str] = field(default_factory=list)


class InputCompletenessScorer:
    def score(
        self,
        text: str,
        conversation_context: str = "",
        known_entities: Optional[list[str]] = None,
        last_missing_dims: Optional[list[str]] = None,
    ) -> ICSResult:
        known_entities = known_entities or []
        last_missing_dims = last_missing_dims or []
        pts: dict[str, int] = {
            "emotional_signal": self._score_emotional_signal(text),
            "context_richness": self._score_context(text),
            "conflict_detail": self._score_conflict(text),
            "desired_outcome": self._score_outcome(text),
            "message_substance": self._score_substance(text),
        }
        answered_by_context: list[str] = []

        if conversation_context:
            if pts["context_richness"] == 0 and (
                self._score_context(conversation_context) > 0 or known_entities
            ):
                pts["context_richness"] = 8
                answered_by_context.append("context_richness")
            if pts["conflict_detail"] == 0 and self._score_conflict(conversation_context) > 0:
                pts["conflict_detail"] = 8
                answered_by_context.append("conflict_detail")
            if pts["desired_outcome"] == 0 and self._score_outcome(conversation_context) > 0:
                pts["desired_outcome"] = 8
                answered_by_context.append("desired_outcome")

        if last_missing_dims and len(text.split()) >= 5:
            for dim in last_missing_dims:
                if dim in answered_by_context and pts[dim] == 8:
                    pts[dim] = 10
            pts["message_substance"] = max(pts["message_substance"], 10)

        total = sum(pts.values())
        missing = [dim for dim, p in pts.items() if p == 0]

        if total >= 70:
            tier = "A"
        elif total >= 50:
            tier = "B"
        else:
            tier = "C"

        scaffold_q = ""
        for dim in [
            "conflict_detail",
            "context_richness",
            "emotional_signal",
            "desired_outcome",
            "message_substance",
        ]:
            if dim in missing:
                scaffold_q = _SCAFFOLD_QUESTIONS[dim]
                break

        return ICSResult(
            score=total,
            tier=tier,
            breakdown=pts,
            missing_dims=missing,
            scaffold_question=scaffold_q,
            should_override_stage=(tier == "C" and len(missing) >= 2),
            answered_by_context=answered_by_context,
        )

    @staticmethod
    def _score_emotional_signal(text: str) -> int:
        matches = len(_EMOTION_WORDS.findall(text))
        if matches >= 3:
            return 20
        if matches == 2:
            return 15
        if matches == 1:
            return 10
        if len(text.split()) <= 6:
            return 8
        return 0

    @staticmethod
    def _score_context(text: str) -> int:
        matches = len(_CONTEXT_SIGNALS.findall(text))
        if matches >= 4:
            return 20
        if matches >= 2:
            return 14
        if matches == 1:
            return 8
        return 0

    @staticmethod
    def _score_conflict(text: str) -> int:
        matches = len(_CONFLICT_SIGNALS.findall(text))
        if matches >= 3:
            return 20
        if matches >= 1:
            return 12
        return 0

    @staticmethod
    def _score_outcome(text: str) -> int:
        if _OUTCOME_SIGNALS.search(text):
            return 20
        if "?" in text:
            return 10
        return 0

    @staticmethod
    def _score_substance(text: str) -> int:
        word_count = len(text.split())
        if word_count >= 30:
            return 20
        if word_count >= 15:
            return 14
        if word_count >= 8:
            return 10
        if word_count >= 4:
            return 6
        return 2
