"""
leap/ics.py — Input Completeness Score
========================================
Scores each user message across 5 dimensions before the emotion model runs.
Gates the pipeline into 3 tiers:

  Tier A  score ≥ 70  → proceed normally, full emotion + stage pipeline
  Tier B  50–69       → proceed but flag missing dims; nudge with S2 if on S2 already
  Tier C  score < 50  → override to S2 Clarify; return a targeted scaffold question

Scoring dimensions (20 pts each):
  1. emotional_signal   — does the text contain emotional language?
  2. context_richness   — who/what/when present?
  3. conflict_detail    — something happened (past-tense events, quotes)?
  4. desired_outcome    — any hint of what they want from this conversation?
  5. message_substance  — raw length/effort (proxy for how much they want help)

All heuristic — no LLM call. Runs in <1ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Word lists ──────────────────────────────────────────────────────────────────
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

# Scaffold questions — one per missing dimension
_SCAFFOLD_QUESTIONS: dict[str, str] = {
    "emotional_signal":  "How are you feeling about this right now?",
    "context_richness":  "Who's involved in this — can you tell me a bit about them?",
    "conflict_detail":   "What actually happened? Even a quick version helps.",
    "desired_outcome":   "What would feel like the best outcome here for you?",
    "message_substance": "Tell me a bit more — I want to make sure I get it.",
}


# ── Result schema ───────────────────────────────────────────────────────────────
@dataclass
class ICSResult:
    score: int                          # 0–100
    tier: str                           # "A" | "B" | "C"
    breakdown: dict[str, int]           # {dimension: pts}
    missing_dims: list[str]             # dims that scored 0
    scaffold_question: str              # best clarifying question if tier C
    should_override_stage: bool         # True if tier C → force S2


# ── Scorer ──────────────────────────────────────────────────────────────────────
class InputCompletenessScorer:
    """
    Scores a single user message and returns an ICSResult.

    Usage in pipeline:
        ics = InputCompletenessScorer()
        result = ics.score("she ignored me")
        if result.should_override_stage:
            stage = "S2"
            scaffold_q = result.scaffold_question

    Thresholds follow LEAP's original spec (70 / 50 / below 50).
    """

    def score(self, text: str) -> ICSResult:
        pts: dict[str, int] = {}

        # Dim 1 — Emotional signal
        pts["emotional_signal"] = self._score_emotional_signal(text)

        # Dim 2 — Context richness (who/where/when)
        pts["context_richness"] = self._score_context(text)

        # Dim 3 — Conflict detail (events, quotes)
        pts["conflict_detail"] = self._score_conflict(text)

        # Dim 4 — Desired outcome
        pts["desired_outcome"] = self._score_outcome(text)

        # Dim 5 — Message substance (length proxy)
        pts["message_substance"] = self._score_substance(text)

        total = sum(pts.values())
        missing = [dim for dim, p in pts.items() if p == 0]

        if total >= 70:
            tier = "A"
        elif total >= 50:
            tier = "B"
        else:
            tier = "C"

        # Pick the most important missing dim as the scaffold question
        scaffold_q = ""
        for dim in ["conflict_detail", "context_richness", "emotional_signal",
                    "desired_outcome", "message_substance"]:
            if dim in missing:
                scaffold_q = _SCAFFOLD_QUESTIONS[dim]
                break

        return ICSResult(
            score=total,
            tier=tier,
            breakdown=pts,
            missing_dims=missing,
            scaffold_question=scaffold_q,
            should_override_stage=(tier == "C"),
        )

    # ── Dimension scorers ───────────────────────────────────────────────────────
    @staticmethod
    def _score_emotional_signal(text: str) -> int:
        matches = len(_EMOTION_WORDS.findall(text))
        if matches >= 3:  return 20
        if matches == 2:  return 15
        if matches == 1:  return 10
        # Even short frustrated messages score partial credit
        if len(text.split()) <= 6:  return 8
        return 0

    @staticmethod
    def _score_context(text: str) -> int:
        matches = len(_CONTEXT_SIGNALS.findall(text))
        if matches >= 4:  return 20
        if matches >= 2:  return 14
        if matches == 1:  return 8
        return 0

    @staticmethod
    def _score_conflict(text: str) -> int:
        matches = len(_CONFLICT_SIGNALS.findall(text))
        if matches >= 3:  return 20
        if matches >= 1:  return 12
        return 0

    @staticmethod
    def _score_outcome(text: str) -> int:
        if _OUTCOME_SIGNALS.search(text):  return 20
        # Question marks suggest they want something even without outcome keywords
        if "?" in text:                    return 10
        return 0

    @staticmethod
    def _score_substance(text: str) -> int:
        word_count = len(text.split())
        if word_count >= 30:  return 20
        if word_count >= 15:  return 14
        if word_count >= 8:   return 10
        if word_count >= 4:   return 6
        return 2   # even "hi" gets 2 — minimum effort acknowledged
