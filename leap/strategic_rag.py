"""
leap/strategic_rag.py — Strategic Framework Retrieval
======================================================
LEAP's "Strategic RAG" adapted for Flank — without a vector database.

Instead of embedding lookup, we use a structured framework store keyed on
(stage, emotion_group) tuples. Each framework includes:
  - A name and description
  - Concrete micro-steps for the LLM to reference
  - A ready-to-inject prompt fragment

When to upgrade to real RAG:
  - When you have 20+ frameworks in the store
  - When techniques vary by cultural context (Flank mentions this in limitations)
  - When frameworks are user-uploaded or admin-managed
  For now, this structured dict is faster, cheaper, and fully explainable.

Emotion groups (collapsed for retrieval):
  distress  = anger + sadness + fear
  stuck     = disgust
  neutral   = neutral + surprise
  positive  = joy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Framework schema ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Framework:
    name: str
    stage: str
    emotion_groups: tuple[str, ...]     # which emotion groups this fits
    description: str                    # one sentence for logging
    micro_steps: tuple[str, ...]        # 2–4 steps the LLM can reference
    prompt_fragment: str                # injected directly into system prompt


# ── Emotion group map ────────────────────────────────────────────────────────────
_EMOTION_TO_GROUP: dict[str, str] = {
    "anger":    "distress",
    "sadness":  "distress",
    "fear":     "distress",
    "disgust":  "stuck",
    "neutral":  "neutral",
    "surprise": "neutral",
    "joy":      "positive",
}


# ── Framework store ──────────────────────────────────────────────────────────────
# Key: (stage, emotion_group)
# If exact match not found → falls back to (stage, "any")
_STORE: dict[tuple[str, str], Framework] = {

    # ── S1 Listen ────────────────────────────────────────────────────────────────
    ("S1", "distress"): Framework(
        name="Reflective Listening",
        stage="S1",
        emotion_groups=("distress",),
        description="Mirror the emotion without interpreting or fixing.",
        micro_steps=(
            "Name the emotion you're hearing — don't guess, pick what's obvious.",
            "Validate without explaining it away ('that makes sense' not 'at least...').",
            "Leave space — one short message, then wait.",
        ),
        prompt_fragment=(
            "[Framework: Reflective Listening] Name the emotion clearly. "
            "Validate it simply. Don't offer solutions yet."
        ),
    ),
    ("S1", "stuck"): Framework(
        name="Acknowledgement Without Judgment",
        stage="S1",
        emotion_groups=("stuck",),
        description="Acknowledge the frustration without taking sides.",
        micro_steps=(
            "Reflect the frustration without calling anyone wrong.",
            "Show you're tracking the full picture, not just one side.",
        ),
        prompt_fragment=(
            "[Framework: Acknowledgement] Show you hear the frustration. "
            "Don't rush to fix or judge."
        ),
    ),

    # ── S3 Reframe ───────────────────────────────────────────────────────────────
    ("S3", "distress"): Framework(
        name="Perspective Shift",
        stage="S3",
        emotion_groups=("distress",),
        description="Gently offer the other person's possible headspace.",
        micro_steps=(
            "Acknowledge their version of events first.",
            "Offer a soft 'maybe they...' framing — never as a fact.",
            "Let them react; don't over-explain the reframe.",
        ),
        prompt_fragment=(
            "[Framework: Perspective Shift] Start by acknowledging their experience. "
            "Then offer one soft alternative reading of the other person's behaviour. "
            "Use 'maybe' or 'could be' — not 'they probably'."
        ),
    ),
    ("S3", "stuck"): Framework(
        name="Pattern Observation",
        stage="S3",
        emotion_groups=("stuck",),
        description="Name a pattern the user may not have seen.",
        micro_steps=(
            "Reflect back the pattern you've noticed across what they've shared.",
            "Frame it as an observation, not a diagnosis.",
            "Ask if that pattern lands for them.",
        ),
        prompt_fragment=(
            "[Framework: Pattern Observation] You've heard enough to notice a pattern. "
            "Name it gently. Ask if it resonates."
        ),
    ),

    # ── S4 Tools ─────────────────────────────────────────────────────────────────
    ("S4", "distress"): Framework(
        name="5-4-3-2-1 Grounding",
        stage="S4",
        emotion_groups=("distress",),
        description="Sensory grounding technique for high emotional activation.",
        micro_steps=(
            "5 things you can see.",
            "4 things you can physically feel.",
            "3 things you can hear.",
            "2 things you can smell.",
            "1 thing you can taste.",
        ),
        prompt_fragment=(
            "[Framework: 5-4-3-2-1 Grounding] Guide them through this step by step, "
            "one sense per message. Keep each prompt very short."
        ),
    ),
    ("S4", "stuck"): Framework(
        name="STOP Technique",
        stage="S4",
        emotion_groups=("stuck",),
        description="Break a mental loop by pausing and grounding deliberately.",
        micro_steps=(
            "Stop — pause whatever you're doing.",
            "Take one slow breath.",
            "Observe — what's actually happening right now?",
            "Proceed with that clearer head.",
        ),
        prompt_fragment=(
            "[Framework: STOP] Walk them through S-T-O-P across short messages. "
            "Don't rush. One step per message."
        ),
    ),
    ("S4", "neutral"): Framework(
        name="I-Statement Builder",
        stage="S4",
        emotion_groups=("neutral",),
        description="Help them prepare what to say using I-statements.",
        micro_steps=(
            "I feel [emotion] when [specific event].",
            "What I need is [concrete request].",
            "Practice saying it out loud once before sending.",
        ),
        prompt_fragment=(
            "[Framework: I-Statement] Help them build a short I-statement. "
            "Keep it to one sentence they could actually say or text."
        ),
    ),

    # ── S5 Next Steps ─────────────────────────────────────────────────────────────
    ("S5", "distress"): Framework(
        name="One Small Step",
        stage="S5",
        emotion_groups=("distress",),
        description="Identify the smallest possible first action.",
        micro_steps=(
            "Name one thing they could do today (not 'fix everything').",
            "Optionally draft the first message or conversation opener.",
            "Acknowledge that it might be hard.",
        ),
        prompt_fragment=(
            "[Framework: One Small Step] Offer one concrete, low-stakes action. "
            "Optionally write a short message template they could send. "
            "Keep it doable, not overwhelming."
        ),
    ),
    ("S5", "neutral"): Framework(
        name="Action Plan Lite",
        stage="S5",
        emotion_groups=("neutral",),
        description="Two-step action plan with optional message script.",
        micro_steps=(
            "Step 1: [first action]",
            "Step 2: [follow-up action]",
            "Optional: a short message they could send.",
        ),
        prompt_fragment=(
            "[Framework: Action Plan Lite] Give 1–2 clear steps. "
            "Write a short optional message script if it fits. Keep it brief."
        ),
    ),

    # ── Fallbacks ──────────────────────────────────────────────────────────────────
    ("S2", "any"): Framework(
        name="Focused Clarification",
        stage="S2",
        emotion_groups=("any",),
        description="One targeted question to fill the most important gap.",
        micro_steps=(
            "Identify the single most important missing piece.",
            "Ask about it directly, without listing all the questions you have.",
        ),
        prompt_fragment=(
            "[Framework: Focused Clarification] Ask exactly one question. "
            "Pick the most important gap. No question stacking."
        ),
    ),
    ("S0", "any"): Framework(
        name="Safety First",
        stage="S0",
        emotion_groups=("any",),
        description="De-escalate and surface support resources calmly.",
        micro_steps=(
            "Acknowledge what they've shared without alarm.",
            "Ask directly: are they safe right now?",
            "If not: encourage a trusted adult or local emergency services.",
        ),
        prompt_fragment=(
            "[Framework: Safety First] Stay calm and direct. "
            "Ask if they're safe. If imminent: mention trusted adult or crisis line."
        ),
    ),
}


# ── Retriever ────────────────────────────────────────────────────────────────────
class FrameworkRetriever:
    """
    Retrieves the best-matching framework for a (stage, emotion) pair.

    Usage:
        retriever = FrameworkRetriever()
        fw = retriever.retrieve("S4", "fear")
        print(fw.prompt_fragment)
        # "[Framework: 5-4-3-2-1 Grounding] Guide them through..."
    """

    def retrieve(
        self,
        stage: str,
        emotion: str,
    ) -> Optional[Framework]:
        """
        Priority:
          1. Exact (stage, emotion_group) match
          2. (stage, "any") fallback
          3. None (pipeline continues without a framework hint)
        """
        group = _EMOTION_TO_GROUP.get(emotion, "neutral")
        return (
            _STORE.get((stage, group))
            or _STORE.get((stage, "any"))
        )

    def list_all(self) -> list[Framework]:
        return list(_STORE.values())
