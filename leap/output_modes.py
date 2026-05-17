"""
leap/output_modes.py — Dual Output Modes
=========================================
Same pipeline, two views:

  whatsapp_casual   The default. reply_messages bubbles ready to send.
                    What your user sees.

  structured        A coach/admin facing view. Full turn analysis with
                    emotion arc, stage reasoning, cascade position, tone
                    blend, ICS score, framework used. What you see when
                    debugging or reviewing a session.

Usage:
    formatter = OutputModeFormatter()
    output = formatter.format(mode="whatsapp_casual", turn_result=...)
    output = formatter.format(mode="structured", turn_result=...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


# ── Input: everything collected during one pipeline turn ─────────────────────────
@dataclass
class TurnResult:
    # Core
    user_message: str
    reply_messages: list[str]          # from LLM JSON
    stage: str
    mode: str
    transition_reason: str
    reply_style: str

    # Emotion layer
    detected_emotion: str
    emotion_confidence: float
    emotion_top_k: list[dict]

    # LEAP additions
    ics_score: int
    ics_tier: str
    ics_missing_dims: list[str]
    tone_blend: dict[str, float]       # {clarity, reassurance, synthesis, action}
    tone_dominant: str
    cascade_phase: str
    cascade_sequence: str
    framework_used: Optional[str]      # framework name or None

    # Meta
    turn_index: int
    emotion_latency_ms: float
    llm_latency_ms: float
    total_tokens: int
    session_dominant_emotion: Optional[str]


# ── Formatter ────────────────────────────────────────────────────────────────────
class OutputModeFormatter:
    """
    Formats a TurnResult for the chosen output mode.

    Returns a dict — your API layer serialises it however it needs.
    """

    def format(self, mode: str, turn: TurnResult) -> dict[str, Any]:
        if mode == "structured":
            return self._structured(turn)
        return self._whatsapp_casual(turn)

    # ── WhatsApp casual ───────────────────────────────────────────────────────────
    @staticmethod
    def _whatsapp_casual(t: TurnResult) -> dict[str, Any]:
        """
        Minimal output — only what the end user (teen) sees.
        reply_messages is a list of bubbles to send in sequence.
        """
        return {
            "mode":           t.mode,
            "stage":          t.stage,
            "reply_messages": t.reply_messages,
        }

    # ── Structured (coach / admin view) ──────────────────────────────────────────
    @staticmethod
    def _structured(t: TurnResult) -> dict[str, Any]:
        """
        Full diagnostic view — everything the pipeline computed.
        Useful for coaches reviewing sessions, or for your own debugging.
        """
        tone_pct = {
            k: f"{v:.0%}" for k, v in t.tone_blend.items()
        }
        return {
            # ── What the user said ───────────────────────────────────────────────
            "input": {
                "user_message": t.user_message,
                "turn_index":   t.turn_index,
            },

            # ── What the pipeline computed ───────────────────────────────────────
            "analysis": {
                "ics": {
                    "score":        t.ics_score,
                    "tier":         t.ics_tier,
                    "missing_dims": t.ics_missing_dims,
                },
                "emotion": {
                    "label":      t.detected_emotion,
                    "confidence": f"{t.emotion_confidence:.0%}",
                    "top_3":      t.emotion_top_k,
                    "session_dominant": t.session_dominant_emotion,
                },
                "stage": {
                    "selected":          t.stage,
                    "transition_reason": t.transition_reason,
                    "reply_style":       t.reply_style,
                },
                "cascade": {
                    "current_phase": t.cascade_phase,
                    "sequence":      t.cascade_sequence,
                },
                "tone_blend": {
                    "weights":   tone_pct,
                    "dominant":  t.tone_dominant,
                },
                "framework":  t.framework_used or "none",
                "mode":       t.mode,
            },

            # ── What Flank replied ────────────────────────────────────────────────
            "output": {
                "reply_messages": t.reply_messages,
            },

            # ── Performance metrics ───────────────────────────────────────────────
            "metrics": {
                "emotion_latency_ms": t.emotion_latency_ms,
                "llm_latency_ms":     t.llm_latency_ms,
                "total_tokens":       t.total_tokens,
            },
        }

    @staticmethod
    def to_json(output: dict[str, Any], indent: int = 2) -> str:
        return json.dumps(output, indent=indent, ensure_ascii=False)
