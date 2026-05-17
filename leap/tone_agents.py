"""
leap/tone_agents.py — Tone Agent Blending
==========================================
LEAP's Tone Agents adapted for Flank's conflict-coaching context.

Four agents (mirroring LEAP's archetypes):
  Clarity      Direct, precise, cuts through confusion.
  Reassurance  Warm, validating, confidence-building.
  Synthesis    Connective, reframing, finding the bigger picture.
  Action       Concrete, decisive, "here's what to actually do".

Each stage has a default blend.
The detected emotion modulates that blend at runtime.

The final blend is serialised into a short prompt fragment injected into
the system prompt — so the LLM adapts tone without needing hardcoded personas.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Blend schema ────────────────────────────────────────────────────────────────
@dataclass
class ToneBlend:
    clarity:     float = 0.0
    reassurance: float = 0.0
    synthesis:   float = 0.0
    action:      float = 0.0

    def normalise(self) -> "ToneBlend":
        """Ensure weights sum to 1.0."""
        total = self.clarity + self.reassurance + self.synthesis + self.action
        if total == 0:
            return ToneBlend(clarity=0.25, reassurance=0.25,
                             synthesis=0.25, action=0.25)
        return ToneBlend(
            clarity=round(self.clarity / total, 2),
            reassurance=round(self.reassurance / total, 2),
            synthesis=round(self.synthesis / total, 2),
            action=round(self.action / total, 2),
        )

    def dominant(self) -> str:
        agents = {
            "clarity":     self.clarity,
            "reassurance": self.reassurance,
            "synthesis":   self.synthesis,
            "action":      self.action,
        }
        return max(agents, key=agents.get)

    def to_prompt_fragment(self) -> str:
        """
        Returns a short natural-language tone instruction for the system prompt.
        Example:
            "Tone blend: lead with Reassurance (0.6), blend in Synthesis (0.3),
             hint of Clarity (0.1). Adjust wording accordingly."
        """
        ordered = sorted(
            [("Clarity", self.clarity), ("Reassurance", self.reassurance),
             ("Synthesis", self.synthesis), ("Action", self.action)],
            key=lambda x: x[1], reverse=True,
        )
        # Only include agents with weight > 0.05
        active = [(name, w) for name, w in ordered if w > 0.05]
        if not active:
            return ""

        desc_map = {
            "Clarity":     "precise and direct",
            "Reassurance": "warm and validating",
            "Synthesis":   "connective and reframing",
            "Action":      "concrete and decisive",
        }

        lead_name, lead_w = active[0]
        parts = [f"Lead with {lead_name} ({lead_w:.0%}) — be {desc_map[lead_name]}"]
        for name, w in active[1:]:
            parts.append(f"blend in {name} ({w:.0%})")

        return (
            "[Tone blend for this reply — internal, do not mention to user]: "
            + ", ".join(parts) + "."
        )


# ── Stage → default blend ────────────────────────────────────────────────────────
# These are tuned for Flank's conflict-coaching context.
# Weights don't need to sum to 1 here — normalise() handles it.
_STAGE_DEFAULTS: dict[str, ToneBlend] = {
    "S0": ToneBlend(clarity=0.4, reassurance=0.5, synthesis=0.0, action=0.1),
    "S1": ToneBlend(clarity=0.1, reassurance=0.8, synthesis=0.1, action=0.0),
    "S2": ToneBlend(clarity=0.5, reassurance=0.3, synthesis=0.2, action=0.0),
    "S3": ToneBlend(clarity=0.1, reassurance=0.3, synthesis=0.6, action=0.0),
    "S4": ToneBlend(clarity=0.3, reassurance=0.2, synthesis=0.2, action=0.3),
    "S5": ToneBlend(clarity=0.2, reassurance=0.2, synthesis=0.1, action=0.5),
    "S6": ToneBlend(clarity=0.2, reassurance=0.6, synthesis=0.2, action=0.0),
}

# ── Emotion modifiers — deltas applied on top of stage default ──────────────────
# Each emotion nudges specific agents up or down.
_EMOTION_MODIFIERS: dict[str, dict[str, float]] = {
    "anger":    {"reassurance": +0.15, "action": -0.10, "clarity": -0.05},
    "sadness":  {"reassurance": +0.20, "synthesis": +0.05, "action": -0.15},
    "fear":     {"reassurance": +0.20, "clarity": +0.10, "action": -0.10},
    "disgust":  {"synthesis": +0.15, "reassurance": +0.10, "action": -0.10},
    "surprise": {"clarity": +0.15, "synthesis": +0.10},
    "joy":      {"action": +0.10, "reassurance": +0.10},
    "neutral":  {},   # no change
}


# ── Blender ─────────────────────────────────────────────────────────────────────
class ToneAgentBlender:
    """
    Computes the final tone blend for a given (stage, emotion) pair.

    Usage:
        blender = ToneAgentBlender()
        blend = blender.compute("S1", "anger")
        print(blend.to_prompt_fragment())
        # "Lead with Reassurance (80%) — be warm and validating,
        #  blend in Synthesis (10%), hint of Clarity (10%)."
    """

    def compute(
        self,
        stage: str,
        emotion: str,
        emotion_confidence: float = 1.0,
    ) -> ToneBlend:
        """
        Args:
            stage:              Selected stage (S0–S6).
            emotion:            Detected emotion label (lowercase).
            emotion_confidence: 0–1. Lower confidence → weaker emotion modifiers.
        """
        base = _STAGE_DEFAULTS.get(stage, _STAGE_DEFAULTS["S2"])
        modifiers = _EMOTION_MODIFIERS.get(emotion, {})

        # Apply modifiers, scaled by confidence (low confidence = mild adjustment)
        scale = emotion_confidence
        blended = ToneBlend(
            clarity=base.clarity     + scale * modifiers.get("clarity", 0),
            reassurance=base.reassurance + scale * modifiers.get("reassurance", 0),
            synthesis=base.synthesis   + scale * modifiers.get("synthesis", 0),
            action=base.action       + scale * modifiers.get("action", 0),
        )

        # Clamp to [0, 1] before normalising
        blended = ToneBlend(
            clarity=max(0.0, min(1.0, blended.clarity)),
            reassurance=max(0.0, min(1.0, blended.reassurance)),
            synthesis=max(0.0, min(1.0, blended.synthesis)),
            action=max(0.0, min(1.0, blended.action)),
        )

        return blended.normalise()
