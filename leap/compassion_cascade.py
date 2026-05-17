"""
leap/compassion_cascade.py — Compassion Cascade
================================================
Tracks where the user is in LEAP's 5-phase emotional arc and maps that
to Flank's stage system. Supports both full and abbreviated sequences.

Full cascade:
  Frame → Feel → Normalize → Anchor → Empower

Abbreviated sequences (for lighter interactions):
  "light touch"   : Normalize → Empower   (already-calm user)
  "action ready"  : Anchor → Empower      (user asks "what should I do")
  "safety"        : Frame → (S0 hold)     (safety situation — don't advance)

How cascade + stages interact:
  Frame     ↔  S2 Clarify      (establish the full picture)
  Feel      ↔  S1 Listen       (sit with the emotion)
  Normalize ↔  S3 Reflect      (it makes sense to feel this way)
  Anchor    ↔  S4 Tools        (grounding / concrete technique)
  Empower   ↔  S5 Next Steps   (here's what you can actually do)

The cascade advances automatically when:
  - The stage router selects a stage ≥ current cascade phase's stage
  - Or the user explicitly asks for the next step

The cascade does NOT advance if:
  - We're in S0 (safety hold)
  - The user re-triggers strong emotion (we fall back to Feel)
  - We're in reset_requested mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Phase definitions ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CascadePhase:
    name: str               # Frame | Feel | Normalize | Anchor | Empower
    index: int              # 0–4, used for advancement logic
    mapped_stage: str       # corresponding Flank stage
    description: str        # short description for logging


PHASES: list[CascadePhase] = [
    CascadePhase("Frame",     0, "S2", "Establish context — who, what, when"),
    CascadePhase("Feel",      1, "S1", "Acknowledge and sit with the emotion"),
    CascadePhase("Normalize", 2, "S3", "Validate — it makes sense to feel this"),
    CascadePhase("Anchor",    3, "S4", "Offer a grounding tool or technique"),
    CascadePhase("Empower",   4, "S5", "Concrete next step the user can take"),
]

_PHASE_BY_STAGE: dict[str, CascadePhase] = {p.mapped_stage: p for p in PHASES}

# Abbreviated sequences — map sequence name → list of phase indices to visit
SEQUENCES: dict[str, list[int]] = {
    "full":         [0, 1, 2, 3, 4],
    "light_touch":  [2, 4],          # Normalize → Empower (calm user)
    "action_ready": [3, 4],          # Anchor → Empower (user wants to act)
    "safety":       [],              # No cascade advancement while S0 is active
}


@dataclass
class CascadeState:
    current_phase: CascadePhase
    sequence: list[int]                     # active sequence (phase indices)
    sequence_name: str
    phase_history: list[str] = field(default_factory=list)
    completed: bool = False

    def advance_hint(self) -> str:
        """Returns a hint string for the system prompt about cascade position."""
        if self.completed:
            return "[Compassion cascade: completed — affirm and close naturally]"
        idx_in_seq = (
            self.sequence.index(self.current_phase.index)
            if self.current_phase.index in self.sequence else 0
        )
        total = len(self.sequence)
        remaining_phases = [
            PHASES[i].name
            for i in self.sequence[idx_in_seq + 1:]
        ]
        remaining_str = " → ".join(remaining_phases) if remaining_phases else "done"
        return (
            f"[Compassion cascade — internal]: "
            f"Currently at '{self.current_phase.name}' phase "
            f"({idx_in_seq + 1}/{total}). "
            f"Remaining: {remaining_str}. "
            f"Don't rush ahead — stay in this phase unless they're clearly ready."
        )


class CompassionCascade:
    """
    Stateful cascade tracker for one session.

    Usage:
        cascade = CompassionCascade()

        # On each turn, after stage is selected:
        cascade.advance(stage="S1", emotion="sadness", mode="continue")
        state = cascade.state
        hint = state.advance_hint()   # inject into system prompt
    """

    def __init__(self):
        self._state = CascadeState(
            current_phase=PHASES[0],
            sequence=SEQUENCES["full"],
            sequence_name="full",
        )

    @property
    def state(self) -> CascadeState:
        return self._state

    def advance(
        self,
        stage: str,
        emotion: str,
        mode: str,
        ics_tier: str = "A",
    ) -> CascadeState:
        """
        Update cascade position based on current stage + context.

        Rules:
          - S0 active → hold at Feel (never advance past that in safety mode)
          - reset_requested → restart
          - Emotion regresses (strong anger/sadness after S3+) → fall back to Feel
          - Stage maps to a cascade phase → advance if that phase > current
        """
        # Safety hold
        if stage == "S0":
            self._state = CascadeState(
                current_phase=PHASES[1],   # Feel
                sequence=SEQUENCES["safety"],
                sequence_name="safety",
                phase_history=self._state.phase_history + ["S0-hold"],
            )
            return self._state

        # Reset
        if mode == "reset_requested":
            self._select_sequence("full", start_phase=PHASES[0])
            return self._state

        # Auto-select abbreviated sequence on first non-greeting turn
        if len(self._state.phase_history) <= 1 and self._state.sequence_name == "full":
            if stage in ("S4", "S5"):
                # User jumped straight to action
                self._select_sequence("action_ready", start_phase=PHASES[3])
            elif stage == "S3" and emotion in ("neutral", "joy"):
                self._select_sequence("light_touch", start_phase=PHASES[2])

        # Emotional regression — strong negative emotion after Anchor/Empower
        if (self._state.current_phase.index >= 3
                and emotion in ("anger", "sadness", "fear")
                and stage == "S1"):
            self._state = CascadeState(
                current_phase=PHASES[1],   # back to Feel
                sequence=self._state.sequence,
                sequence_name=self._state.sequence_name,
                phase_history=self._state.phase_history + ["regress→Feel"],
            )
            return self._state

        # Normal advancement — move to the phase that matches current stage
        target_phase = _PHASE_BY_STAGE.get(stage)
        if target_phase and self._can_advance_to(target_phase):
            history = self._state.phase_history + [self._state.current_phase.name]
            seq = self._state.sequence
            completed = (
                seq and
                seq[-1] == target_phase.index and
                target_phase.index in seq
            )
            self._state = CascadeState(
                current_phase=target_phase,
                sequence=seq,
                sequence_name=self._state.sequence_name,
                phase_history=history,
                completed=completed,
            )

        return self._state

    def _can_advance_to(self, target: CascadePhase) -> bool:
        """Only advance forward, never skip more than one phase."""
        cur_idx = self._state.current_phase.index
        return cur_idx < target.index <= cur_idx + 1

    def _select_sequence(self, name: str, start_phase: CascadePhase) -> None:
        self._state = CascadeState(
            current_phase=start_phase,
            sequence=SEQUENCES[name],
            sequence_name=name,
            phase_history=[],
        )
