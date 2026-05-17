"""
leap/memory.py — Conversation Memory
=====================================
Maintains rolling conversation history with:
  - Full message + metadata per turn (capped at MAX_TURNS_VERBATIM)
  - Compressed summary of older turns (to keep context window lean)
  - Emotion arc tracking across the session
  - Entity tracking (people, places mentioned)

No external dependencies beyond stdlib.
Drop-in for your DynamicPromptingLayer — just pass `memory` into it.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ── Config ─────────────────────────────────────────────────────────────────────
MAX_TURNS_VERBATIM = 6      # keep last N turns in full; older turns → summary
MAX_SUMMARY_CHARS  = 400    # cap on the compressed summary string


# ── Turn record ────────────────────────────────────────────────────────────────
@dataclass
class Turn:
    turn_index: int
    user_message: str
    assistant_reply: Optional[str]      # joined reply_messages from LLM
    detected_emotion: str
    emotion_confidence: float
    stage: str
    mode: str
    timestamp: float = field(default_factory=time.time)


# ── Entity extractor (simple heuristic — no NLP library needed) ────────────────
_PRONOUN = re.compile(
    r"\b(she|he|they|her|him|them|his|hers)\b", re.IGNORECASE
)
_PROPER = re.compile(r"\b([A-Z][a-z]{2,})\b")   # rough proper-noun detector


def _extract_entities(text: str) -> list[str]:
    """Pull likely people-names + pronoun groups from a message."""
    names = _PROPER.findall(text)
    # Filter out common sentence-starters that aren't names
    stopwords = {"The", "She", "He", "They", "It", "We", "You", "I", "My",
                 "Our", "But", "And", "So", "Because", "When"}
    names = [n for n in names if n not in stopwords]
    pronouns = list(set(m.lower() for m in _PRONOUN.findall(text)))
    return list(set(names)) + pronouns


# ── Main memory class ──────────────────────────────────────────────────────────
class ConversationMemory:
    """
    Stateful session memory.

    Usage inside DynamicPromptingLayer:
        self.memory = ConversationMemory()

    Each turn:
        self.memory.add_turn(turn)          # after routing, before LLM call
        self.memory.set_assistant_reply(    # after LLM responds
            reply_text, turn_index
        )
        context = self.memory.get_context_string()   # inject into system prompt
    """

    def __init__(self):
        self._turns: list[Turn] = []
        self._summary: str = ""         # compressed summary of older turns
        self._entities: Counter = Counter()
        self._emotion_arc: list[tuple[str, float]] = []   # (label, confidence)

    # ── Write ──────────────────────────────────────────────────────────────────
    def add_turn(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._emotion_arc.append((turn.detected_emotion, turn.emotion_confidence))
        for entity in _extract_entities(turn.user_message):
            self._entities[entity] += 1
        self._maybe_compress()

    def set_assistant_reply(self, reply_text: str, turn_index: int) -> None:
        """Call this once you have the LLM's reply, so memory stores both sides."""
        for turn in reversed(self._turns):
            if turn.turn_index == turn_index:
                turn.assistant_reply = reply_text
                break

    # ── Read ───────────────────────────────────────────────────────────────────
    def get_context_string(self) -> str:
        """
        Returns a compact string to inject into the system prompt.
        Format:
            [Session memory]
            Summary: <compressed older turns>
            Recent: <last N turns verbatim>
            Emotion arc: anger → sadness → neutral
            People mentioned: Alex, her, him
        """
        parts: list[str] = ["[Session memory — internal, do NOT surface to user]"]

        if self._summary:
            parts.append(f"Summary of earlier turns: {self._summary}")

        verbatim = self._turns[-MAX_TURNS_VERBATIM:]
        if verbatim:
            parts.append("Recent turns:")
            for t in verbatim:
                parts.append(f"  [{t.stage}] User: {t.user_message}")
                if t.assistant_reply:
                    parts.append(f"         Flank: {t.assistant_reply}")

        if self._emotion_arc:
            arc_str = " → ".join(e for e, _ in self._emotion_arc[-6:])
            parts.append(f"Emotion arc (recent): {arc_str}")

        top_entities = [e for e, _ in self._entities.most_common(4)]
        if top_entities:
            parts.append(f"People/entities mentioned: {', '.join(top_entities)}")

        return "\n".join(parts)

    def get_last_n_messages(self, n: int = 6) -> list[dict]:
        """Returns last N turns as [{"role": ..., "content": ...}] for LangChain."""
        msgs: list[dict] = []
        for turn in self._turns[-n:]:
            msgs.append({"role": "user", "content": turn.user_message})
            if turn.assistant_reply:
                msgs.append({"role": "assistant", "content": turn.assistant_reply})
        return msgs

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def dominant_emotion(self) -> Optional[str]:
        """Most frequent emotion in this session."""
        if not self._emotion_arc:
            return None
        counts: Counter = Counter(e for e, _ in self._emotion_arc)
        return counts.most_common(1)[0][0]

    @property
    def last_stage(self) -> str:
        if self._turns:
            return self._turns[-1].stage
        return "none"

    @property
    def last_mode(self) -> str:
        if self._turns:
            return self._turns[-1].mode
        return "continue"

    # ── Compression ────────────────────────────────────────────────────────────
    def _maybe_compress(self) -> None:
        """
        When history exceeds MAX_TURNS_VERBATIM, compress older turns into a
        plain-English summary and remove them from the verbatim list.

        This is a heuristic compressor — no LLM call required.
        For production, replace _heuristic_summarise() with an LLM call.
        """
        if len(self._turns) <= MAX_TURNS_VERBATIM:
            return

        old_turns = self._turns[:-MAX_TURNS_VERBATIM]
        self._turns = self._turns[-MAX_TURNS_VERBATIM:]
        self._summary = self._heuristic_summarise(old_turns)

    @staticmethod
    def _heuristic_summarise(turns: list[Turn]) -> str:
        """
        Builds a compact plain-English summary from old turns without an LLM.
        Replace this with an actual LLM call for higher quality.

        Example output:
            "User vented about a friend ignoring them (anger, S1→S3).
             Flank reflected and reframed. User asked what to do next (S4)."
        """
        stage_seq = " → ".join(t.stage for t in turns)
        emotions   = ", ".join(set(t.detected_emotion for t in turns))
        # Pick the most emotionally loaded message as the "topic"
        topic_turn = max(turns, key=lambda t: t.emotion_confidence)
        snippet    = topic_turn.user_message[:120].rstrip()
        summary = (
            f"Earlier: user discussed '{snippet}…' "
            f"Emotions detected: {emotions}. "
            f"Stage progression: {stage_seq}."
        )
        return summary[:MAX_SUMMARY_CHARS]

    def reset(self) -> None:
        """Hard reset — call when mode == reset_requested."""
        self._turns.clear()
        self._summary = ""
        self._entities.clear()
        self._emotion_arc.clear()
