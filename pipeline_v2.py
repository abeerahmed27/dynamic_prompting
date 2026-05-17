"""
pipeline_v2.py — Flank Dynamic Prompting + LEAP Integration
=============================================================
Extends your existing pipeline.py with all 5 LEAP concepts + conversation memory.

New pipeline order per turn:
  1. ICS               Score input quality → maybe override to S2 scaffold
  2. EmotionDetector   Detect emotion (unchanged from your v1)
  3. Memory            Retrieve conversation context
  4. Mode detection    continue | new_topic | reset_requested (unchanged)
  5. Stage routing     S0–S6 (unchanged, but ICS can override)
  6. CompassionCascade Track where we are in Frame→Feel→Normalize→Anchor→Empower
  7. ToneAgentBlender  Compute tone blend for this stage × emotion
  8. FrameworkRetriever Pull the right technique/micro-steps for this context
  9. Prompt builder    Inject all hints into system prompt
  10. Memory write     Store this turn
  11. Return payload   LLMPayload (backward-compatible) + TurnResult for structured view

Your existing EmotionDetector, detect_mode, route_stage, and SYSTEM_PROMPT_v5
are imported unchanged — nothing in your current code breaks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ── Your existing pipeline (unchanged) ────────────────────────────────────────
from emotional_prompting_layer import (
    SYSTEM_PROMPT_v5,
    EmotionDetector,
    EmotionResult,
    LLMPayload,
    detect_mode,
    route_stage,
)

# ── New LEAP modules ───────────────────────────────────────────────────────────
from leap.memory            import ConversationMemory, Turn
from leap.ics               import InputCompletenessScorer, ICSResult
from leap.tone_agents       import ToneAgentBlender, ToneBlend
from leap.compassion_cascade import CompassionCascade
from leap.strategic_rag     import FrameworkRetriever, Framework
from leap.output_modes      import OutputModeFormatter, TurnResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Extended payload (backward-compatible with your LLMPayload) ────────────────
@dataclass
class EnrichedLLMPayload(LLMPayload):
    """
    Adds LEAP metadata on top of your existing LLMPayload.
    Your LLM client receives system_prompt + user_message as before.
    Everything else is for logging, evaluation, and structured output.
    """
    # ICS
    ics_score:         int   = 0
    ics_tier:          str   = "A"
    ics_missing_dims:  list  = field(default_factory=list)

    # Tone blend
    tone_blend:        dict  = field(default_factory=dict)
    tone_dominant:     str   = ""

    # Cascade
    cascade_phase:     str   = ""
    cascade_sequence:  str   = "full"
    cascade_hint:      str   = ""

    # Framework
    framework_name:    str   = ""
    framework_steps:   tuple = ()

    # Memory
    session_dominant_emotion: Optional[str] = None
    conversation_context:     str   = ""

    # Output mode
    output_mode:       str   = "whatsapp_casual"


# ── Main orchestrator ─────────────────────────────────────────────────────────
class DynamicPromptingLayerV2:
    """
    Drop-in replacement for your DynamicPromptingLayer.
    Same interface: call .prepare_turn(user_message) each turn.
    Returns an EnrichedLLMPayload.

    After the LLM responds, call .record_reply(reply_text, turn_index)
    so memory stores both sides of the conversation.

    Usage:
        layer = DynamicPromptingLayerV2(output_mode="whatsapp_casual")
        payload = layer.prepare_turn("she ignored me all day")

        # Send to your LLM client:
        llm_response = your_llm_client.call(
            system=payload.system_prompt,
            user=payload.user_message,
        )
        parsed = json.loads(llm_response)
        layer.record_reply(
            reply_text=" ".join(parsed["reply_messages"]),
            turn_index=payload.turn_index,
        )

        # Get structured view:
        structured = layer.get_structured_output(payload, parsed, llm_latency_ms=320)
    """

    def __init__(
        self,
        channel: str = "whatsapp",
        output_mode: str = "whatsapp_casual",   # "whatsapp_casual" | "structured"
    ):
        self.channel       = channel
        self.output_mode   = output_mode

        # Existing components (unchanged)
        self.emotion_detector = EmotionDetector()

        # LEAP components (new)
        self.memory            = ConversationMemory()
        self.ics               = InputCompletenessScorer()
        self.tone_blender      = ToneAgentBlender()
        self.cascade           = CompassionCascade()
        self.framework_store   = FrameworkRetriever()
        self.output_formatter  = OutputModeFormatter()

        # Session state (replaces your self.last_stage + self.turn_index)
        # Now read from memory instead
        self._pending_turn_index: Optional[int] = None

    # ── Main entry point ───────────────────────────────────────────────────────
    def prepare_turn(self, user_message: str) -> EnrichedLLMPayload:
        turn_index = self.memory.turn_count + 1
        self._pending_turn_index = turn_index

        # ── Step 1: ICS — quality-gate the input ─────────────────────────────
        ics_result: ICSResult = self.ics.score(user_message)
        logger.info(
            "[T%d] ICS score=%d tier=%s missing=%s",
            turn_index, ics_result.score, ics_result.tier, ics_result.missing_dims,
        )

        # ── Step 2: Emotion detection ─────────────────────────────────────────
        emotion: EmotionResult = self.emotion_detector.detect(user_message)
        logger.info(
            "[T%d] Emotion: %s (%.0f%%) | %.0fms",
            turn_index, emotion.label, emotion.score * 100, emotion.latency_ms,
        )

        # ── Step 3: Mode detection ────────────────────────────────────────────
        last_stage = self.memory.last_stage
        mode = detect_mode(user_message, last_stage, turn_index)
        if mode == "reset_requested":
            self.memory.reset()
            self.cascade._state  # cascade handles reset internally via .advance()

        # ── Step 4: Stage routing (with ICS override) ─────────────────────────
        stage = route_stage(emotion, user_message, last_stage, mode)

        # ICS tier C override — not enough info, force S2 Clarify
        if ics_result.should_override_stage and stage not in ("S0", "S6"):
            logger.info("[T%d] ICS tier C → overriding %s to S2", turn_index, stage)
            stage = "S2"

        logger.info("[T%d] Stage: %s | Mode: %s", turn_index, stage, mode)

        # ── Step 5: Compassion cascade advancement ────────────────────────────
        cascade_state = self.cascade.advance(
            stage=stage,
            emotion=emotion.label,
            mode=mode,
            ics_tier=ics_result.tier,
        )

        # ── Step 6: Tone blend ────────────────────────────────────────────────
        blend: ToneBlend = self.tone_blender.compute(
            stage=stage,
            emotion=emotion.label,
            emotion_confidence=emotion.score,
        )

        # ── Step 7: Framework retrieval ───────────────────────────────────────
        framework: Optional[Framework] = self.framework_store.retrieve(
            stage=stage,
            emotion=emotion.label,
        )

        # ── Step 8: Build enriched system prompt ──────────────────────────────
        memory_context = self.memory.get_context_string()
        system_prompt  = self._build_system_prompt(
            base_prompt    = SYSTEM_PROMPT_v5,
            emotion        = emotion,
            stage          = stage,
            mode           = mode,
            blend          = blend,
            cascade_hint   = cascade_state.advance_hint(),
            framework      = framework,
            memory_context = memory_context,
            ics_result     = ics_result,
        )

        # ── Step 9: Store turn in memory (assistant reply added later) ─────────
        turn = Turn(
            turn_index        = turn_index,
            user_message      = user_message,
            assistant_reply   = None,   # filled by record_reply()
            detected_emotion  = emotion.label,
            emotion_confidence= emotion.score,
            stage             = stage,
            mode              = mode,
        )
        self.memory.add_turn(turn)

        # ── Step 10: Assemble payload ─────────────────────────────────────────
        return EnrichedLLMPayload(
            # LLMPayload fields (your existing interface)
            system_prompt      = system_prompt,
            user_message       = user_message,
            detected_emotion   = emotion.label,
            emotion_confidence = emotion.score,
            selected_stage     = stage,
            selected_mode      = mode,
            prompt_version     = "v5-leap",
            emotion_top_k      = emotion.top_k,
            emotion_latency_ms = emotion.latency_ms,

            # LEAP additions
            ics_score          = ics_result.score,
            ics_tier           = ics_result.tier,
            ics_missing_dims   = ics_result.missing_dims,
            tone_blend         = {
                "clarity":     blend.clarity,
                "reassurance": blend.reassurance,
                "synthesis":   blend.synthesis,
                "action":      blend.action,
            },
            tone_dominant      = blend.dominant(),
            cascade_phase      = cascade_state.current_phase.name,
            cascade_sequence   = cascade_state.sequence_name,
            cascade_hint       = cascade_state.advance_hint(),
            framework_name     = framework.name if framework else "",
            framework_steps    = framework.micro_steps if framework else (),
            session_dominant_emotion = self.memory.dominant_emotion,
            conversation_context     = memory_context,
            output_mode        = self.output_mode,
        )

    # ── Call after LLM responds ────────────────────────────────────────────────
    def record_reply(self, reply_text: str, turn_index: int) -> None:
        """Store the assistant's reply so memory has both sides."""
        self.memory.set_assistant_reply(reply_text, turn_index)

    # ── Structured output builder ──────────────────────────────────────────────
    def get_structured_output(
        self,
        payload: EnrichedLLMPayload,
        llm_parsed_json: dict,
        llm_latency_ms: float = 0.0,
        total_tokens: int = 0,
    ) -> dict[str, Any]:
        """
        Builds the structured coach/admin view after the LLM responds.
        llm_parsed_json: the parsed JSON from your LLM response.
        """
        turn = TurnResult(
            user_message       = payload.user_message,
            reply_messages     = llm_parsed_json.get("reply_messages", []),
            stage              = llm_parsed_json.get("stage", payload.selected_stage),
            mode               = llm_parsed_json.get("mode", payload.selected_mode),
            transition_reason  = llm_parsed_json.get("transition_reason", ""),
            reply_style        = llm_parsed_json.get("reply_style", ""),
            detected_emotion   = payload.detected_emotion,
            emotion_confidence = payload.emotion_confidence,
            emotion_top_k      = payload.emotion_top_k,
            ics_score          = payload.ics_score,
            ics_tier           = payload.ics_tier,
            ics_missing_dims   = payload.ics_missing_dims,
            tone_blend         = payload.tone_blend,
            tone_dominant      = payload.tone_dominant,
            cascade_phase      = payload.cascade_phase,
            cascade_sequence   = payload.cascade_sequence,
            framework_used     = payload.framework_name or None,
            turn_index         = self.memory.turn_count,
            emotion_latency_ms = payload.emotion_latency_ms,
            llm_latency_ms     = llm_latency_ms,
            total_tokens       = total_tokens,
            session_dominant_emotion = payload.session_dominant_emotion,
        )
        return self.output_formatter.format(self.output_mode, turn)

    # ── Prompt builder ─────────────────────────────────────────────────────────
    @staticmethod
    def _build_system_prompt(
        base_prompt:    str,
        emotion:        EmotionResult,
        stage:          str,
        mode:           str,
        blend:          ToneBlend,
        cascade_hint:   str,
        framework:      Optional[Framework],
        memory_context: str,
        ics_result:     ICSResult,
    ) -> str:
        """
        Injects all LEAP hints into the base system prompt.
        All hints are marked [internal] so the LLM knows not to surface them.
        """
        parts = [base_prompt.strip()]

        # Memory context (session history + emotion arc)
        if memory_context:
            parts.append(memory_context)

        # Emotion + stage routing hint
        parts.append(
            f"\n[Internal routing — do NOT mention to user]: "
            f"detected_emotion={emotion.label} ({emotion.score:.0%}), "
            f"routed_stage={stage}, conversation_mode={mode}."
        )

        # Tone blend
        tone_fragment = blend.to_prompt_fragment()
        if tone_fragment:
            parts.append(tone_fragment)

        # Compassion cascade position
        if cascade_hint:
            parts.append(cascade_hint)

        # Strategic framework
        if framework:
            parts.append(framework.prompt_fragment)

        # ICS scaffold question (tier C only)
        if ics_result.should_override_stage and ics_result.scaffold_question:
            parts.append(
                f"[ICS override]: Input was incomplete (score {ics_result.score}/100). "
                f"Use S2 Clarify. Suggested question: \"{ics_result.scaffold_question}\""
            )

        return "\n\n".join(parts)


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import dataclasses

    layer = DynamicPromptingLayerV2(
        channel="whatsapp",
        output_mode="whatsapp_casual",   # switch to "structured" to see full analysis
    )

    conversation = [
        "hey",
        "she",                                  # ICS tier C → should force S2
        "my best friend completely ignored me at school today in front of everyone",
        "i just feel like she hates me now, i always mess everything up",
        "what should i even say to her",
        "actually never mind, it's not about her. it's my parents",
    ]

    for user_msg in conversation:
        payload = layer.prepare_turn(user_msg)

        print("\n" + "═" * 65)
        print(f"  User        : {payload.user_message}")
        print(f"  Emotion     : {payload.detected_emotion} ({payload.emotion_confidence:.0%})")
        print(f"  ICS         : {payload.ics_score}/100 (tier {payload.ics_tier})")
        print(f"  Stage       : {payload.selected_stage} | Mode: {payload.selected_mode}")
        print(f"  Tone        : {payload.tone_dominant} — "
              + " / ".join(f"{k}={v:.0%}" for k, v in payload.tone_blend.items()))
        print(f"  Cascade     : {payload.cascade_phase} ({payload.cascade_sequence})")
        print(f"  Framework   : {payload.framework_name or 'none'}")
        print(f"  Memory turns: {layer.memory.turn_count}")

        # ── Simulate LLM response (in real use, call your LLM client here) ───
        fake_llm_response = json.dumps({
            "mode":              payload.selected_mode,
            "stage":             payload.selected_stage,
            "transition_reason": "simulated",
            "reply_style":       "simulated",
            "reply_messages":    [
                f"[Simulated reply for stage {payload.selected_stage}]"
            ],
        })
        parsed = json.loads(fake_llm_response)

        # Record the reply so memory stores both sides
        layer.record_reply(
            reply_text=" ".join(parsed["reply_messages"]),
            turn_index=layer.memory.turn_count,
        )

        # Switch to structured output for the last turn to demo it
        if user_msg == "what should i even say to her":
            layer.output_mode = "structured"
            structured = layer.get_structured_output(payload, parsed, llm_latency_ms=310)
            print("\n  ── Structured output (coach view) ──")
            print(json.dumps(structured, indent=2))
            layer.output_mode = "whatsapp_casual"
