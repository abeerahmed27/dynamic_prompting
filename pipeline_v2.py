"""
Enhanced dynamic prompting pipeline with LEAP-style support modules.

This version keeps the original interface but improves:
1. Context-aware ICS scoring.
2. Fewer unnecessary S1 -> S2 clarifications.
3. More stable emotion handling across turns.
4. Cleaner reset behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from emotional_prompting_layer import (
    SYSTEM_PROMPT_v5,
    EmotionDetector,
    EmotionResult,
    LLMPayload,
    detect_mode,
    route_stage,
)
from leap.compassion_cascade import CompassionCascade
from leap.ics import ICSResult, InputCompletenessScorer
from leap.memory import ConversationMemory, Turn
from leap.output_modes import OutputModeFormatter, TurnResult
from leap.strategic_rag import Framework, FrameworkRetriever
from leap.tone_agents import ToneAgentBlender, ToneBlend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


@dataclass
class EnrichedLLMPayload(LLMPayload):
    ics_score: int = 0
    ics_tier: str = "A"
    ics_missing_dims: list = field(default_factory=list)
    tone_blend: dict = field(default_factory=dict)
    tone_dominant: str = ""
    cascade_phase: str = ""
    cascade_sequence: str = "full"
    cascade_hint: str = ""
    framework_name: str = ""
    framework_steps: tuple = ()
    session_dominant_emotion: Optional[str] = None
    conversation_context: str = ""
    output_mode: str = "whatsapp_casual"
    turn_index: int = 0


class DynamicPromptingLayerV2:
    def __init__(self, channel: str = "whatsapp", output_mode: str = "whatsapp_casual"):
        self.channel = channel
        self.output_mode = output_mode
        self.emotion_detector = EmotionDetector()
        self.memory = ConversationMemory()
        self.ics = InputCompletenessScorer()
        self.tone_blender = ToneAgentBlender()
        self.cascade = CompassionCascade()
        self.framework_store = FrameworkRetriever()
        self.output_formatter = OutputModeFormatter()

    def prepare_turn(self, user_message: str) -> EnrichedLLMPayload:
        turn_index = self.memory.turn_count + 1
        memory_context = self.memory.get_context_string()
        last_stage = self.memory.last_stage
        last_missing_dims = ["context_richness", "conflict_detail", "desired_outcome"] if last_stage == "S2" else []

        ics_result: ICSResult = self.ics.score(
            user_message,
            conversation_context=memory_context,
            known_entities=self.memory.known_entities,
            last_missing_dims=last_missing_dims,
        )
        logger.info(
            "[T%d] ICS score=%d tier=%s missing=%s",
            turn_index,
            ics_result.score,
            ics_result.tier,
            ics_result.missing_dims,
        )

        emotion = self._stabilize_emotion(self.emotion_detector.detect(user_message))
        logger.info(
            "[T%d] Emotion: %s (%.0f%%) | %.0fms",
            turn_index,
            emotion.label,
            emotion.score * 100,
            emotion.latency_ms,
        )

        mode = detect_mode(user_message, last_stage, turn_index)
        if mode == "reset_requested":
            self.memory.reset()
            turn_index = 1
            last_stage = "none"
            memory_context = ""

        if self._should_hold_safety(last_stage, user_message):
            stage = "S0"
        else:
            stage = route_stage(emotion, user_message, last_stage, mode)
            stage = self._promote_stage_for_intent(
                user_message=user_message,
                stage=stage,
                last_stage=last_stage,
                emotion=emotion,
                ics_result=ics_result,
            )

        if self._should_force_clarify(stage, last_stage, emotion, ics_result):
            logger.info("[T%d] ICS tier C -> overriding %s to S2", turn_index, stage)
            stage = "S2"
        elif last_stage == "S2" and stage == "S2" and emotion.label in ("anger", "sadness", "fear"):
            logger.info("[T%d] Emotional continuity detected -> staying in S1", turn_index)
            stage = "S1"
        elif (
            stage == "S2"
            and last_stage == "S1"
            and self.memory.turn_count > 0
            and ics_result.breakdown.get("conflict_detail", 0) >= 12
        ):
            logger.info("[T%d] Same-thread detail added -> staying in S1", turn_index)
            stage = "S1"

        logger.info("[T%d] Stage: %s | Mode: %s", turn_index, stage, mode)

        cascade_state = self.cascade.advance(
            stage=stage,
            emotion=emotion.label,
            mode=mode,
            ics_tier=ics_result.tier,
        )
        blend: ToneBlend = self.tone_blender.compute(
            stage=stage,
            emotion=emotion.label,
            emotion_confidence=emotion.score,
        )
        framework: Optional[Framework] = self.framework_store.retrieve(
            stage=stage,
            emotion=emotion.label,
            intent=self._detect_intent(user_message),
        )
        system_prompt = self._build_system_prompt(
            base_prompt=SYSTEM_PROMPT_v5,
            emotion=emotion,
            stage=stage,
            mode=mode,
            blend=blend,
            cascade_hint=cascade_state.advance_hint(),
            framework=framework,
            memory_context=memory_context,
            ics_result=ics_result,
        )

        self.memory.add_turn(
            Turn(
                turn_index=turn_index,
                user_message=user_message,
                assistant_reply=None,
                detected_emotion=emotion.label,
                emotion_confidence=emotion.score,
                stage=stage,
                mode=mode,
            )
        )

        return EnrichedLLMPayload(
            system_prompt=system_prompt,
            user_message=user_message,
            detected_emotion=emotion.label,
            emotion_confidence=emotion.score,
            selected_stage=stage,
            selected_mode=mode,
            prompt_version="v5-leap",
            emotion_top_k=emotion.top_k,
            emotion_latency_ms=emotion.latency_ms,
            ics_score=ics_result.score,
            ics_tier=ics_result.tier,
            ics_missing_dims=ics_result.missing_dims,
            tone_blend={
                "clarity": blend.clarity,
                "reassurance": blend.reassurance,
                "synthesis": blend.synthesis,
                "action": blend.action,
            },
            tone_dominant=blend.dominant(),
            cascade_phase=cascade_state.current_phase.name,
            cascade_sequence=cascade_state.sequence_name,
            cascade_hint=cascade_state.advance_hint(),
            framework_name=framework.name if framework else "",
            framework_steps=framework.micro_steps if framework else (),
            session_dominant_emotion=self.memory.dominant_emotion,
            conversation_context=memory_context,
            output_mode=self.output_mode,
            turn_index=turn_index,
        )

    def record_reply(self, reply_text: str, turn_index: int) -> None:
        self.memory.set_assistant_reply(reply_text, turn_index)

    def get_structured_output(
        self,
        payload: EnrichedLLMPayload,
        llm_parsed_json: dict,
        llm_latency_ms: float = 0.0,
        total_tokens: int = 0,
    ) -> dict[str, Any]:
        turn = TurnResult(
            user_message=payload.user_message,
            reply_messages=llm_parsed_json.get("reply_messages", []),
            stage=llm_parsed_json.get("stage", payload.selected_stage),
            mode=llm_parsed_json.get("mode", payload.selected_mode),
            transition_reason=llm_parsed_json.get("transition_reason", ""),
            reply_style=llm_parsed_json.get("reply_style", ""),
            detected_emotion=payload.detected_emotion,
            emotion_confidence=payload.emotion_confidence,
            emotion_top_k=payload.emotion_top_k,
            ics_score=payload.ics_score,
            ics_tier=payload.ics_tier,
            ics_missing_dims=payload.ics_missing_dims,
            tone_blend=payload.tone_blend,
            tone_dominant=payload.tone_dominant,
            cascade_phase=payload.cascade_phase,
            cascade_sequence=payload.cascade_sequence,
            framework_used=payload.framework_name or None,
            turn_index=payload.turn_index,
            emotion_latency_ms=payload.emotion_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_tokens=total_tokens,
            session_dominant_emotion=payload.session_dominant_emotion,
        )
        return self.output_formatter.format(self.output_mode, turn)

    @staticmethod
    def _build_system_prompt(
        base_prompt: str,
        emotion: EmotionResult,
        stage: str,
        mode: str,
        blend: ToneBlend,
        cascade_hint: str,
        framework: Optional[Framework],
        memory_context: str,
        ics_result: ICSResult,
    ) -> str:
        parts = [base_prompt.strip()]
        if memory_context:
            parts.append(memory_context)
        parts.append(
            f"[Internal routing - do NOT mention to user]: detected_emotion={emotion.label} "
            f"({emotion.score:.0%}), routed_stage={stage}, conversation_mode={mode}."
        )
        tone_fragment = blend.to_prompt_fragment()
        if tone_fragment:
            parts.append(tone_fragment)
        if cascade_hint:
            parts.append(cascade_hint)
        if framework:
            parts.append(framework.prompt_fragment)
        if ics_result.should_override_stage and ics_result.scaffold_question:
            parts.append(
                f"[ICS override]: Input was incomplete (score {ics_result.score}/100). "
                f'Use S2 Clarify. Suggested question: "{ics_result.scaffold_question}"'
            )
        return "\n\n".join(parts)

    @staticmethod
    def _should_force_clarify(
        stage: str,
        last_stage: str,
        emotion: EmotionResult,
        ics_result: ICSResult,
    ) -> bool:
        return (
            ics_result.should_override_stage
            and stage != "S2"
            and stage not in ("S0", "S6")
            and not (last_stage == "S2" and ics_result.answered_by_context)
            and emotion.label not in ("anger", "sadness", "fear")
        )

    @staticmethod
    def _should_hold_safety(last_stage: str, user_message: str) -> bool:
        if last_stage != "S0":
            return False
        text = user_message.lower().strip()
        return text in {
            "no",
            "not safe",
            "i am not safe",
            "no i am not safe",
            "i don't feel safe",
            "i do not feel safe",
        }

    @staticmethod
    def _is_safe_confirmation(last_stage: str, user_message: str) -> bool:
        if last_stage != "S0":
            return False
        return user_message.lower().strip() in {"yes", "yes i am safe", "i am safe", "yeah", "yep"}

    @staticmethod
    def _detect_intent(user_message: str) -> str:
        text = user_message.lower().strip()
        if any(
            phrase in text
            for phrase in (
                "what should i say",
                "how should i say",
                "should i text",
                "what do i text",
                "help me say",
                "clear the air",
            )
        ):
            return "communication"
        return ""

    def _promote_stage_for_intent(
        self,
        user_message: str,
        stage: str,
        last_stage: str,
        emotion: EmotionResult,
        ics_result: ICSResult,
    ) -> str:
        text = user_message.lower().strip()

        action_signals = (
            "what should i say",
            "what should i do",
            "should i text",
            "should i talk",
            "how should i say",
            "best way to say",
            "help me say",
            "help me figure out what to say",
            "clear the air",
        )
        reframe_signals = (
            "maybe",
            "i keep replaying",
            "i thought they were",
            "why would they",
            "i do not even know why",
            "maybe they are",
        )
        affirmation_signals = (
            "i think i know what to do",
            "i guess i should",
            "i will text them",
            "i will talk to them",
            "that makes sense",
            "okay i can do that",
        )

        if any(signal in text for signal in affirmation_signals):
            return "S5"

        if self._is_safe_confirmation(last_stage, user_message):
            return "S2"

        if any(signal in text for signal in action_signals):
            if ics_result.breakdown.get("conflict_detail", 0) >= 8 or self.memory.turn_count >= 3:
                return "S4"

        if any(signal in text for signal in reframe_signals):
            if stage in ("S1", "S2") and self.memory.turn_count >= 2:
                return "S3"

        if (
            last_stage == "S1"
            and stage == "S2"
            and (
                ics_result.breakdown.get("conflict_detail", 0) >= 8
                or ics_result.breakdown.get("context_richness", 0) >= 8
            )
            and emotion.label in ("anger", "sadness", "fear", "surprise")
        ):
            return "S1"

        return stage

    def _stabilize_emotion(self, emotion: EmotionResult) -> EmotionResult:
        if len(emotion.top_k) < 2 or not self.memory.recent_emotions:
            return emotion
        top = emotion.top_k[0]
        second = emotion.top_k[1]
        recent = self.memory.recent_emotions[-1]
        if recent in {str(top["label"]).lower(), str(second["label"]).lower()}:
            if abs(float(top["score"]) - float(second["score"])) < 0.15:
                chosen_score = next(
                    round(float(item["score"]), 4)
                    for item in emotion.top_k
                    if str(item["label"]).lower() == recent
                )
                return EmotionResult(
                    label=recent,
                    score=chosen_score,
                    top_k=[
                        {
                            "label": str(item["label"]).lower(),
                            "score": round(float(item["score"]), 4),
                        }
                        for item in emotion.top_k[:3]
                    ],
                    latency_ms=emotion.latency_ms,
                )
        return emotion


if __name__ == "__main__":
    layer = DynamicPromptingLayerV2(channel="whatsapp", output_mode="whatsapp_casual")
    conversation = [
        "hey",
        "she",
        "my best friend completely ignored me at school today in front of everyone",
        "i just feel like she hates me now, i always mess everything up",
        "what should i even say to her",
    ]

    for user_msg in conversation:
        payload = layer.prepare_turn(user_msg)
        print("\n" + "=" * 65)
        print(f"User        : {payload.user_message}")
        print(f"Emotion     : {payload.detected_emotion} ({payload.emotion_confidence:.0%})")
        print(f"ICS         : {payload.ics_score}/100 (tier {payload.ics_tier})")
        print(f"Stage       : {payload.selected_stage} | Mode: {payload.selected_mode}")
        print(f"Tone        : {payload.tone_dominant}")
        fake_llm_response = json.dumps(
            {
                "mode": payload.selected_mode,
                "stage": payload.selected_stage,
                "transition_reason": "simulated",
                "reply_style": "simulated",
                "reply_messages": [f"[Simulated reply for stage {payload.selected_stage}]"],
            }
        )
        parsed = json.loads(fake_llm_response)
        layer.record_reply(" ".join(parsed["reply_messages"]), payload.turn_index)
