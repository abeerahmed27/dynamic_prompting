import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotional_prompting_layer import EmotionResult, detect_mode, detect_safety_risk, route_stage
from leap.ics import InputCompletenessScorer
from pipeline_v2 import DynamicPromptingLayerV2
from run_flank import generate_reply


class RoutingTests(unittest.TestCase):
    def test_greeting_detection_requires_exact_greeting(self):
        emotion = EmotionResult(label="neutral", score=0.9, top_k=[], latency_ms=0)
        self.assertNotEqual(route_stage(emotion, "history keeps repeating", "none", "continue"), "S6")

    def test_negated_safety_phrase_is_not_crisis(self):
        self.assertFalse(detect_safety_risk("I am not going to kill myself"))

    def test_direct_safety_phrase_is_crisis(self):
        self.assertTrue(detect_safety_risk("I want to hurt myself"))

    def test_short_actually_does_not_force_new_topic(self):
        self.assertEqual(detect_mode("actually yeah that hurt", "S1", 2), "continue")

    def test_actually_without_real_shift_stays_continue(self):
        self.assertEqual(
            detect_mode("i thought they were the one person who actually understood me", "S1", 6),
            "continue",
        )


class ICSTests(unittest.TestCase):
    def test_context_can_fill_missing_dimensions(self):
        scorer = InputCompletenessScorer()
        result = scorer.score(
            "i feel betrayed",
            conversation_context="User said their best friend ignored them at school yesterday.",
            known_entities=["friend"],
            last_missing_dims=["context_richness", "conflict_detail"],
        )
        self.assertIn("context_richness", result.answered_by_context)
        self.assertGreaterEqual(result.score, 30)


class ReplyTests(unittest.TestCase):
    def test_s2_reply_does_not_leak_internal_hint(self):
        payload = type(
            "Payload",
            (),
            {
                "ics_tier": "B",
                "ics_missing_dims": [],
                "cascade_hint": "[Compassion cascade - internal]",
                "selected_stage": "S2",
                "selected_mode": "continue",
                "tone_dominant": "clarity",
                "framework_steps": (),
                "detected_emotion": "sadness",
            },
        )()
        parsed = generate_reply(payload)
        self.assertTrue(all("Compassion cascade" not in msg for msg in parsed["reply_messages"]))

    def test_s1_reply_uses_more_natural_wording(self):
        payload = type(
            "Payload",
            (),
            {
                "ics_tier": "B",
                "ics_missing_dims": [],
                "cascade_hint": "",
                "selected_stage": "S1",
                "selected_mode": "continue",
                "tone_dominant": "reassurance",
                "framework_steps": (),
                "detected_emotion": "anger",
            },
        )()
        parsed = generate_reply(payload)
        self.assertTrue(any("You sound really angry" in msg for msg in parsed["reply_messages"]))

    def test_s4_communication_reply_uses_message_draft(self):
        payload = type(
            "Payload",
            (),
            {
                "ics_tier": "B",
                "ics_missing_dims": [],
                "cascade_hint": "",
                "selected_stage": "S4",
                "selected_mode": "continue",
                "tone_dominant": "clarity",
                "framework_steps": ("Start with the specific moment that hurt.",),
                "framework_name": "Message Draft Builder",
                "detected_emotion": "anger",
            },
        )()
        parsed = generate_reply(payload)
        self.assertTrue(any("You could say:" in msg for msg in parsed["reply_messages"]))


class StagePromotionTests(unittest.TestCase):
    def test_action_request_promotes_to_s4(self):
        layer = DynamicPromptingLayerV2.__new__(DynamicPromptingLayerV2)
        layer.memory = type("Memory", (), {"turn_count": 4})()
        emotion = EmotionResult(label="sadness", score=0.8, top_k=[], latency_ms=0)
        ics_result = type("ICS", (), {"breakdown": {"conflict_detail": 12, "context_richness": 14}})()
        stage = layer._promote_stage_for_intent(
            "what should i say to clear the air without making it worse",
            "S1",
            "S1",
            emotion,
            ics_result,
        )
        self.assertEqual(stage, "S4")

    def test_reframe_signal_promotes_to_s3(self):
        layer = DynamicPromptingLayerV2.__new__(DynamicPromptingLayerV2)
        layer.memory = type("Memory", (), {"turn_count": 3})()
        emotion = EmotionResult(label="sadness", score=0.8, top_k=[], latency_ms=0)
        ics_result = type("ICS", (), {"breakdown": {"conflict_detail": 8, "context_richness": 14}})()
        stage = layer._promote_stage_for_intent(
            "maybe they are annoyed at me, but i do not even know why",
            "S1",
            "S1",
            emotion,
            ics_result,
        )
        self.assertEqual(stage, "S3")

    def test_detect_intent_for_communication(self):
        self.assertEqual(
            DynamicPromptingLayerV2._detect_intent("what should i say to them without making it worse"),
            "communication",
        )

    def test_safety_hold_keeps_s0_on_no(self):
        self.assertTrue(DynamicPromptingLayerV2._should_hold_safety("S0", "no"))

    def test_safe_confirmation_detected_after_s0(self):
        self.assertTrue(DynamicPromptingLayerV2._is_safe_confirmation("S0", "yes"))


if __name__ == "__main__":
    unittest.main()
