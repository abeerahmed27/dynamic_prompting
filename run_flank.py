"""
Interactive CLI for testing the dynamic prompting pipeline without an LLM call.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from pipeline_v2 import DynamicPromptingLayerV2, EnrichedLLMPayload

_REPLY_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "S0": {"_any": ["I want to make sure you're safe right now. Are you safe?"]},
    "S1": {
        "reassurance": [
            "You sound really {emotion}. I hear you.",
            "Ugh, that's a lot to carry. Makes sense you'd feel {emotion}.",
            "Yeah, that tracks. You're not overreacting.",
        ],
        "clarity": [
            "Okay, I'm tracking. You sound genuinely {emotion}.",
            "Got it. That's a rough one.",
            "I hear you, and that makes sense.",
        ],
        "synthesis": [
            "There's a lot going on there. Makes sense you'd feel {emotion}.",
            "That sounds tangled, and honestly your reaction makes complete sense.",
            "A lot happening at once. No wonder it feels heavy.",
        ],
    },
    "S2": {
        "clarity": [
            "{scaffold_q}",
            "One thing I want to understand better: {scaffold_q}",
            "Can I ask: {scaffold_q}",
        ]
    },
    "S3": {
        "synthesis": [
            "I wonder if maybe they didn't realize how that would come across?",
            "There might be something else going on for them. Not excusing it, just worth considering.",
        ],
        "reassurance": [
            "Your reaction makes sense. I also wonder if there may be more going on on their side than you can see.",
            "You might be reading it right, but there could also be something happening with them that is spilling over here.",
        ],
    },
    "S4": {
        "action": [
            "{message_draft}",
            "Try this: {message_draft}",
            "One concrete thing: {message_draft}",
        ],
        "reassurance": [
            "If you want to keep it calm, start with this: {message_draft}",
            "A good next move would be: {message_draft}",
        ],
        "clarity": [
            "Here is the cleanest next step: {message_draft}",
            "What I would say first is: {message_draft}",
        ],
    },
    "S5": {
        "action": ["Here's a starting point: {framework_step}"],
        "reassurance": [
            "You are clearer now than you were before. {framework_step}",
            "That sounds like a steady next step. {framework_step}",
        ],
    },
    "S6": {"_any": ["Hey. What's going on?", "Hi. What's on your mind?"]},
}

_FALLBACK_REPLIES = [
    "Tell me more. I want to make sure I understand.",
    "I'm listening. What's going on?",
]

_EMOTION_PHRASES = {
    "anger": "angry",
    "sadness": "hurt",
    "fear": "on edge",
    "surprise": "thrown off",
    "disgust": "really put off",
    "joy": "lighter",
    "neutral": "flat",
}

_TRANSITION_REASONS = {
    "S0": "Safety keywords detected.",
    "S1": "Strong emotion present, so the system is listening first.",
    "S2": "More context is needed before moving forward.",
    "S3": "Enough context to offer a soft reframe.",
    "S4": "User is ready for a concrete technique.",
    "S5": "User is ready for a next step.",
    "S6": "Fresh start or greeting.",
}

_REPLY_STYLES = {
    "S0": "calm, direct, no advice",
    "S1": "warm validation, short",
    "S2": "one clean question",
    "S3": "soft reframe",
    "S4": "practical, step-by-step",
    "S5": "action-forward, encouraging",
    "S6": "warm welcome",
}


def _pick_templates(stage: str, tone_dominant: str) -> list[str]:
    stage_map = _REPLY_TEMPLATES.get(stage, {})
    return stage_map.get(tone_dominant) or stage_map.get("_any") or _FALLBACK_REPLIES


def _fill(template: str, payload: EnrichedLLMPayload, scaffold_q: str) -> str:
    framework_step = payload.framework_steps[0] if payload.framework_steps else "take one small step today"
    message_draft = "You could say: 'Hey, I felt hurt when you ignored me today. Did I do something wrong?'"
    if payload.selected_stage == "S4" and payload.framework_name == "Message Draft Builder":
        message_draft = "You could say: 'Hey, I felt hurt when you ignored me earlier. I just want to understand what happened.'"
    if payload.selected_stage == "S5" and payload.framework_name:
        framework_step = "You have a clearer plan now. Try sending one calm message when you're ready."
    if payload.selected_stage == "S2" and payload.selected_mode == "continue" and payload.detected_emotion in ("sadness", "fear"):
        scaffold_q = scaffold_q or "Are you somewhere safe with someone you trust right now?"
    return (
        template.replace("{emotion}", _EMOTION_PHRASES.get(payload.detected_emotion, payload.detected_emotion))
        .replace("{framework_step}", framework_step)
        .replace("{scaffold_q}", scaffold_q or "What happened exactly?")
        .replace("{message_draft}", message_draft)
    )


def generate_reply(payload: EnrichedLLMPayload) -> dict:
    scaffold_q = ""
    if payload.ics_tier == "C" and payload.ics_missing_dims:
        from leap.ics import _SCAFFOLD_QUESTIONS

        scaffold_q = _SCAFFOLD_QUESTIONS.get(payload.ics_missing_dims[0], "Can you tell me more?")
    elif payload.selected_stage == "S2":
        scaffold_q = "What happened exactly?"

    templates = _pick_templates(payload.selected_stage, payload.tone_dominant)
    replies = [_fill(t, payload, scaffold_q) for t in templates]
    return {
        "mode": payload.selected_mode,
        "stage": payload.selected_stage,
        "transition_reason": _TRANSITION_REASONS.get(payload.selected_stage, "routing decision"),
        "reply_style": _REPLY_STYLES.get(payload.selected_stage, "conversational"),
        "reply_messages": replies,
    }


def print_output(parsed: dict, mode: str, payload: EnrichedLLMPayload) -> None:
    replies = parsed.get("reply_messages", [])
    if mode in ("casual", "both"):
        print()
        for msg in replies:
            print(f"  Flank > {msg}")
        print()
    if mode in ("structured", "both"):
        print(
            f"\n  Emotion   : {payload.detected_emotion} ({payload.emotion_confidence:.0%})\n"
            f"  ICS       : {payload.ics_score}/100 (tier {payload.ics_tier})\n"
            f"  Stage     : {payload.selected_stage} | Mode: {payload.selected_mode}\n"
            f"  Tone      : {payload.tone_dominant}\n"
            f"  Framework : {payload.framework_name or 'none'}\n"
        )
        if mode == "structured" and replies:
            print(f"  Flank > {replies[0]}\n")


def run(initial_mode: str = "casual") -> None:
    output_mode = initial_mode
    last_payload: Optional[EnrichedLLMPayload] = None
    print("\n" + "=" * 60)
    print("  Flank - Pipeline Test")
    print("  Type /help for commands")
    print("=" * 60 + "\n")
    layer = DynamicPromptingLayerV2(channel="whatsapp", output_mode="whatsapp_casual")

    while True:
        try:
            user_input = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Session ended]")
            break

        if not user_input:
            continue
        lower = user_input.lower()

        if lower in ("/quit", "/exit"):
            print("\n  [Session ended]")
            break
        if lower == "/reset":
            layer = DynamicPromptingLayerV2(channel="whatsapp", output_mode="whatsapp_casual")
            last_payload = None
            print("\n  [Conversation reset]\n")
            continue
        if lower == "/help":
            print("\n  Commands: /quit /exit /reset /mode casual|structured|both /debug\n")
            continue
        if lower.startswith("/mode "):
            chosen = lower.split("/mode ", 1)[1].strip()
            if chosen in ("casual", "structured", "both"):
                output_mode = chosen
                print(f"\n  [Output mode -> {output_mode}]\n")
            continue
        if lower == "/debug":
            print(json.dumps(last_payload.__dict__ if last_payload else {"status": "No turn yet"}, indent=2, default=str))
            continue

        try:
            payload = layer.prepare_turn(user_input)
            last_payload = payload
        except Exception as exc:
            print(f"\n  [Pipeline error: {exc}]\n")
            continue

        parsed = generate_reply(payload)
        layer.record_reply(" ".join(parsed.get("reply_messages", [])), payload.turn_index)
        print_output(parsed, output_mode, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flank pipeline CLI")
    parser.add_argument("--mode", choices=["casual", "structured", "both"], default="casual")
    args = parser.parse_args()
    run(initial_mode=args.mode)
