"""
run_flank.py — Interactive CLI for Flank (no API key required)
===============================================================
Tests your pipeline end-to-end using a rule-based responder that reads
the pipeline's own output (stage, emotion, cascade, tone, ICS, framework)
to generate realistic Flank-style replies.

Nothing is mocked or hardcoded — every decision comes from your pipeline.

    python run_flank.py                    # replies only (default)
    python run_flank.py --mode structured  # full pipeline analysis + reply
    python run_flank.py --mode both        # replies + analysis panel

In-session commands:
    /quit | /exit    — end session
    /reset           — wipe memory, start fresh
    /mode casual     — replies only
    /mode structured — analysis + first reply
    /mode both       — replies + full analysis
    /debug           — print last payload metadata as JSON

Place alongside pipeline_v2.py, emotional_prompting_layer.py, and leap/.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

try:
    from pipeline_v2 import DynamicPromptingLayerV2, EnrichedLLMPayload
except ImportError as e:
    print(f"\n[ERROR] Could not import pipeline_v2: {e}")
    print("        Make sure pipeline_v2.py and leap/ modules are in the same directory.")
    sys.exit(1)


# ── Rule-based reply engine ────────────────────────────────────────────────────
# All replies are driven entirely by the pipeline's own output:
# stage, tone_dominant, ics_tier, framework_steps, detected_emotion.

_REPLY_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "S0": {
        "_any": [
            "Hey — I want to make sure you're okay right now. Are you safe?",
            "That sounds really heavy. Before anything else — are you safe right now?",
            "I'm here. Are you okay? That's the only thing that matters right now.",
        ],
    },
    "S1": {
        "reassurance": [
            "That sounds really {emotion}. I hear you.",
            "Ugh, that's a lot to carry. Makes sense you're feeling {emotion}.",
            "Yeah, that tracks. You're not overreacting.",
        ],
        "clarity": [
            "Okay, I'm tracking. That sounds genuinely {emotion}.",
            "Got it. That's a rough one.",
            "I hear you — and that makes sense.",
        ],
        "synthesis": [
            "There's a lot going on there. Makes sense you're feeling {emotion}.",
            "That sounds tangled — and honestly, your reaction makes complete sense.",
            "A lot happening at once. No wonder it feels heavy.",
        ],
    },
    "S2": {
        "clarity": [
            "{scaffold_q}",
            "One thing I want to understand better — {scaffold_q}",
            "Can I ask — {scaffold_q}",
        ],
        "reassurance": [
            "I want to make sure I've got the full picture — {scaffold_q}",
            "Before I say anything else — {scaffold_q}",
            "{scaffold_q}",
        ],
        "synthesis": [
            "To understand what's going on — {scaffold_q}",
            "One thing that would help me — {scaffold_q}",
            "{scaffold_q}",
        ],
    },
    "S3": {
        "synthesis": [
            "I wonder if — and tell me if this lands — maybe they didn't realise how that would come across?",
            "There might be something else going on for them. Not excusing it, just… worth considering.",
            "What if they're dealing with something you can't see right now? Not saying you're wrong — just a thought.",
        ],
        "reassurance": [
            "Your read might be right. But sometimes people act out of their own stress without meaning to hurt anyone.",
            "It might not be about you at all — even if it really feels like it is.",
            "Sometimes the people who matter most to us hurt us by accident.",
        ],
        "clarity": [
            "One possible read: they might not have meant it the way it landed.",
            "There could be something going on with them that has nothing to do with you.",
            "A different angle — what if they're struggling with something and it's coming out sideways?",
        ],
    },
    "S4": {
        "action": [
            "{framework_step}",
            "Try this: {framework_step}",
            "One concrete thing — {framework_step}",
        ],
        "clarity": [
            "Here's something practical: {framework_step}",
            "{framework_step} — that's a solid first move.",
            "Practical next step: {framework_step}",
        ],
        "reassurance": [
            "You've got this. Start with: {framework_step}",
            "One step at a time — {framework_step}",
            "{framework_step} — no pressure to do more than that.",
        ],
    },
    "S5": {
        "action": [
            "{framework_step} That's the move.",
            "You already know what to do. {framework_step}",
            "Here's a starting point: {framework_step}",
        ],
        "reassurance": [
            "You're more ready than you think. {framework_step}",
            "{framework_step} — and it's okay if it's not perfect.",
            "One small step: {framework_step}",
        ],
        "synthesis": [
            "You've done the hard part — figuring out how you feel. Now: {framework_step}",
            "{framework_step} That's really all it takes to start.",
            "The clarity you have now? Use it. {framework_step}",
        ],
    },
    "S6": {
        "_any": [
            "Hey! What's going on?",
            "Hi! What's on your mind?",
            "Hey — what's up? Tell me what's happening.",
        ],
    },
}

_FALLBACK_REPLIES = [
    "Tell me more — I want to make sure I understand.",
    "I'm listening. What's going on?",
    "Say more — I'm here.",
]

_TRANSITION_REASONS: dict[str, str] = {
    "S0": "Safety keywords detected — holding here until confirmed safe.",
    "S1": "Strong emotion present — listening before anything else.",
    "S2": "Need more context before moving forward.",
    "S3": "Enough context to offer a gentle reframe.",
    "S4": "User is ready for a concrete technique.",
    "S5": "User is action-ready — time to name the next step.",
    "S6": "Fresh start or greeting — opening the door.",
}

_REPLY_STYLES: dict[str, str] = {
    "S0": "calm, direct, no advice",
    "S1": "warm validation, short",
    "S2": "one clean question",
    "S3": "soft reframe with 'maybe'",
    "S4": "practical, step-by-step",
    "S5": "action-forward, encouraging",
    "S6": "warm welcome, open door",
}


def _pick_templates(stage: str, tone_dominant: str) -> list[str]:
    stage_map = _REPLY_TEMPLATES.get(stage, {})
    return (
        stage_map.get(tone_dominant)
        or stage_map.get("_any")
        or _FALLBACK_REPLIES
    )


def _fill(template: str, payload: EnrichedLLMPayload, scaffold_q: str) -> str:
    framework_step = (
        payload.framework_steps[0]
        if payload.framework_steps
        else "take one small step today"
    )
    return (
        template
        .replace("{emotion}", payload.detected_emotion)
        .replace("{framework_step}", framework_step)
        .replace("{scaffold_q}", scaffold_q or "What happened exactly?")
        .replace("{cascade_phase}", payload.cascade_phase)
    )


def generate_reply(payload: EnrichedLLMPayload) -> dict:
    """
    Produces a Flank-schema dict driven entirely by pipeline metadata.
    No LLM call — tests that your routing, cascade, tone, and ICS logic
    are producing sensible decisions before you wire up a real model.
    """
    stage = payload.selected_stage
    tone  = payload.tone_dominant

    # ICS tier C → pull the scaffold question computed by the pipeline
    scaffold_q = ""
    if payload.ics_tier == "C" and payload.ics_missing_dims:
        from leap.ics import _SCAFFOLD_QUESTIONS
        scaffold_q = _SCAFFOLD_QUESTIONS.get(payload.ics_missing_dims[0], "Can you tell me more?")

    templates = _pick_templates(stage, tone)
    replies   = [_fill(t, payload, scaffold_q) for t in templates]

    return {
        "mode":              payload.selected_mode,
        "stage":             stage,
        "transition_reason": _TRANSITION_REASONS.get(stage, "routing decision"),
        "reply_style":       _REPLY_STYLES.get(stage, "conversational"),
        "reply_messages":    replies,
    }


# ── Output rendering ───────────────────────────────────────────────────────────
def print_output(parsed: dict, mode: str, payload: EnrichedLLMPayload) -> None:
    replies: list[str] = parsed.get("reply_messages", [])

    if mode in ("casual", "both"):
        print()
        for msg in replies:
            print(f"  Flank › {msg}")
        print()

    if mode in ("structured", "both"):
        tone_str    = " / ".join(f"{k}={v:.0%}" for k, v in payload.tone_blend.items())
        missing_str = (f" — missing: {', '.join(payload.ics_missing_dims)}"
                       if payload.ics_missing_dims else "")
        steps_str   = (f"\n  Steps     : {'; '.join(payload.framework_steps[:2])}"
                       if payload.framework_steps else "")
        print(
            f"\n  ── Pipeline Analysis ─────────────────────────────────\n"
            f"  Emotion   : {payload.detected_emotion} ({payload.emotion_confidence:.0%})"
            f"  |  latency {payload.emotion_latency_ms:.0f}ms\n"
            f"  ICS       : {payload.ics_score}/100 (tier {payload.ics_tier}){missing_str}\n"
            f"  Stage     : {payload.selected_stage}  |  Mode: {payload.selected_mode}\n"
            f"  Cascade   : {payload.cascade_phase} ({payload.cascade_sequence})\n"
            f"  Tone      : {payload.tone_dominant} — {tone_str}\n"
            f"  Framework : {payload.framework_name or 'none'}{steps_str}\n"
            f"  Reason    : {parsed.get('transition_reason', '—')}\n"
            f"  ──────────────────────────────────────────────────────\n"
        )
        if mode == "structured" and replies:
            print(f"  Flank › {replies[0]}\n")


# ── Main REPL ──────────────────────────────────────────────────────────────────
def run(initial_mode: str = "casual") -> None:
    output_mode  = initial_mode
    last_payload: Optional[EnrichedLLMPayload] = None

    print("\n" + "═" * 60)
    print("  Flank — Pipeline Test  (no API key needed)")
    print("  Type /help for commands")
    print("═" * 60 + "\n")

    layer = DynamicPromptingLayerV2(channel="whatsapp", output_mode="whatsapp_casual")

    while True:
        try:
            user_input = input("  You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Session ended]")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # ── Commands ───────────────────────────────────────────────────────────
        if lower in ("/quit", "/exit"):
            print("\n  [Session ended]")
            break

        if lower == "/reset":
            layer        = DynamicPromptingLayerV2(channel="whatsapp", output_mode="whatsapp_casual")
            last_payload = None
            print("\n  [Conversation reset — fresh start]\n")
            continue

        if lower == "/help":
            print(
                "\n  Commands:\n"
                "    /quit | /exit        — end session\n"
                "    /reset               — clear memory, start over\n"
                "    /mode casual         — show replies only\n"
                "    /mode structured     — show pipeline analysis + first reply\n"
                "    /mode both           — show replies + full analysis\n"
                "    /debug               — print last payload as JSON\n"
            )
            continue

        if lower.startswith("/mode "):
            chosen = lower.split("/mode ", 1)[1].strip()
            if chosen in ("casual", "structured", "both"):
                output_mode = chosen
                print(f"\n  [Output mode → {output_mode}]\n")
            else:
                print("  [Unknown mode — use: casual | structured | both]\n")
            continue

        if lower == "/debug":
            if last_payload is None:
                print("  [No turn yet]\n")
            else:
                p = last_payload
                print(json.dumps({
                    "stage":           p.selected_stage,
                    "mode":            p.selected_mode,
                    "emotion":         p.detected_emotion,
                    "confidence":      f"{p.emotion_confidence:.0%}",
                    "emotion_top_k":   p.emotion_top_k,
                    "ics_score":       p.ics_score,
                    "ics_tier":        p.ics_tier,
                    "ics_missing":     p.ics_missing_dims,
                    "cascade_phase":   p.cascade_phase,
                    "cascade_seq":     p.cascade_sequence,
                    "tone_dominant":   p.tone_dominant,
                    "tone_blend":      {k: f"{v:.0%}" for k, v in p.tone_blend.items()},
                    "framework":       p.framework_name or None,
                    "framework_steps": list(p.framework_steps),
                    "session_emotion": p.session_dominant_emotion,
                    "memory_turns":    layer.memory.turn_count,
                }, indent=2))
                print()
            continue

        # ── Pipeline turn ──────────────────────────────────────────────────────
        try:
            payload      = layer.prepare_turn(user_input)
            last_payload = payload
        except Exception as e:
            print(f"\n  [Pipeline error: {e}]\n")
            continue

        # ── Generate reply from pipeline metadata ──────────────────────────────
        parsed     = generate_reply(payload)
        reply_text = " ".join(parsed.get("reply_messages", []))

        # Store in memory so context accumulates correctly turn-to-turn
        layer.record_reply(reply_text=reply_text, turn_index=layer.memory.turn_count)

        print_output(parsed, output_mode, payload)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flank pipeline CLI — no API key needed")
    parser.add_argument(
        "--mode",
        choices=["casual", "structured", "both"],
        default="casual",
        help="Output mode: casual (replies only), structured (analysis + reply), both",
    )
    args = parser.parse_args()
    run(initial_mode=args.mode)
