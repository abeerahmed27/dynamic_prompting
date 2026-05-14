"""
Flank Emotion Layer — Dynamic Prompting System
================================================
Scope: emotion detection → stage routing → prompt selection → payload assembly.
The assembled payload is ready to send to any LLM; the actual call is out of scope here.

"""

import time
import logging
from dataclasses import dataclass

from transformers import pipeline

# ── Your prompt versions (keep in prompting_version.py) ────────────────────
# from prompting_version import (
#     SYSTEM_PROMPT_v1, SYSTEM_PROMPT_v2, SYSTEM_PROMPT_v3,
#     SYSTEM_PROMPT_v4, SYSTEM_PROMPT_v5,
# )
#
# For demo purposes, placeholder strings are used below.
SYSTEM_PROMPT_v1 = "You are a compassionate support assistant. [v1 rules here]"
SYSTEM_PROMPT_v2 = "You are a compassionate support assistant. [v2 rules here]"
SYSTEM_PROMPT_v3 = "You are a compassionate support assistant. [v3 rules here]"
SYSTEM_PROMPT_v4 = "You are a compassionate support assistant. [v4 rules here]"
SYSTEM_PROMPT_v5 = "You are a compassionate support assistant. [v5 rules here]"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Data schema ──────────────────────────────────────────────────────────────

@dataclass
class EmotionResult:
    label: str          # dominant emotion label
    score: float        # confidence 0–1
    top_k: list         # top-3 [{label, score}, ...]
    latency_ms: float


@dataclass
class LLMPayload:
    """
    Everything needed to make one LLM call.
    Hand this object to your LLM client of choice.
    """
    system_prompt: str          # full system prompt with injected emotion hint
    user_message: str           # original user input, unchanged
    detected_emotion: str       # for your records / logging
    emotion_confidence: float
    selected_stage: str         # S0–S6
    prompt_version: str         # v1–v5
    emotion_top_k: list
    emotion_latency_ms: float


# ── Emotion-to-stage routing ─────────────────────────────────────────────────

EMOTION_STAGE_MAP = {
    "fear":     "S0_or_S1",   # disambiguated by keyword scan below
    "anger":    "S1",
    "sadness":  "S1",
    "disgust":  "S3",
    "surprise": "S2",
    "joy":      "S5",
    "neutral":  "S2",
}

HARM_KEYWORDS = [
    "hurt myself", "kill", "end it", "suicide", "cut myself",
    "abuse", "threatened", "hit me", "danger",
]

GREETINGS = ["hi", "hello", "hey", "hiya", "yo"]


def route_stage(emotion: EmotionResult, user_text: str, last_stage: str) -> str:
    """
    Map an EmotionResult + raw text to a conversation stage (S0–S6).

    Priority order:
      1. Safety keywords  → S0 (always)
      2. Greeting opener  → S6
      3. Low confidence   → S1 (listen first, safest default)
      4. Emotion map      → base stage
      5. Anti-loop rule   → avoid S2 twice in a row
    """
    text_lower = user_text.lower().strip()

    # 1. Safety override
    if any(kw in text_lower for kw in HARM_KEYWORDS):
        return "S0"

    # 2. Fresh greeting
    if any(text_lower.startswith(g) for g in GREETINGS):
        return "S6"

    # 3. Low-confidence fallback
    if emotion.score < 0.50:
        return "S1"

    # 4. Emotion → base stage
    base = EMOTION_STAGE_MAP.get(emotion.label, "S1")

    # Disambiguate fear with no harm keywords → treat as venting
    if base == "S0_or_S1":
        return "S1"

    # 5. Anti-loop: don't stay on S2 twice
    if base == "S2" and last_stage == "S2":
        return "S1"

    return base


# ── Prompt version registry ──────────────────────────────────────────────────

PROMPT_VERSIONS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_v1,
    "v2": SYSTEM_PROMPT_v2,
    "v3": SYSTEM_PROMPT_v3,
    "v4": SYSTEM_PROMPT_v4,
    "v5": SYSTEM_PROMPT_v5,
}


def select_prompt_version(ab_variant: str = "v5") -> tuple[str, str]:
    """
    Returns (version_key, raw_system_prompt_string).
    Pass ab_variant to A/B test any version; falls back to v5.
    """
    key = ab_variant if ab_variant in PROMPT_VERSIONS else "v5"
    return key, PROMPT_VERSIONS[key]


# ── Emotion detection ────────────────────────────────────────────────────────

class EmotionDetector:
    """
    Wraps HuggingFace j-hartmann/emotion-english-distilroberta-base.
    Returns all 7 label scores; picks the highest as dominant emotion.
    """

    MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self):
        logger.info("Loading emotion model: %s", self.MODEL_ID)
        self.pipe = pipeline(
            "text-classification",
            model=self.MODEL_ID,
            top_k=None,       # return all label scores
            truncation=True,
            max_length=512,
        )

    def detect(self, text: str) -> EmotionResult:
        t0 = time.perf_counter()
        results = self.pipe(text)[0]                            # list of {label, score}
        latency_ms = (time.perf_counter() - t0) * 1000

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        top = sorted_results[0]

        return EmotionResult(
            label=top["label"].lower(),
            score=round(top["score"], 4),
            top_k=sorted_results[:3],
            latency_ms=round(latency_ms, 2),
        )


# ── Payload builder (the dynamic prompting core) ─────────────────────────────

def build_llm_payload(
    user_message: str,
    emotion: EmotionResult,
    stage: str,
    version_key: str,
    base_system_prompt: str,
) -> LLMPayload:
    """
    Injects a hidden internal context hint into the system prompt,
    then returns a ready-to-send LLMPayload.

    The hint is invisible to the user but guides the LLM's stage adherence.
    Format follows the convention already in your v5 prompt.
    """
    hint = (
        f"\n\n[Internal context — do NOT surface to user: "
        f"detected_emotion={emotion.label} "
        f"(confidence {emotion.score:.0%}), "
        f"routed_stage={stage}]"
    )

    return LLMPayload(
        system_prompt=base_system_prompt + hint,
        user_message=user_message,
        detected_emotion=emotion.label,
        emotion_confidence=emotion.score,
        selected_stage=stage,
        prompt_version=version_key,
        emotion_top_k=emotion.top_k,
        emotion_latency_ms=emotion.latency_ms,
    )


# ── Main orchestrator ────────────────────────────────────────────────────────

class DynamicPromptingLayer:
    """
    Stateful session wrapper.

    Call .prepare_turn(user_message) each turn.
    It returns an LLMPayload — send that to your LLM client.
    """

    def __init__(
        self,
        prompt_version: str = "v5",
        channel: str = "whatsapp",     # reserved for future per-channel prompt logic
    ):
        self.prompt_version = prompt_version
        self.channel = channel
        self.last_stage = "none"
        self.turn_index = 0

        self.emotion_detector = EmotionDetector()

    def prepare_turn(self, user_message: str) -> LLMPayload:
        """
        Full pre-LLM pipeline for one conversation turn:
          1. Detect emotion
          2. Route to stage
          3. Select prompt version
          4. Inject emotion hint into system prompt
          5. Return assembled LLMPayload (ready to send)
        """
        self.turn_index += 1

        # Step 1 — emotion detection
        emotion = self.emotion_detector.detect(user_message)
        logger.info(
            "[Turn %d] Emotion: %s (%.2f) | latency: %.0fms",
            self.turn_index, emotion.label, emotion.score, emotion.latency_ms,
        )

        # Step 2 — stage routing
        stage = route_stage(emotion, user_message, self.last_stage)
        logger.info("[Turn %d] Routed stage: %s", self.turn_index, stage)
        self.last_stage = stage     # update state for anti-loop rule next turn

        # Step 3 — prompt version selection
        version_key, base_system_prompt = select_prompt_version(self.prompt_version)

        # Step 4 + 5 — inject hint, build payload
        payload = build_llm_payload(
            user_message=user_message,
            emotion=emotion,
            stage=stage,
            version_key=version_key,
            base_system_prompt=base_system_prompt,
        )

        logger.info(
            "[Turn %d] Payload ready | version=%s stage=%s emotion=%s",
            self.turn_index, version_key, stage, emotion.label,
        )
        return payload


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    layer = DynamicPromptingLayer(prompt_version="v5", channel="whatsapp")

    conversation = [
        "hey",
        "my best friend completely ignored me at school today in front of everyone",
        "i just feel like she hates me now, i always mess everything up",
        "what should i even say to her",
    ]

    for message in conversation:
        payload = layer.prepare_turn(message)

        print("\n" + "─" * 60)
        print(f"User input      : {payload.user_message}")
        print(f"Detected emotion: {payload.detected_emotion} ({payload.emotion_confidence:.0%})")
        print(f"Top-3 emotions  : {payload.emotion_top_k}")
        print(f"Selected stage  : {payload.selected_stage}")
        print(f"Prompt version  : {payload.prompt_version}")
        print(f"Emotion latency : {payload.emotion_latency_ms:.0f}ms")
        print(f"\n--- System prompt (first 300 chars) ---")
        print(payload.system_prompt[:300] + "...")
        print(f"\n--- Ready to send to LLM ---")
        # Your LLM call goes here, e.g.:
        #   response = your_llm_client.call(
        #       system=payload.system_prompt,
        #       user=payload.user_message,
        #   )