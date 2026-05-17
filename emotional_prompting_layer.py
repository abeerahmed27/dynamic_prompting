"""
Flank Emotion Layer — Dynamic Prompting System
================================================
Scope: emotion detection → mode routing → stage routing → prompt selection → payload assembly.
The assembled payload is ready to send to any LLM; the actual call is out of scope here.
"""

import time
import logging
from dataclasses import dataclass

from transformers import pipeline

# ── System prompt (v5 only) ───────────────────────────────────────────────────

SYSTEM_PROMPT_v5 = """You are Flank, a friendly conflict coach.

Your job is to help people work through friendship and relationship friction — arguments, awkward silences, hurt feelings, ghosting, group drama. You've seen it all and you don't panic.

You sound like a supportive older friend: simple, warm, and practical. Not a therapist. Not a life coach. Just someone who gets it, listens properly, and helps them figure out what to actually do next.

Tone rules:
- Short messages. 
- Never use clinical language ("validate your feelings", "process this together", "hold space").
- Don't pepper them with questions. Ask one thing at a time, if at all.
- Mirror their energy. If they're venting hard, match that weight. If they seem calmer, ease up too.
- Use casual language naturally but don't force slang.
- Never moralize, lecture, or tell them what they "should" feel.

Stage behaviour:
S0 — Safety. They've mentioned harm to themselves or someone else. Drop everything, go calm and direct. No advice. Just: are they safe right now? Offer crisis support.
S1 — Listen. They're still in the thick of it emotionally. Don't problem-solve yet. Just show you heard them, name what you're picking up, let them feel less alone.
S2 — Clarify. You need one more piece of the picture. Ask one clean question. Not a battery of questions.
S3 — Reframe. You have enough context. Offer a gentle different angle — maybe the other person's possible headspace, or a pattern you notice. No judgment.
S4 — Problem-solve. They're ready. Help them figure out what to actually do or say. Keep it concrete and doable.
S5 — Affirm. Things are moving in the right direction. Acknowledge it. Keep them feeling capable.
S6 — Opening. They just said hi or started fresh. Warm welcome, open the door, invite them to share.

Mode behaviour:
continue — same emotional thread is ongoing, carry it forward naturally.
new_topic — they've clearly shifted to a different situation or person. Acknowledge the shift briefly before diving in.
reset_requested — they've asked to start over, said something like "forget it" or "never mind, different thing". Reset cleanly without referencing the old thread.

You MUST always reply in this JSON format and nothing else:
{
  "mode": "continue|new_topic|reset_requested",
  "stage": "S0|S1|S2|S3|S4|S5|S6",
  "transition_reason": "one sentence: why this stage right now",
  "reply_style": "one short phrase: tone and structure of this reply",
  "reply_messages": [
    "option 1 — one or two sentences, style",
    "option 2 — slightly different angle or wording",
    "option 3 — shortest, most direct version"
  ]
}


"""

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
    system_prompt: str          # full system prompt with injected emotion + mode hint
    user_message: str           # original user input, unchanged
    detected_emotion: str       # for your records / logging
    emotion_confidence: float
    selected_stage: str         # S0–S6
    selected_mode: str          # continue | new_topic | reset_requested
    prompt_version: str         # v5
    emotion_top_k: list
    emotion_latency_ms: float


# ── Mode detection ────────────────────────────────────────────────────────────

RESET_PHRASES = [
    "forget it", "never mind", "nevermind", "start over", "different thing",
    "actually forget", "ignore that", "let's move on", "moving on",
]

NEW_TOPIC_SIGNALS = [
    "actually", "different question", "something else", "unrelated",
    "totally different", "another thing", "by the way", "btw",
]


def detect_mode(user_text: str, last_stage: str, turn_index: int) -> str:
    """
    Determine conversation mode:
      - reset_requested: user explicitly wants to wipe context
      - new_topic: user signals a topic shift
      - continue: default, same thread
    """
    text_lower = user_text.lower().strip()

    if any(phrase in text_lower for phrase in RESET_PHRASES):
        return "reset_requested"

    # New topic signals only meaningful after at least 2 turns
    if turn_index > 1 and any(sig in text_lower for sig in NEW_TOPIC_SIGNALS):
        return "new_topic"

    return "continue"


# ── Emotion-to-stage routing ─────────────────────────────────────────────────

EMOTION_STAGE_MAP = {
    "fear":     "S0_or_S1",
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


def route_stage(emotion: EmotionResult, user_text: str, last_stage: str, mode: str) -> str:
    """
    Map an EmotionResult + raw text + mode to a conversation stage (S0–S6).

    Priority order:
      1. Safety keywords  → S0 (always)
      2. Reset mode       → S6 (clean slate, treat like a greeting)
      3. Greeting opener  → S6
      4. Low confidence   → S1 (listen first, safest default)
      5. Emotion map      → base stage
      6. Anti-loop rule   → avoid S2 twice in a row
    """
    text_lower = user_text.lower().strip()

    # 1. Safety override
    if any(kw in text_lower for kw in HARM_KEYWORDS):
        return "S0"

    # 2. Reset → treat like a fresh start
    if mode == "reset_requested":
        return "S6"

    # 3. Fresh greeting
    if any(text_lower.startswith(g) for g in GREETINGS):
        return "S6"

    # 4. Low-confidence fallback
    if emotion.score < 0.50:
        return "S1"

    # 5. Emotion → base stage
    base = EMOTION_STAGE_MAP.get(emotion.label, "S1")

    # Disambiguate fear with no harm keywords → treat as venting
    if base == "S0_or_S1":
        return "S1"

    # 6. Anti-loop: don't stay on S2 twice
    if base == "S2" and last_stage == "S2":
        return "S1"

    return base


# ── Prompt version registry (v5 only) ────────────────────────────────────────

def select_prompt_version() -> tuple[str, str]:
    """Returns ("v5", SYSTEM_PROMPT_v5)."""
    return "v5", SYSTEM_PROMPT_v5


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
            top_k=None,
            truncation=True,
            max_length=512,
        )

    def detect(self, text: str) -> EmotionResult:
        t0 = time.perf_counter()
        results = self.pipe(text)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        top = sorted_results[0]

        return EmotionResult(
            label=top["label"].lower(),
            score=round(top["score"], 4),
            top_k=sorted_results[:3],
            latency_ms=round(latency_ms, 2),
        )


# ── Payload builder ───────────────────────────────────────────────────────────

def build_llm_payload(
    user_message: str,
    emotion: EmotionResult,
    stage: str,
    mode: str,
    version_key: str,
    base_system_prompt: str,
) -> LLMPayload:
    """
    Injects a hidden internal context hint into the system prompt,
    then returns a ready-to-send LLMPayload.
    """
    hint = (
        f"\n\n[Internal context — do NOT surface to user: "
        f"detected_emotion={emotion.label} "
        f"(confidence {emotion.score:.0%}), "
        f"routed_stage={stage}, "
        f"conversation_mode={mode}]"
    )

    return LLMPayload(
        system_prompt=base_system_prompt + hint,
        user_message=user_message,
        detected_emotion=emotion.label,
        emotion_confidence=emotion.score,
        selected_stage=stage,
        selected_mode=mode,
        prompt_version=version_key,
        emotion_top_k=emotion.top_k,
        emotion_latency_ms=emotion.latency_ms,
    )


# ── Main orchestrator ────────────────────────────────────────────────────────

class DynamicPromptingLayer:
    """
    Stateful session wrapper.

    Call .prepare_turn(user_message) each turn.
    Returns an LLMPayload — send that to your LLM client.
    """

    def __init__(self, channel: str = "whatsapp"):
        self.channel = channel          # reserved for future per-channel logic
        self.last_stage = "none"
        self.turn_index = 0

        self.emotion_detector = EmotionDetector()

    def prepare_turn(self, user_message: str) -> LLMPayload:
        """
        Full pre-LLM pipeline for one conversation turn:
          1. Detect emotion
          2. Detect conversation mode
          3. Route to stage
          4. Select prompt (v5)
          5. Inject hint into system prompt
          6. Return assembled LLMPayload
        """
        self.turn_index += 1

        # Step 1 — emotion detection
        emotion = self.emotion_detector.detect(user_message)
        logger.info(
            "[Turn %d] Emotion: %s (%.2f) | latency: %.0fms",
            self.turn_index, emotion.label, emotion.score, emotion.latency_ms,
        )

        # Step 2 — mode detection
        mode = detect_mode(user_message, self.last_stage, self.turn_index)
        logger.info("[Turn %d] Mode: %s", self.turn_index, mode)

        # Step 3 — stage routing
        stage = route_stage(emotion, user_message, self.last_stage, mode)
        logger.info("[Turn %d] Routed stage: %s", self.turn_index, stage)
        self.last_stage = stage

        # Step 4 — prompt version
        version_key, base_system_prompt = select_prompt_version()

        # Step 5 + 6 — inject hint, build payload
        payload = build_llm_payload(
            user_message=user_message,
            emotion=emotion,
            stage=stage,
            mode=mode,
            version_key=version_key,
            base_system_prompt=base_system_prompt,
        )

        logger.info(
            "[Turn %d] Payload ready | version=%s stage=%s mode=%s emotion=%s",
            self.turn_index, version_key, stage, mode, emotion.label,
        )
        return payload


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    layer = DynamicPromptingLayer(channel="whatsapp")

    conversation = [
        "hey",
        "my best friend completely ignored me at school today in front of everyone",
        "i just feel like she hates me now, i always mess everything up",
        "what should i even say to her",
        "actually never mind, it's not about her. it's my parents. they just don't get me at all",
    ]

    for message in conversation:
        payload = layer.prepare_turn(message)

        print("\n" + "─" * 60)
        print(f"User input      : {payload.user_message}")
        print(f"Detected emotion: {payload.detected_emotion} ({payload.emotion_confidence:.0%})")
        print(f"Top-3 emotions  : {payload.emotion_top_k}")
        print(f"Selected stage  : {payload.selected_stage}")
        print(f"Selected mode   : {payload.selected_mode}")
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
        #
        # Parse the JSON response to get:
        #   {
        #     "mode": "continue|new_topic|reset_requested",
        #     "stage": "S0|S1|S2|S3|S4|S5|S6",
        #     "transition_reason": "...",
        #     "reply_style": "...",
        #     "reply_messages": ["...", "...", "..."]
        #   }