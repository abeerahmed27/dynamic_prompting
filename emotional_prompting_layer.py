"""
Flank Emotion Layer — Dynamic Prompting System
================================================
Stack:
  - Emotion detection : HuggingFace Transformers
  - Prompt orchestration : LangChain LCEL
  - Logging : Weights & Biases + MLflow
  - Evaluation : DeepEval

Install:
  pip install transformers torch langchain langchain-anthropic \
              wandb mlflow deepeval python-dotenv
"""

import os
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import wandb
import mlflow
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_anthropic import ChatAnthropic
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# ── Prompt versions from prompting_version.py ──────────────────────────────
from prompting_version import (
    SYSTEM_PROMPT_v1,
    SYSTEM_PROMPT_v2,
    SYSTEM_PROMPT_v3,
    SYSTEM_PROMPT_v4,
    SYSTEM_PROMPT_v5,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Data schemas ────────────────────────────────────────────────────────────

@dataclass
class EmotionResult:
    label: str
    score: float
    top_k: list[dict]
    latency_ms: float


@dataclass
class TurnLog:
    """Complete log record for one conversation turn — used by W&B + MLflow."""
    raw_input: str
    detected_emotion: str
    emotion_confidence: float
    emotion_top_k: list[dict]
    prompt_version: str
    selected_stage: str
    generated_system_prompt: str
    llm_output_raw: str
    llm_output_parsed: Optional[dict]
    emotion_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    session_id: str
    turn_index: int


# ── Emotion-to-stage routing ─────────────────────────────────────────────────

EMOTION_STAGE_MAP = {
    # High-urgency emotions with potential harm keywords
    "fear":    "S0_or_S1",   # disambiguated by keyword scan below
    "anger":   "S1",
    "sadness": "S1",
    "disgust": "S3",
    "surprise": "S2",
    "joy":     "S5",
    "neutral": "S2",
}

HARM_KEYWORDS = [
    "hurt myself", "kill", "end it", "suicide", "cut myself",
    "abuse", "threatened", "hit me", "danger",
]

def route_stage(emotion: EmotionResult, user_text: str, last_stage: str) -> str:
    """
    Map emotion detection output to a conversation stage (S0–S6).
    Incorporates anti-loop rules from SYSTEM_PROMPT_v5.
    """
    text_lower = user_text.lower()

    # Safety override — always highest priority
    if any(kw in text_lower for kw in HARM_KEYWORDS):
        return "S0"

    # Greeting / new session
    greetings = ["hi", "hello", "hey", "hiya", "yo"]
    if any(text_lower.strip().startswith(g) for g in greetings):
        return "S6"

    # Low-confidence fallback → listen first
    if emotion.score < 0.50:
        return "S1"

    base = EMOTION_STAGE_MAP.get(emotion.label, "S1")

    # Disambiguate fear: if no harm keywords, treat as S1 (venting)
    if base == "S0_or_S1":
        return "S1"

    # Anti-loop: avoid S2 twice in a row
    if base == "S2" and last_stage == "S2":
        return "S1"

    return base


# ── Prompt version selector ──────────────────────────────────────────────────

PROMPT_VERSIONS = {
    "v1": SYSTEM_PROMPT_v1,
    "v2": SYSTEM_PROMPT_v2,
    "v3": SYSTEM_PROMPT_v3,
    "v4": SYSTEM_PROMPT_v4,
    "v5": SYSTEM_PROMPT_v5,
}

def select_prompt_version(channel: str = "whatsapp", ab_variant: str = "v5") -> tuple[str, str]:
    """
    Returns (version_key, system_prompt_string).
    ab_variant allows A/B testing across v1–v5.
    """
    version = ab_variant if ab_variant in PROMPT_VERSIONS else "v5"
    return version, PROMPT_VERSIONS[version]


# ── Emotion detection layer ──────────────────────────────────────────────────

class EmotionDetector:
    """Wraps HuggingFace emotion classifier with latency measurement."""

    MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self):
        logger.info("Loading emotion model: %s", self.MODEL_ID)
        self.pipe = pipeline(
            "text-classification",
            model=self.MODEL_ID,
            top_k=None,          # return all label scores
            truncation=True,
            max_length=512,
        )

    def detect(self, text: str) -> EmotionResult:
        t0 = time.perf_counter()
        results = self.pipe(text)[0]          # list of {label, score}
        latency_ms = (time.perf_counter() - t0) * 1000

        # Sort descending by score
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        top = sorted_results[0]

        return EmotionResult(
            label=top["label"].lower(),
            score=round(top["score"], 4),
            top_k=sorted_results[:3],
            latency_ms=round(latency_ms, 2),
        )


# ── LangChain prompt builder ─────────────────────────────────────────────────

def build_langchain_chain(system_prompt: str, llm: ChatAnthropic):
    """
    Builds a simple LCEL chain:
      input dict → inject emotion hint → ChatPromptTemplate → LLM
    """
    # Inject emotion awareness as a soft hint inside the system prompt
    def inject_emotion_context(inputs: dict) -> dict:
        hint = (
            f"\n\n[Internal context — do NOT mention to user: "
            f"Detected emotion = {inputs['detected_emotion']} "
            f"(confidence {inputs['emotion_confidence']:.0%}), "
            f"last_stage = {inputs.get('last_stage', 'none')}]"
        )
        return {**inputs, "system_with_hint": system_prompt + hint}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_with_hint}"),
        ("human",  "{user_message}"),
    ])

    chain = (
        RunnableLambda(inject_emotion_context)
        | prompt
        | llm
    )
    return chain


# ── Main orchestrator ────────────────────────────────────────────────────────

class EmotionPromptingLayer:
    """
    Orchestrates: emotion detection → stage routing → prompt selection
    → LLM call → structured logging → evaluation.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        wandb_project: str = "flank-emotion-layer",
        mlflow_experiment: str = "flank-prompting",
        prompt_version: str = "v5",
        channel: str = "whatsapp",
    ):
        self.prompt_version = prompt_version
        self.channel = channel

        # Sub-components
        self.emotion_detector = EmotionDetector()
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=anthropic_api_key,
            max_tokens=300,
        )

        # Conversation state
        self.session_id = f"session_{int(time.time())}"
        self.turn_index = 0
        self.last_stage = "none"

        # Logging setup
        self._init_wandb(wandb_project)
        self._init_mlflow(mlflow_experiment)

    # ── Init logging ─────────────────────────────────────────────────────────

    def _init_wandb(self, project: str):
        wandb.init(
            project=project,
            config={
                "prompt_version": self.prompt_version,
                "channel": self.channel,
                "emotion_model": EmotionDetector.MODEL_ID,
                "llm_model": "claude-sonnet-4-20250514",
            },
            reinit=True,
        )

    def _init_mlflow(self, experiment: str):
        mlflow.set_experiment(experiment)
        self.mlflow_run = mlflow.start_run(run_name=f"{self.prompt_version}_{self.session_id}")
        mlflow.log_params({
            "prompt_version": self.prompt_version,
            "channel": self.channel,
            "emotion_model": EmotionDetector.MODEL_ID,
        })

    # ── Core turn processing ──────────────────────────────────────────────────

    def process_turn(self, user_message: str) -> dict:
        """
        Full pipeline for one conversation turn.
        Returns the parsed LLM response dict.
        """
        t_total_start = time.perf_counter()
        self.turn_index += 1

        # 1. Emotion detection
        emotion = self.emotion_detector.detect(user_message)
        logger.info("[Turn %d] Emotion: %s (%.2f)", self.turn_index, emotion.label, emotion.score)

        # 2. Stage routing
        stage = route_stage(emotion, user_message, self.last_stage)
        logger.info("[Turn %d] Routed to stage: %s", self.turn_index, stage)

        # 3. Prompt version selection
        version_key, system_prompt = select_prompt_version(
            channel=self.channel,
            ab_variant=self.prompt_version,
        )

        # 4. Build chain & invoke LLM
        chain = build_langchain_chain(system_prompt, self.llm)

        t_llm_start = time.perf_counter()
        response = chain.invoke({
            "user_message": user_message,
            "detected_emotion": emotion.label,
            "emotion_confidence": emotion.score,
            "last_stage": self.last_stage,
        })
        llm_latency_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)

        # 5. Parse LLM output
        raw_text = response.content
        parsed_output = self._safe_parse_json(raw_text)

        # Update conversation state
        if parsed_output:
            self.last_stage = parsed_output.get("stage", stage)

        total_latency_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

        # 6. Extract token usage from LangChain response metadata
        usage = getattr(response, "usage_metadata", {}) or {}
        input_tokens  = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # 7. Build log record
        log = TurnLog(
            raw_input=user_message,
            detected_emotion=emotion.label,
            emotion_confidence=emotion.score,
            emotion_top_k=emotion.top_k,
            prompt_version=version_key,
            selected_stage=stage,
            generated_system_prompt=system_prompt,
            llm_output_raw=raw_text,
            llm_output_parsed=parsed_output,
            emotion_latency_ms=emotion.latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            session_id=self.session_id,
            turn_index=self.turn_index,
        )

        # 8. Log everything
        self._log_wandb(log)
        self._log_mlflow(log)

        logger.info(
            "[Turn %d] Done. total_ms=%.0f | emotion_ms=%.0f | llm_ms=%.0f | tokens=%d+%d",
            self.turn_index, total_latency_ms, emotion.latency_ms,
            llm_latency_ms, input_tokens, output_tokens,
        )

        return {
            "log": asdict(log),
            "response": parsed_output or raw_text,
        }

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_wandb(self, log: TurnLog):
        """Log structured turn data to Weights & Biases."""
        wandb.log({
            "turn": log.turn_index,
            "emotion/label": log.detected_emotion,
            "emotion/confidence": log.emotion_confidence,
            "stage/selected": log.selected_stage,
            "stage/llm_chosen": (log.llm_output_parsed or {}).get("stage", "unknown"),
            "latency/emotion_ms": log.emotion_latency_ms,
            "latency/llm_ms": log.llm_latency_ms,
            "latency/total_ms": log.total_latency_ms,
            "tokens/input": log.input_tokens,
            "tokens/output": log.output_tokens,
            "tokens/total": log.input_tokens + log.output_tokens,
            "prompt_version": log.prompt_version,
        })

        # W&B Table for qualitative review
        table = wandb.Table(
            columns=["turn", "input", "emotion", "confidence", "stage", "reply"],
            data=[[
                log.turn_index,
                log.raw_input[:120],
                log.detected_emotion,
                round(log.emotion_confidence, 3),
                log.selected_stage,
                str((log.llm_output_parsed or {}).get("reply") or
                    (log.llm_output_parsed or {}).get("reply_messages", ""))[:200],
            ]],
        )
        wandb.log({"turn_table": table})

    def _log_mlflow(self, log: TurnLog):
        """Log metrics + artifacts to MLflow."""
        step = log.turn_index

        mlflow.log_metrics({
            "emotion_confidence": log.emotion_confidence,
            "emotion_latency_ms": log.emotion_latency_ms,
            "llm_latency_ms": log.llm_latency_ms,
            "total_latency_ms": log.total_latency_ms,
            "input_tokens": log.input_tokens,
            "output_tokens": log.output_tokens,
        }, step=step)

        # Log full turn as JSON artifact
        artifact_path = f"/tmp/turn_{step:04d}.json"
        with open(artifact_path, "w") as f:
            json.dump(asdict(log), f, indent=2)
        mlflow.log_artifact(artifact_path, artifact_path="turns")

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[dict]:
        """Safely parse the LLM's JSON output."""
        try:
            clean = text.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse LLM JSON output: %s", text[:200])
            return None

    def close(self):
        """Flush logs and close W&B + MLflow runs."""
        wandb.finish()
        mlflow.end_run()


# ── DeepEval evaluation harness ───────────────────────────────────────────────

def run_deepeval_evaluation(turn_logs: list[dict]):
    """
    Run DeepEval on a batch of logged turns.
    Uses AnswerRelevancyMetric + custom word-count check.

    turn_logs: list of TurnLog dicts (from asdict(log))
    """
    test_cases = []

    for log in turn_logs:
        parsed = log.get("llm_output_parsed") or {}
        reply = (
            parsed.get("reply") or
            " ".join(parsed.get("reply_messages", [])) or
            log.get("llm_output_raw", "")
        )

        test_cases.append(LLMTestCase(
            input=log["raw_input"],
            actual_output=reply,
            expected_output="",       # populate with golden set for supervised eval
            context=[
                f"detected_emotion={log['detected_emotion']}",
                f"confidence={log['emotion_confidence']}",
                f"stage={log['selected_stage']}",
            ],
        ))

    # Built-in relevancy metric
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")

    # Run evaluation
    evaluate(test_cases, [relevancy_metric])


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    layer = EmotionPromptingLayer(
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        prompt_version="v5",
        channel="whatsapp",
    )

    # Simulated conversation turns
    conversation = [
        "hey",
        "my best friend completely ignored me at school today in front of everyone",
        "i just feel like she hates me now, i always mess everything up",
        "what should i even say to her",
    ]

    all_logs = []
    for message in conversation:
        result = layer.process_turn(message)
        all_logs.append(result["log"])
        response = result["response"]

        print("\n" + "─" * 60)
        print(f"User   : {message}")
        if isinstance(response, dict):
            stage   = response.get("stage", "?")
            mode    = response.get("mode", "")
            replies = response.get("reply_messages") or [response.get("reply", "")]
            print(f"Stage  : {stage}  |  Mode: {mode}")
            for i, r in enumerate(replies, 1):
                print(f"Reply {i}: {r}")
        else:
            print(f"Reply  : {response}")

    # Run batch evaluation at end of session
    run_deepeval_evaluation(all_logs)

    layer.close()