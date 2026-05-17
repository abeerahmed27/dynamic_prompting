"""
leap/ — LEAP integration modules for Flank's dynamic prompting layer.

Import everything you need from here:
    from leap import (
        ConversationMemory,
        InputCompletenessScorer,
        ToneAgentBlender,
        CompassionCascade,
        FrameworkRetriever,
        OutputModeFormatter,
    )
"""
from leap.memory             import ConversationMemory, Turn
from leap.ics                import InputCompletenessScorer, ICSResult
from leap.tone_agents        import ToneAgentBlender, ToneBlend
from leap.compassion_cascade import CompassionCascade, CascadeState
from leap.strategic_rag      import FrameworkRetriever, Framework
from leap.output_modes       import OutputModeFormatter, TurnResult

__all__ = [
    "ConversationMemory", "Turn",
    "InputCompletenessScorer", "ICSResult",
    "ToneAgentBlender", "ToneBlend",
    "CompassionCascade", "CascadeState",
    "FrameworkRetriever", "Framework",
    "OutputModeFormatter", "TurnResult",
]
