"""FSM data types. See docs/specs/conversation-state.md.

The Pipeline class in orchestrator.py owns the actual transition dispatch;
this module just defines the vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class State(StrEnum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


# Event type constants — string values so they can be logged directly.
WAKE_DETECTED = "wake_detected"
SPEECH_START = "speech_start"
SPEECH_END = "speech_end"
UTTERANCE_TIMEOUT = "utterance_timeout"
NO_SPEECH_TIMEOUT = "no_speech_timeout"
STT_ERROR = "stt_error"
LLM_FINAL_TEXT = "llm_final_text"
LLM_TIMEOUT = "llm_timeout"
LLM_ERROR = "llm_error"
TTS_COMPLETE = "tts_complete"
TTS_INTERRUPTED = "tts_interrupted"
TTS_ERROR = "tts_error"
FOLLOWUP_EXPIRED = "followup_expired"
ERROR_ANNOUNCED = "error_announced"
SHUTDOWN = "shutdown"
