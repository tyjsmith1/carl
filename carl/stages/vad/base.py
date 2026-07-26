"""VAD stage interface. See docs/specs/pipeline.md (VAD section)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SpeechStart:
    timestamp: float


@dataclass(frozen=True, slots=True)
class SpeechEnd:
    timestamp: float


VADEvent = SpeechStart | SpeechEnd


class VAD(Protocol):
    sample_rate: int
    frame_length: int

    async def process(self, pcm_i16: bytes) -> VADEvent | None:
        """Called with one audio frame. Returns None if nothing has changed;
        emits SpeechStart on onset and SpeechEnd on offset."""
        ...

    def reset(self) -> None:
        """Reset any internal state between sessions."""
        ...
