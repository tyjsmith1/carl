"""STT stage interface. See docs/specs/pipeline.md (STT section)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    is_final: bool
    confidence: float | None = None


OnTranscriptCallback = Callable[[Transcript], Awaitable[None]]


class STTSession(Protocol):
    async def send(self, pcm_i16: bytes) -> None: ...

    async def close(self) -> str:
        """Close the stream and return the final full transcript."""
        ...


class STT(Protocol):
    async def open_stream(self, on_transcript: OnTranscriptCallback) -> STTSession: ...
