"""Wake word stage interface. See docs/specs/pipeline.md (Wake word section)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

OnWakeCallback = Callable[[dict[str, Any]], Awaitable[None]]


class WakeWord(Protocol):
    """Runs continuously in the background. Fires `on_wake` when the wake
    trigger is detected. Audio-driven implementations (OpenWakeWord) receive
    frames via `feed()`; trigger-driven implementations (mock stdin,
    physical button) ignore `feed()` and use their own event source.
    """

    async def start(self, on_wake: OnWakeCallback) -> None:
        """Begin listening. Idempotent — calling twice is a bug but must not raise."""
        ...

    async def stop(self) -> None:
        """Cleanly shut down."""
        ...

    def wants_audio(self) -> bool:
        """Return True if this impl needs audio frames via feed()."""
        ...

    def feed(self, pcm_i16: bytes) -> None:
        """Called by the audio router only if wants_audio() is True."""
        ...
