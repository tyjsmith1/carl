"""TTS stage interface. See docs/specs/pipeline.md (TTS section)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

ChunkOutCallback = Callable[[bytes], Awaitable[None]]


class TTS(Protocol):
    async def speak(
        self,
        text: str,
        chunk_out: ChunkOutCallback,
        cancel: asyncio.Event,
    ) -> None:
        """Synthesize `text`. Push PCM chunks to chunk_out. Break on cancel."""
        ...
