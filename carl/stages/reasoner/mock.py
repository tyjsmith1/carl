"""Mock reasoner. Echoes back the user transcript for verification."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from carl.stages.reasoner.base import (
    OnFillerNeededCallback,
    OnToolCallCallback,
    ReasonerTurn,
    SpokenChunk,
)


class EchoReasoner:
    def __init__(self, config: Any = None):
        self._aborted = False

    async def turn(
        self,
        request: ReasonerTurn,
        on_tool_call: OnToolCallCallback | None = None,
        on_filler_needed: OnFillerNeededCallback | None = None,
    ) -> AsyncIterator[SpokenChunk]:
        self._aborted = False
        text = f"You said: {request.user_transcript}."
        yield SpokenChunk(text=text, is_final=True)

    async def abort(self) -> None:
        self._aborted = True
