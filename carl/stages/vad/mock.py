"""Mock VAD. No-op — the mock pipeline uses CannedTranscriptSTT which
returns a transcript synchronously on close(), so the FSM triggers
speech_end explicitly rather than via VAD in mock mode."""

from __future__ import annotations

from typing import Any

from carl.stages.vad.base import VADEvent


class NoopVAD:
    """Returns None on every frame. Real VAD arrives in Milestone 2."""

    def __init__(self, config: Any = None):
        self.sample_rate = 16000
        self.frame_length = 512

    async def process(self, pcm_i16: bytes) -> VADEvent | None:
        return None

    def reset(self) -> None:
        return
