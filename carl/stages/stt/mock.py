"""Mock STT. Cycles through a canned list of transcripts. Ignores audio.

Env var MOCK_STT_TRANSCRIPTS (semicolon-separated) overrides the list.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

from carl.stages.stt.base import OnTranscriptCallback, Transcript

_DEFAULT_TRANSCRIPTS = [
    "when was chicago founded",
    "what time is it",
    "tell me a joke",
    "hello",
]


class CannedTranscriptSTT:
    def __init__(self, config: Any = None):
        transcripts_env = os.environ.get("MOCK_STT_TRANSCRIPTS")
        if transcripts_env:
            self._transcripts = [t.strip() for t in transcripts_env.split(";") if t.strip()]
        else:
            self._transcripts = list(_DEFAULT_TRANSCRIPTS)
        self._cycle = itertools.cycle(self._transcripts)

    async def open_stream(self, on_transcript: OnTranscriptCallback) -> CannedSTTSession:
        text = next(self._cycle)
        return CannedSTTSession(text, on_transcript)


class CannedSTTSession:
    def __init__(self, text: str, on_transcript: OnTranscriptCallback):
        self._text = text
        self._on_transcript = on_transcript
        self._sent_partial = False

    async def send(self, pcm_i16: bytes) -> None:
        # First send() triggers a fake partial for realism; subsequent calls no-op.
        if not self._sent_partial:
            self._sent_partial = True
            await self._on_transcript(Transcript(text=self._text[: len(self._text) // 2],
                                                 is_final=False))

    async def close(self) -> str:
        await self._on_transcript(Transcript(text=self._text, is_final=True, confidence=1.0))
        return self._text
