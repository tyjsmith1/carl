"""Mock TTS. Prints the text to stderr instead of speaking.
`chunk_out` still receives one dummy PCM chunk so the pipeline's playback
plumbing sees data (useful when the real playback layer arrives later)."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from carl.stages.tts.base import ChunkOutCallback


class PrintTTS:
    def __init__(self, config: Any = None):
        pass

    async def speak(
        self,
        text: str,
        chunk_out: ChunkOutCallback,
        cancel: asyncio.Event,
    ) -> None:
        if cancel.is_set():
            return
        # Print with a visible prefix so it's easy to eyeball in the terminal.
        print(f"[TTS] {text}", file=sys.stderr, flush=True)
        # Emit a placeholder PCM chunk so playback layer sees something.
        # 24 kHz * 2 bytes * 0.1s = 4800 bytes of silence.
        await chunk_out(b"\x00" * 4800)
