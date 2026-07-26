"""OpenAI TTS stage. See docs/specs/pipeline.md.

Streaming PCM output so playback can start before the whole utterance is
synthesized, and mid-utterance cancellation (barge-in) actually cuts audio
at the byte level rather than waiting for a codec frame boundary.

Format contract:
- `response_format="pcm"` → 16-bit signed little-endian mono at 24 kHz.
- Router.OUT_SAMPLE_RATE is 24000 to match — no resampling needed.

Cancellation:
- The `cancel` asyncio.Event is checked in the chunk loop.
- On set, we `break`, which triggers the `async with` to exit and closes
  the underlying HTTP stream (stops delivery).
- OpenAI may still bill for the full requested text — the HTTP close stops
  delivery but not upstream generation. Mitigate by chunking long replies at
  sentence boundaries (which the reasoner already does).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog

from carl.config.schema import OpenAITTSConfig
from carl.stages.tts.base import ChunkOutCallback

log = structlog.get_logger()


class OpenAITTS:
    def __init__(self, config: OpenAITTSConfig):
        # Lazy import — openai SDK loads http transport.
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set in environment. "
                "Add it to .env — see .env.example."
            )
        self._client = AsyncOpenAI(api_key=api_key)
        self._config = config
        self._instructions = self._load_instructions()

    def _load_instructions(self) -> str | None:
        if self._config.instructions_path is None:
            return None
        path = Path(self._config.instructions_path)
        if not path.exists():
            log.warning("tts_instructions_missing", path=str(path))
            return None
        return path.read_text().strip() or None

    async def speak(
        self,
        text: str,
        chunk_out: ChunkOutCallback,
        cancel: asyncio.Event,
    ) -> None:
        """Synthesize `text` and push PCM chunks to `chunk_out` until done or cancelled."""
        if cancel.is_set() or not text.strip():
            return

        # Build the params dict — gpt-4o-mini-tts accepts `instructions`,
        # older tts-1 models reject it.
        params: dict = {
            "model": self._config.model,
            "voice": self._config.voice,
            "input": text,
            "response_format": self._config.response_format,   # "pcm" per config
        }
        if self._instructions and "gpt-4o" in self._config.model:
            params["instructions"] = self._instructions

        import time
        t0 = time.monotonic()
        bytes_sent = 0
        log.info("tts_speak_begin", chars=len(text),
                 text_preview=text[:60].strip())

        # Do NOT swallow exceptions here — the orchestrator needs to see them
        # so it can transition to ERROR state / speak an apology instead of
        # cheerfully logging tts_complete for silence.
        try:
            async with self._client.audio.speech.with_streaming_response.create(**params) as response:
                async for chunk in response.iter_bytes(chunk_size=4096):
                    if cancel.is_set():
                        break
                    await chunk_out(chunk)
                    bytes_sent += len(chunk)
        finally:
            # ~24 kHz * 2 bytes/sample = 48 KB/sec of audio
            duration_ms = int(bytes_sent / (24000 * 2) * 1000)
            log.info("tts_speak_end", chars=len(text), bytes=bytes_sent,
                     audio_duration_ms=duration_ms,
                     elapsed_ms=int((time.monotonic() - t0) * 1000))
