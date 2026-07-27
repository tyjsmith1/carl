"""ElevenLabs TTS stage. Mirrors OpenAITTS's shape — same speak() signature,
same PCM-24k output contract so it drops into the existing router without
a resampling stage.

Model choice matters for latency: `eleven_flash_v2_5` runs ~200-300ms TTFB
which keeps conversation flow tight. `eleven_multilingual_v2` sounds better
but adds 500+ms per utterance — enough to feel laggy in a voice loop.

Cancellation: the SDK's stream() returns an async iterator; breaking out
closes the underlying HTTP stream and stops delivery. Note that ElevenLabs
bills per generated character regardless of whether we consumed them —
mid-utterance cancels don't refund. Mitigate the same way we do for OpenAI:
sentence-boundary chunking at the reasoner so we hand shorter texts.
"""

from __future__ import annotations

import asyncio
import os
import time

import structlog

from carl.config.schema import ElevenLabsTTSConfig
from carl.stages.tts.base import ChunkOutCallback

log = structlog.get_logger()


class ElevenLabsTTS:
    def __init__(self, config: ElevenLabsTTSConfig):
        # Lazy import — elevenlabs SDK pulls in httpx + pydantic models.
        from elevenlabs.client import AsyncElevenLabs

        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY not set in environment. "
                "Add it to .env — see .env.example."
            )
        self._client = AsyncElevenLabs(api_key=api_key)
        self._config = config

    async def speak(
        self,
        text: str,
        chunk_out: ChunkOutCallback,
        cancel: asyncio.Event,
    ) -> None:
        """Synthesize `text` and push PCM chunks to `chunk_out` until done or cancelled."""
        if cancel.is_set() or not text.strip():
            return

        # Build voice_settings — only include speed when non-default because
        # older model_ids may reject it, and 1.0 is the API default anyway.
        voice_settings: dict = {
            "stability": self._config.stability,
            "similarity_boost": self._config.similarity_boost,
            "style": self._config.style,
            "use_speaker_boost": self._config.use_speaker_boost,
        }
        if self._config.speed != 1.0:
            voice_settings["speed"] = self._config.speed

        t0 = time.monotonic()
        bytes_sent = 0
        log.info("tts_speak_begin", chars=len(text),
                 text_preview=text[:60].strip())

        try:
            # Do NOT swallow exceptions — orchestrator needs to see them.
            audio_stream = self._client.text_to_speech.stream(
                voice_id=self._config.voice_id,
                text=text,
                model_id=self._config.model_id,
                output_format=self._config.output_format,
                voice_settings=voice_settings,
            )
            async for chunk in audio_stream:
                if cancel.is_set():
                    break
                await chunk_out(chunk)
                bytes_sent += len(chunk)
        finally:
            # 24 kHz * 2 bytes/sample = 48 KB/sec — matches OpenAI TTS format.
            duration_ms = int(bytes_sent / (24000 * 2) * 1000)
            log.info("tts_speak_end", chars=len(text), bytes=bytes_sent,
                     audio_duration_ms=duration_ms,
                     elapsed_ms=int((time.monotonic() - t0) * 1000))
