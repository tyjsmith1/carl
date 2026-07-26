"""Deepgram Nova-3 streaming STT stage. See docs/specs/pipeline.md.

Uses the deepgram-sdk v7+ API surface: `client.listen.v2.connect(...)` as a
context manager. Older v3/v4 patterns (`listen.asynclive.v("1")`,
`LiveOptions`, `LiveTranscriptionEvents`) are dead — do not port them.

Session lifecycle:
    stt = DeepgramSTT(cfg)
    session = await stt.open_stream(on_transcript)   # opens WebSocket
    await session.send(pcm_bytes)                    # push audio frames
    ...
    final_text = await session.close()               # tears down, returns joined transcript

The Deepgram library's connection object is a context manager that spins up
its own event loop for the WebSocket. We enter it manually via `__enter__`
so we can span its lifetime across our own async work; we exit via `__exit__`
in `close()`.
"""

from __future__ import annotations

import asyncio
import os
import time

import structlog

from carl.config.schema import DeepgramConfig
from carl.stages.stt.base import OnTranscriptCallback, Transcript

log = structlog.get_logger()


class DeepgramSTT:
    def __init__(self, config: DeepgramConfig):
        # Lazy import — deepgram-sdk pulls its own transport layer.
        from deepgram import DeepgramClient

        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY not set in environment. "
                "Add it to .env — see .env.example."
            )
        self._client = DeepgramClient(api_key=api_key)
        self._config = config

    async def open_stream(self, on_transcript: OnTranscriptCallback) -> DeepgramSession:
        session = DeepgramSession(self._client, self._config, on_transcript)
        await session._open()
        return session


class DeepgramSession:
    def __init__(self, client, config: DeepgramConfig, on_transcript: OnTranscriptCallback):
        self._client = client
        self._config = config
        self._on_transcript = on_transcript
        self._cm = None                 # holds the connection context manager
        self._conn = None               # the connection object from __enter__
        self._loop = asyncio.get_event_loop()
        self._final_parts: list[str] = []
        self._opened_at: float | None = None

    async def _open(self) -> None:
        # Lazy import so this file is safe to import without deepgram-sdk present.
        from deepgram.core.events import EventType

        self._cm = self._client.listen.v2.connect(
            model=self._config.model,
            language=self._config.language,
            encoding="linear16",
            sample_rate=16000,
            interim_results=self._config.interim_results,
            vad_events=True,
            utterance_end_ms=self._config.utterance_end_ms,
            smart_format=self._config.smart_format,
            endpointing=self._config.endpointing_ms,
        )
        self._conn = self._cm.__enter__()

        # Handlers run on Deepgram's own thread — bridge to our asyncio loop
        # via loop.call_soon_threadsafe so on_transcript is awaited from the
        # right context.
        self._conn.on(EventType.MESSAGE, self._on_message)
        self._conn.on(EventType.ERROR, self._on_error)
        self._conn.on(EventType.CLOSE, self._on_close)
        self._conn.start_listening()
        self._opened_at = time.monotonic()
        log.info("deepgram_stream_opened", model=self._config.model)

    def _on_message(self, message) -> None:
        # Called on Deepgram's thread. Bridge to asyncio.
        try:
            msg_type = getattr(message, "type", "")
            if msg_type == "Transcript" or msg_type == "Results":
                # Nova-3 messages: message.channel.alternatives[0].transcript
                alt = message.channel.alternatives[0]
                text = alt.transcript
                is_final = bool(getattr(message, "is_final", False))
                confidence = float(getattr(alt, "confidence", 0.0))
                if is_final and text:
                    self._final_parts.append(text)
                self._loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self._on_transcript(Transcript(text=text, is_final=is_final,
                                                   confidence=confidence)),
                )
        except Exception:  # noqa: BLE001
            log.exception("deepgram_message_parse_failed")

    def _on_error(self, err) -> None:
        log.error("deepgram_error", error=str(err))

    def _on_close(self, *_args) -> None:
        log.info("deepgram_stream_closed")

    async def send(self, pcm_i16: bytes) -> None:
        if self._conn is None:
            return
        # Deepgram v7 conn.send is sync + non-blocking (queues to background thread).
        # Wrap in try/except so a mid-utterance transport hiccup doesn't crash us.
        try:
            self._conn.send(pcm_i16)
        except Exception:  # noqa: BLE001
            log.exception("deepgram_send_failed")

    async def close(self) -> str:
        # Exit the context manager to close cleanly (sends CloseStream, waits for final finalization).
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                log.exception("deepgram_close_failed")
            self._cm = None
            self._conn = None
        # Give the Deepgram thread a moment to deliver any final transcript
        # that arrived after CloseStream.
        await asyncio.sleep(0.05)
        return " ".join(self._final_parts).strip()
