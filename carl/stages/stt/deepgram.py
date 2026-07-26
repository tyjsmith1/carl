"""Deepgram Nova-3 streaming STT stage. See docs/specs/pipeline.md.

Uses the deepgram-sdk v7+ API surface: `client.listen.v1.connect(...)` as a
context manager.

Naming trap (verified empirically 2026-07-26): in SDK v7, `v1` is the general
STT streaming surface — Nova-3, all the endpointing / interim / smart-format
knobs — while `v2` is a *different* WebSocket API for the Flux voice-agent
models only (`flux-general-en`, `flux-general-multi`). Use `v1` for Nova-3.
Older v3/v4 patterns (`listen.asynclive.v("1")`, `LiveOptions`,
`LiveTranscriptionEvents`) are dead — do not port them.

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
import contextlib
import os
import threading
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
        self._bytes_sent = 0
        self._send_count = 0
        self._keepalive_task: asyncio.Task | None = None
        self._listen_thread: threading.Thread | None = None

    async def _open(self) -> None:
        # Lazy import so this file is safe to import without deepgram-sdk present.
        from deepgram.core.events import EventType

        self._cm = self._client.listen.v1.connect(
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
        # start_listening() is a BLOCKING receive loop — it runs the
        # WebSocket read pump until close. Must be on its own thread; if
        # called from the asyncio loop it blocks the whole app for the
        # duration of the session (which then races Deepgram's 10s
        # no-audio timeout and closes with 1011).
        self._listen_thread = threading.Thread(
            target=self._conn.start_listening,
            name="deepgram_listen",
            daemon=True,
        )
        self._listen_thread.start()
        self._opened_at = time.monotonic()
        log.info("deepgram_stream_opened", model=self._config.model)
        # Keep-alive every 5 s so we don't hit Deepgram's 10 s
        # net0001 no-audio timeout during any quiet period.
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(),
                                                    name="deepgram_keepalive")

    async def _keepalive_loop(self) -> None:
        try:
            while self._conn is not None:
                await asyncio.sleep(5.0)
                if self._conn is None:
                    return
                try:
                    self._conn.send_keep_alive()
                except Exception:  # noqa: BLE001
                    log.exception("deepgram_keepalive_failed")
                    return
        except asyncio.CancelledError:
            return

    def _on_message(self, message) -> None:
        # Called on Deepgram's thread. Bridge to asyncio.
        # In SDK v7, listen.v1 messages are typed pydantic objects. The one
        # we care about is ListenV1Results (type="Results"); UtteranceEnd,
        # SpeechStarted, and Metadata also arrive here but we ignore them
        # for now (VAD is running client-side).
        try:
            if getattr(message, "type", "") != "Results":
                return
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
        # V1SocketClient has `send_media(bytes)` for audio, not `.send()`.
        # It's sync + non-blocking (queues to a background writer thread).
        try:
            self._conn.send_media(pcm_i16)
            self._bytes_sent += len(pcm_i16)
            self._send_count += 1
            # First-send timing tells us how long audio took to start flowing.
            if self._send_count == 1 and self._opened_at is not None:
                log.info("deepgram_first_send",
                         since_open_ms=int((time.monotonic() - self._opened_at) * 1000),
                         nbytes=len(pcm_i16))
        except Exception:  # noqa: BLE001
            log.exception("deepgram_send_failed",
                          send_count=self._send_count,
                          bytes_sent=self._bytes_sent,
                          since_open_ms=(
                              int((time.monotonic() - self._opened_at) * 1000)
                              if self._opened_at else -1
                          ))

    async def close(self) -> str:
        # Stop keep-alive first so it doesn't race with close_stream.
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
        self._keepalive_task = None
        # Send CloseStream so Deepgram flushes any pending finals before
        # tearing down the WebSocket, then exit the context manager to
        # actually close the connection.
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.send_close_stream()
        # Give Deepgram a moment to deliver the last final transcript
        # after CloseStream but before the socket goes away.
        await asyncio.sleep(0.15)
        if self._cm is not None:
            with contextlib.suppress(Exception):
                self._cm.__exit__(None, None, None)
            self._cm = None
            self._conn = None
        log.info("deepgram_session_closed",
                 bytes_sent=self._bytes_sent,
                 send_count=self._send_count,
                 final_parts=len(self._final_parts))
        return " ".join(self._final_parts).strip()
