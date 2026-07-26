"""Structured event log. Append-only. Non-blocking writes from stages.

Design (see docs/specs/event-log.md):
- Stages call `event_log.log(...)` — a *sync* call that puts an event on an
  internal asyncio.Queue. Never blocks on I/O.
- A single background writer task drains the queue in batches, executing
  batched INSERTs against SQLite in a dedicated thread (SQLite writes are
  sync). This keeps the audio-path event loop free.
- If the queue overflows (should never happen at realistic rates), events
  drop with a stderr warning — preferring drop to blocking a stage.
- Payloads are JSON-serialized. Secret patterns are stripped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from carl.storage.db import connect

log = structlog.get_logger()

_MAX_QUEUE_SIZE = 10_000
_BATCH_SIZE = 64

# Very rough secret-shape patterns. Not compliance-grade, just defense-in-depth
# to keep casually-obvious API keys out of the payload log.
# NOTE: patterns are anchored to known vendor prefixes to avoid false positives
# on hex hashes, ULIDs, JWT segments, long user transcripts, and other benign
# long alphanumeric strings that would otherwise get redacted and destroy the
# debug value of the event log.
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),     # Anthropic (matched first — more specific)
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),         # OpenAI
    re.compile(r"tvly-[A-Za-z0-9]{20,}"),          # Tavily
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.=+/]{20,}"),  # Authorization headers
]


def _sanitize(value: Any) -> Any:
    """Recursively strip likely secret patterns from a payload."""
    if isinstance(value, str):
        for pat in _SECRET_PATTERNS:
            value = pat.sub("[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class Event:
    stage: str
    event_type: str
    payload: dict[str, Any]
    conversation_id: str | None = None
    turn_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> tuple:
        return (
            self.conversation_id,
            self.turn_id,
            self.stage,
            self.event_type,
            json.dumps(_sanitize(self.payload), default=str),
            self.created_at,
        )


class EventLog:
    """Async, fire-and-forget event log."""

    def __init__(self, db_path: Path, stdout_mirror: bool = False):
        self._db_path = db_path
        self._stdout_mirror = stdout_mirror
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._writer_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    def start(self) -> None:
        """Kick off the background writer task. Call once, from an asyncio context."""
        if self._writer_task is not None:
            return
        self._writer_task = asyncio.create_task(self._writer_loop(), name="event_log_writer")

    async def stop(self, drain: bool = True) -> None:
        """Signal shutdown; optionally drain remaining events before returning."""
        self._shutdown.set()
        if self._writer_task is not None:
            if drain:
                await self._writer_task
            else:
                self._writer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._writer_task
            self._writer_task = None

    def log(
        self,
        stage: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Sync, non-blocking. Puts the event on the queue or drops with warning."""
        event = Event(
            stage=stage,
            event_type=event_type,
            payload=payload or {},
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        if self._stdout_mirror:
            print(f"[{stage}] {event_type} {json.dumps(_sanitize(event.payload), default=str)}",
                  file=sys.stderr)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("event_log_queue_full_dropping", event_type=event_type)

    async def _writer_loop(self) -> None:
        while not (self._shutdown.is_set() and self._queue.empty()):
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except TimeoutError:
                continue

            batch: list[Event] = [first]
            for _ in range(_BATCH_SIZE - 1):
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                await asyncio.to_thread(self._flush, batch)
            except Exception:
                log.exception("event_log_flush_failed", batch_size=len(batch))

    def _flush(self, batch: list[Event]) -> None:
        """Sync write executed in a worker thread."""
        conn = connect(self._db_path)
        try:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO events "
                "(conversation_id, turn_id, stage, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [e.to_row() for e in batch],
            )
            conn.execute("COMMIT")
        finally:
            conn.close()
