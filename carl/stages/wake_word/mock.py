"""Mock stdin-triggered wake word. Hitting Enter fires a wake event.
Typing 'exit' raises SIGINT so main's signal handler triggers shutdown.

Design note: stdin.readline() cannot be interrupted from another thread —
so we read from a daemon thread that will die when the interpreter exits,
and we bridge lines back to the asyncio loop via an asyncio.Queue.
Cancelling the reader coroutine no longer hangs shutdown on a blocked
executor thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import threading
from typing import Any

import structlog

from carl.stages.wake_word.base import OnWakeCallback

log = structlog.get_logger()


class StdinTriggeredWakeWord:
    """Reads stdin lines. Any Enter → fires on_wake. 'exit' → SIGINT for graceful shutdown."""

    def __init__(self, config: Any = None):
        self._on_wake: OnWakeCallback | None = None
        self._task: asyncio.Task | None = None
        self._reader_thread: threading.Thread | None = None
        self._line_queue: asyncio.Queue[str | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown = False

    async def start(self, on_wake: OnWakeCallback) -> None:
        if self._task is not None:
            return
        self._on_wake = on_wake
        self._loop = asyncio.get_running_loop()
        self._line_queue = asyncio.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_stdin_forever,
            name="mock_wake_word_stdin_reader",
            daemon=True,
        )
        self._reader_thread.start()
        self._task = asyncio.create_task(self._loop_coro(), name="mock_wake_word_stdin")

    async def stop(self) -> None:
        self._shutdown = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        # The daemon reader thread stays alive if it's blocked on readline —
        # that's fine, it dies with the interpreter and won't hang shutdown.

    def wants_audio(self) -> bool:
        return False

    def feed(self, pcm_i16: bytes) -> None:
        return  # ignored — trigger-driven

    def _read_stdin_forever(self) -> None:
        """Runs on a daemon thread. Pushes lines (or None on EOF) to the queue
        via call_soon_threadsafe so the asyncio side can await them."""
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    self._push_line(None)
                    return
                self._push_line(line)
        except Exception:  # noqa: BLE001
            self._push_line(None)

    def _push_line(self, line: str | None) -> None:
        loop, queue = self._loop, self._line_queue
        if loop is None or queue is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(queue.put_nowait, line)

    async def _loop_coro(self) -> None:
        assert self._line_queue is not None
        print("IDLE. press enter to trigger wake word, type 'exit' to quit.", flush=True)
        while not self._shutdown:
            line = await self._line_queue.get()
            if line is None:
                return
            text = line.strip()
            if text == "exit":
                log.info("mock_wake_word_exit_requested")
                os.kill(os.getpid(), signal.SIGINT)
                return
            if self._on_wake is not None:
                await self._on_wake({"keyword": "hey_carl_mock", "score": 1.0})
