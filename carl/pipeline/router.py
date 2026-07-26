"""Audio router. See docs/specs/pipeline.md.

Milestone 1: no real sounddevice. This is a stub that supports the
interface so real audio can drop in at Milestone 2 without rewriting
the pipeline orchestrator.

Design (post-Milestone-2):
- One continuous full-duplex audio stream (sounddevice) runs the mic.
- The router receives PCM frames on an audio-callback thread and
  dispatches them to registered consumers based on the currently-active
  set of consumers (which the pipeline updates per FSM state).
- Ring buffer holds ~500ms of pre-wake audio so the wake phrase +
  utterance start can be prepended to the STT stream.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable

FrameConsumer = Callable[[bytes], Awaitable[None] | None]


class AudioRouter:
    """Milestone 1 stub. Consumer registration/activation works; audio
    dispatch does not (there's no audio in mock mode). Ring buffer stays
    empty. push_output / abort_output are no-ops.

    Real implementation arrives in Milestone 2 (see plans/milestone-2-hello-carl.md).
    """

    FRAME_SIZE = 512   # samples at 16 kHz = 32ms

    def __init__(self, ring_buffer_frames: int = 16):
        self._ring: deque[bytes] = deque(maxlen=ring_buffer_frames)
        self._consumers: dict[str, FrameConsumer] = {}
        self._active: set[str] = set()
        self._output_bytes_written = 0

    # ─── Consumer management ──────────────────────────────────────

    def register(self, name: str, consumer: FrameConsumer) -> None:
        self._consumers[name] = consumer

    def activate(self, *names: str) -> None:
        for n in names:
            if n not in self._consumers:
                raise KeyError(f"router consumer '{n}' not registered")
        self._active = set(names)

    def deactivate(self, *names: str) -> None:
        for n in names:
            self._active.discard(n)

    def active(self) -> set[str]:
        return set(self._active)

    # ─── Ring buffer replay (used when opening STT after wake) ────

    async def replay_ring(self, consumer: FrameConsumer) -> None:
        for frame in list(self._ring):
            result = consumer(frame)
            if asyncio.iscoroutine(result):
                await result

    # ─── Output side ──────────────────────────────────────────────

    async def push_output(self, pcm: bytes) -> None:
        """Push PCM bytes to the speaker. Milestone 1: track byte count only."""
        self._output_bytes_written += len(pcm)

    async def abort_output(self) -> None:
        """Flush any pending output. Milestone 1: reset counter."""
        self._output_bytes_written = 0

    def output_bytes_written(self) -> int:
        return self._output_bytes_written

    def reset_output_counter(self) -> None:
        self._output_bytes_written = 0

    # ─── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """No-op in Milestone 1. Real impl starts sounddevice streams."""
        return

    async def stop(self) -> None:
        return
