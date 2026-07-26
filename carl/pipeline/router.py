"""Audio router. See docs/specs/pipeline.md.

Real full-duplex audio router built on sounddevice.

Threading contract (the most bug-prone part of this file):
- sounddevice runs two native audio threads: input callback + output callback.
  Both fire on threads that are NOT the asyncio event loop.
- Input frames from the audio thread cross into the asyncio world via
  `janus.Queue.sync_q.put_nowait()`. An asyncio task (`_dispatch_loop`) drains
  the async side and delivers frames to registered consumers.
- Output frames go the other way: async code calls `push_output(bytes)` which
  puts on a queue; the output callback thread drains and fills its `outdata`
  buffer, padding with silence on underrun.
- Consumer `feed()` callbacks run on the asyncio loop (in the dispatch task),
  not on the audio thread. Consumers must be SYNCHRONOUS and fast (< 1ms).
  Slow consumers (STT socket writes, LLM calls) must return quickly and defer
  their real work to their own tasks.

Frame contract:
- Input: 16 kHz mono int16, delivered as 1024-byte blobs (512 samples).
- Output: 24 kHz mono int16, arbitrary chunk sizes.

Ring buffer: last ~500 ms of input frames, replayable via `replay_ring()`.
Used at wake-detected time to prepend the wake phrase + start of utterance
into the STT stream (real STT connections have startup latency, so we need
audio from *before* the connection existed).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable

import janus
import sounddevice as sd
import structlog

log = structlog.get_logger()

FrameConsumer = Callable[[bytes], None]   # sync — dispatch task calls this


class AudioRouter:
    IN_SAMPLE_RATE = 16000
    OUT_SAMPLE_RATE = 24000
    FRAME_SIZE = 512                       # samples per input frame = 32 ms @ 16 kHz
    # OpenAI TTS streams chunks in bursts over HTTPS; the audio callback pulls
    # continuously at OUT_SAMPLE_RATE. If the audio callback is too greedy
    # (small block size), the queue underruns between HTTP chunks and we hear
    # gaps. 4096 samples = ~170 ms per callback gives the network enough
    # headroom to refill between callbacks.
    OUT_BLOCK_SIZE = 4096                  # samples per output callback (~170 ms at 24 kHz)
    IN_QUEUE_MAX = 100                     # ~3.2 s of buffered audio before drops
    OUT_QUEUE_MAX = 200                    # bytes-buffer count, generous
    RING_BUFFER_FRAMES = 16                # ~512 ms at 32 ms/frame

    def __init__(
        self,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
    ):
        self._input_device = input_device if input_device != "default" else None
        self._output_device = output_device if output_device != "default" else None

        # Consumer registration
        self._consumers: dict[str, FrameConsumer] = {}
        self._active: set[str] = set()

        # Ring buffer of recent input frames (deque append is atomic under CPython GIL)
        self._ring: deque[bytes] = deque(maxlen=self.RING_BUFFER_FRAMES)

        # Cross-thread queues
        self._in_queue: janus.Queue[bytes] | None = None
        self._out_queue: janus.Queue[bytes] | None = None

        # Streams and dispatch task, created in start()
        self._in_stream: sd.RawInputStream | None = None
        self._out_stream: sd.RawOutputStream | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Diagnostic counters
        self._output_bytes_written = 0
        self._input_frames_dropped = 0
        self._output_underruns = 0

    # ─── Consumer management ──────────────────────────────────────

    def register(self, name: str, consumer: FrameConsumer) -> None:
        self._consumers[name] = consumer

    def activate(self, *names: str) -> None:
        """Additive: adds `names` to the active set. Fix for the code-review
        finding that made this a full replacement (bug for Milestone 5)."""
        for n in names:
            if n not in self._consumers:
                raise KeyError(f"router consumer '{n}' not registered")
            self._active.add(n)

    def deactivate(self, *names: str) -> None:
        for n in names:
            self._active.discard(n)

    def set_active(self, *names: str) -> None:
        """Explicit full replacement, when that's what you want."""
        for n in names:
            if n not in self._consumers:
                raise KeyError(f"router consumer '{n}' not registered")
        self._active = set(names)

    def active(self) -> set[str]:
        return set(self._active)

    # ─── Ring buffer replay (used when opening STT after wake) ────

    async def replay_ring(self, consumer: FrameConsumer) -> None:
        for frame in list(self._ring):
            consumer(frame)

    async def drain_ring(self) -> None:
        """Discard buffered input frames. Used by the follow-up window before
        activating VAD so the tail of Carl's own TTS (captured via the mic
        during the guard delay) isn't the first thing VAD sees and mistakes
        for user speech."""
        self._ring.clear()

    # ─── Output side ──────────────────────────────────────────────

    async def push_output(self, pcm: bytes) -> None:
        """Push PCM bytes to the speaker. Awaits if the queue is full — this
        backpressures fast producers (OpenAI TTS downloads at network speed,
        the audio callback drains at realtime playback speed). Without this
        wait, a multi-sentence reply piles multiple speeches into the queue
        and drops audio, producing choppy playback + apparent overlap."""
        if self._out_queue is None:
            return    # not started; drop silently (mock mode passes through)
        await self._out_queue.async_q.put(pcm)
        self._output_bytes_written += len(pcm)

    async def abort_output(self) -> None:
        """Flush pending output: drain the queue AND flush the OS-level audio buffer.
        Both are required — closing the HTTP stream alone leaves ~100-200ms of
        audio already-committed to the device."""
        if self._out_queue is None or self._out_stream is None:
            return
        # Drain the async queue
        while True:
            try:
                self._out_queue.async_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Abort OS buffer + restart the stream (abort leaves it stopped)
        with contextlib.suppress(sd.PortAudioError):
            self._out_stream.abort()
        with contextlib.suppress(sd.PortAudioError):
            self._out_stream.start()
        self._output_bytes_written = 0

    def output_bytes_written(self) -> int:
        return self._output_bytes_written

    def reset_output_counter(self) -> None:
        self._output_bytes_written = 0

    # ─── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Open sounddevice streams + spawn the dispatch task.
        Called from an asyncio context so we can bind janus queues to this loop."""
        self._loop = asyncio.get_running_loop()
        self._in_queue = janus.Queue(maxsize=self.IN_QUEUE_MAX)
        self._out_queue = janus.Queue(maxsize=self.OUT_QUEUE_MAX)

        self._in_stream = sd.RawInputStream(
            samplerate=self.IN_SAMPLE_RATE,
            blocksize=self.FRAME_SIZE,
            channels=1,
            dtype="int16",
            device=self._input_device,
            callback=self._on_input,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=self.OUT_SAMPLE_RATE,
            blocksize=self.OUT_BLOCK_SIZE,
            channels=1,
            dtype="int16",
            device=self._output_device,
            callback=self._on_output,
        )
        self._in_stream.start()
        self._out_stream.start()

        self._dispatch_task = asyncio.create_task(self._dispatch_loop(), name="router_dispatch")
        log.info("audio_router_started",
                 input_device=self._input_device or "default",
                 output_device=self._output_device or "default",
                 in_sr=self.IN_SAMPLE_RATE, out_sr=self.OUT_SAMPLE_RATE)

    async def stop(self) -> None:
        if self._dispatch_task is not None and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatch_task
        self._dispatch_task = None

        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                with contextlib.suppress(sd.PortAudioError):
                    stream.stop()
                with contextlib.suppress(sd.PortAudioError):
                    stream.close()
        self._in_stream = None
        self._out_stream = None

        for q in (self._in_queue, self._out_queue):
            if q is not None:
                q.close()
                await q.wait_closed()
        self._in_queue = None
        self._out_queue = None

        log.info("audio_router_stopped",
                 dropped_frames=self._input_frames_dropped,
                 output_underruns=self._output_underruns)

    # ─── Audio-thread callbacks ───────────────────────────────────

    def _on_input(self, indata, frames, time_info, status) -> None:
        """Fires on sounddevice's input thread. Must be non-blocking."""
        if status:
            log.warning("input_status", flags=str(status))
        # indata is a memoryview of int16 bytes (dtype='int16')
        frame_bytes = bytes(indata)
        # Append to ring buffer (deque.append is atomic under GIL)
        self._ring.append(frame_bytes)
        # Push to async side; drop if full (input can't be paused)
        if self._in_queue is not None:
            try:
                self._in_queue.sync_q.put_nowait(frame_bytes)
            except janus.SyncQueueFull:
                self._input_frames_dropped += 1

    def _on_output(self, outdata, frames, time_info, status) -> None:
        """Fires on sounddevice's output thread. Must fill outdata buffer."""
        if status:
            log.warning("output_status", flags=str(status))
        needed_bytes = frames * 2   # 2 bytes per int16 sample
        filled = 0
        # Drain queue until outdata is full
        while filled < needed_bytes and self._out_queue is not None:
            try:
                chunk = self._out_queue.sync_q.get_nowait()
            except janus.SyncQueueEmpty:
                break
            take = min(len(chunk), needed_bytes - filled)
            outdata[filled : filled + take] = chunk[:take]
            filled += take
            # Any leftover from this chunk gets pushed back (rare with our sizes)
            if take < len(chunk):
                leftover = chunk[take:]
                # push leftover back on the tail; janus doesn't support head-push.
                # Fine — audio ordering doesn't cross frame boundaries here.
                # If the queue is full, we drop the leftover to keep playing.
                with contextlib.suppress(janus.SyncQueueFull):
                    self._out_queue.sync_q.put_nowait(leftover)
        if filled < needed_bytes:
            # Underrun — pad with silence
            outdata[filled:needed_bytes] = b"\x00" * (needed_bytes - filled)
            if filled > 0:
                self._output_underruns += 1

    # ─── Dispatch coroutine (asyncio loop side) ───────────────────

    async def _dispatch_loop(self) -> None:
        """Drains the input queue and dispatches to active consumers."""
        assert self._in_queue is not None
        while True:
            frame = await self._in_queue.async_q.get()
            # Snapshot active set — mutation during dispatch is fine
            for name in list(self._active):
                consumer = self._consumers.get(name)
                if consumer is None:
                    continue
                try:
                    consumer(frame)
                except Exception:  # noqa: BLE001
                    log.exception("consumer_raised", consumer=name)
