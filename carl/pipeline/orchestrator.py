"""Pipeline orchestrator. Owner of FSM state transitions.

Milestone 1 note: with mock stages, there's no real audio flowing, so the
"RECORDING → THINKING" and "SPEAKING → RECORDING" transitions happen inline
inside the wake handler rather than being event-queue-driven. Milestone 2
(when real VAD, STT and TTS land) refactors this into fully-event-driven
transitions where each transition responds to an event on the queue. The
state-machine surface is the same either way; only the trigger source changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any

import structlog
import ulid

from carl.config import Config
from carl.events import EventLog
from carl.memory import MemoryStore
from carl.pipeline.fsm import (
    FOLLOWUP_EXPIRED,
    SHUTDOWN,
    SPEECH_END,
    TTS_COMPLETE,
    WAKE_DETECTED,
    PipelineEvent,
    State,
)
from carl.pipeline.router import AudioRouter
from carl.stages import Stages
from carl.stages.reasoner import Message, ReasonerTurn, ToolCall, ToolResult
from carl.stages.stt import Transcript
from carl.stages.vad import SpeechEnd
from carl.storage.db import connect
from carl.tools import ToolContext, ToolRegistry
from carl.tools.base import ToolResult as ProviderToolResult

log = structlog.get_logger()


@dataclass(slots=True)
class TurnRecord:
    id: str
    turn_index: int
    started_at: float
    user_transcript: str | None = None
    assistant_text: str | None = None
    ended_at: float | None = None
    latency_ms_to_first_audio: int | None = None
    interrupted: bool = False


class Pipeline:
    """Owns the FSM. Single writer to self.state. Stages request transitions
    by putting events on self._event_queue; the run loop dispatches them.

    For Milestone 1, only WAKE_DETECTED and SHUTDOWN events flow through
    the queue. Real audio-driven events (speech_start/end, tts_interrupted)
    plug into the same loop at Milestones 2 and 5.
    """

    def __init__(
        self,
        config: Config,
        stages: Stages,
        memory_store: MemoryStore,
        event_log: EventLog,
        tool_registry: ToolRegistry,
        router: AudioRouter,
    ):
        self.config = config
        self.stages = stages
        self.memory_store = memory_store
        self.event_log = event_log
        self.tool_registry = tool_registry
        self.router = router

        self.state = State.IDLE
        self._event_queue: asyncio.Queue[PipelineEvent] = asyncio.Queue()
        self._shutdown = asyncio.Event()

        self._conversation_id: str | None = None
        self._current_turn: TurnRecord | None = None
        self._turn_index_by_conv: dict[str, int] = {}
        self._history: list[Message] = []
        self._tts_cancel: asyncio.Event = asyncio.Event()

        # Set in run() based on whether the wake stage is audio-driven.
        # Determines whether _on_wake_detected pumps real audio through VAD/STT
        # or uses the mock's synchronous canned-transcript path.
        self._audio_mode: bool = False

    # ─── Lifecycle ───────────────────────────────────────────────

    async def run(self) -> None:
        self.event_log.start()
        self._log_session_start()

        # Only start the audio router (which opens the mic) if we have an
        # audio-driven stage that needs it. Fully-mock config skips this so
        # dev/test runs don't require mic permission.
        self._audio_mode = self.stages.wake_word.wants_audio()
        if self._audio_mode:
            self.router.register("wake_word", self.stages.wake_word.feed)
            self.router.activate("wake_word")
            self.router.start()

        # Start wake-word listener; it will put wake_detected events on the queue.
        await self.stages.wake_word.start(self._on_wake_from_stage)

        try:
            while not self._shutdown.is_set():
                event = await self._event_queue.get()
                await self._dispatch(event)
        finally:
            await self._cleanup()

    async def shutdown(self) -> None:
        """Idempotent. Callable from a signal handler."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        # Wake the run loop out of queue.get()
        self._event_queue.put_nowait(PipelineEvent(type=SHUTDOWN))

    async def _cleanup(self) -> None:
        await self.stages.wake_word.stop()
        await self.router.stop()
        # Drain memory-store fire-and-forget tasks before tearing down the loop.
        close = getattr(self.memory_store, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()
        self._log_session_end()
        await self.event_log.stop(drain=True)

    # ─── Event queue bridge ──────────────────────────────────────

    async def _on_wake_from_stage(self, payload: dict[str, Any]) -> None:
        """Callback the wake-word stage invokes. Bridges to the FSM queue."""
        await self._event_queue.put(PipelineEvent(type=WAKE_DETECTED, payload=payload))

    # ─── Dispatch ────────────────────────────────────────────────

    async def _dispatch(self, event: PipelineEvent) -> None:
        if event.type == SHUTDOWN:
            return
        try:
            if event.type == WAKE_DETECTED:
                await self._on_wake_detected(event)
            else:
                log.warning("unhandled_event", event_type=event.type, state=self.state.value)
        except Exception:  # noqa: BLE001
            log.exception("dispatch_failed", event_type=event.type, state=self.state.value)
            # Clean up half-completed turn state so we don't leak orphan DB rows
            # or corrupt turn_index bookkeeping. Best-effort — reasoner.abort()
            # cancels any in-flight streaming, _end_turn writes whatever we have.
            with contextlib.suppress(Exception):
                await self.stages.reasoner.abort()
            with contextlib.suppress(Exception):
                await self._end_turn()
            self._transition(State.IDLE, trigger="error_recovery")

    # ─── Handlers ────────────────────────────────────────────────

    async def _on_wake_detected(self, event: PipelineEvent) -> None:
        if self.state != State.IDLE:
            log.info("wake_ignored_not_idle", state=self.state.value)
            return

        # Start conversation if none active; always start a new turn.
        if self._conversation_id is None:
            await self._start_conversation()
        await self._start_turn()

        self._transition(State.RECORDING, trigger=WAKE_DETECTED)
        self.event_log.log(
            "wake_word", "wake_detected", event.payload,
            conversation_id=self._conversation_id, turn_id=self._current_turn.id,
        )

        # Open STT session. Mock returns canned transcript on close();
        # real STT streams audio via the router until VAD fires speech_end.
        stt_session = await self.stages.stt.open_stream(self._on_transcript)

        if self._audio_mode:
            transcript_text = await self._record_utterance_real(stt_session)
        else:
            # Mock path: send a dummy frame to trigger the fake partial,
            # then close to get the canned final transcript.
            await stt_session.send(b"")
            transcript_text = await stt_session.close()

        self._transition(State.THINKING, trigger=SPEECH_END)
        assert self._current_turn is not None
        self._current_turn.user_transcript = transcript_text
        self.event_log.log(
            "stt", "stt_final",
            {"text": transcript_text, "confidence": 1.0, "latency_ms": 0},
            conversation_id=self._conversation_id, turn_id=self._current_turn.id,
        )

        # Retrieve memories (empty in Milestone 1 — no memories saved yet).
        memories = await self.memory_store.retrieve(query=transcript_text)
        self.event_log.log(
            "memory", "memory_retrieve",
            {"query": transcript_text, "results_count": len(memories),
             "result_ids": [m.id for m in memories]},
            conversation_id=self._conversation_id, turn_id=self._current_turn.id,
        )

        # Run reasoner + stream chunks to TTS.
        request = ReasonerTurn(
            user_transcript=transcript_text,
            conversation_id=self._conversation_id,
            turn_id=self._current_turn.id,
            history=list(self._history),
            memories=list(memories),
        )
        assistant_text_parts: list[str] = []
        turn_start = time.time()
        first_chunk = True
        # NOTE: _tts_cancel producer arrives in Milestone 5 (barge-in). Until
        # then, the Event is threaded but never set — TTS speak() honors it
        # for interface symmetry with the real barge-in flow.
        self._tts_cancel.clear()

        self.event_log.log(
            "reasoner", "llm_request",
            {"model": "mock", "history_turns": len(self._history)},
            conversation_id=self._conversation_id, turn_id=self._current_turn.id,
        )

        async for chunk in self.stages.reasoner.turn(
            request,
            on_tool_call=self._make_on_tool_call(),
            on_filler_needed=self._on_filler_needed,
        ):
            if first_chunk:
                self._transition(State.SPEAKING, trigger="llm_final_text")
                first_chunk = False
                latency_ms = int((time.time() - turn_start) * 1000)
                self.event_log.log(
                    "tts", "tts_first_byte",
                    {"latency_ms": latency_ms},
                    conversation_id=self._conversation_id, turn_id=self._current_turn.id,
                )
                self._current_turn.latency_ms_to_first_audio = latency_ms
            assistant_text_parts.append(chunk.text)
            await self.stages.tts.speak(chunk.text, self.router.push_output, self._tts_cancel)

        # Concatenate chunks verbatim — streaming reasoners emit sub-word
        # deltas ("Hel", "lo ", "world."); inserting whitespace or stripping
        # per-chunk would corrupt the text handed to TTS and stored in history.
        assistant_text = "".join(assistant_text_parts)
        self._current_turn.assistant_text = assistant_text

        self.event_log.log(
            "reasoner", "llm_response",
            {"stop_reason": "end_turn", "text_preview": assistant_text[:80]},
            conversation_id=self._conversation_id, turn_id=self._current_turn.id,
        )

        self.event_log.log(
            "tts", "tts_complete",
            {"chars": len(assistant_text)},
            conversation_id=self._conversation_id, turn_id=self._current_turn.id,
        )

        # Record turn history for future context.
        self._history.append(Message(role="user", content=transcript_text))
        self._history.append(Message(role="assistant", content=assistant_text))
        history_cap = self.config.conversation.history_turns * 2
        if len(self._history) > history_cap:
            self._history = self._history[-history_cap:]

        # Persist the turn record now that it's complete.
        await self._end_turn()

        # Transition SPEAKING → RECORDING (follow-up window) → IDLE
        self._transition(State.RECORDING, trigger=TTS_COMPLETE)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._shutdown.wait(),
                timeout=self.config.conversation.follow_up_ms / 1000,
            )
        self._transition(State.IDLE, trigger=FOLLOWUP_EXPIRED)

    async def _on_transcript(self, transcript: Transcript) -> None:
        """Called by STT session on every partial/final. Milestone 1: log only."""
        if not transcript.is_final:
            self.event_log.log(
                "stt", "stt_partial", {"text": transcript.text},
                conversation_id=self._conversation_id,
                turn_id=self._current_turn.id if self._current_turn else None,
            )

    async def _record_utterance_real(self, stt_session) -> str:
        """Real-audio RECORDING path: pump router audio into STT and VAD in
        parallel until VAD fires speech_end (or utterance timeout). Returns
        the final transcript."""
        # Reset VAD state between utterances — Silero is stateful.
        reset = getattr(self.stages.vad, "reset", None)
        if reset is not None:
            reset()

        speech_end = asyncio.Event()

        # Frame queues so audio flows in ORDER through STT and VAD. The
        # router's consumer callback is sync (runs on the dispatch task);
        # putting on a Queue and draining via a dedicated task preserves
        # ordering that a bare `create_task` per frame would risk.
        stt_frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        vad_frames: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def _stt_drain() -> None:
            while True:
                frame = await stt_frames.get()
                if frame is None:
                    return
                await stt_session.send(frame)

        async def _vad_drain() -> None:
            while True:
                frame = await vad_frames.get()
                if frame is None:
                    return
                evt = await self.stages.vad.process(frame)
                if isinstance(evt, SpeechEnd):
                    speech_end.set()

        stt_task = asyncio.create_task(_stt_drain(), name="stt_drain")
        vad_task = asyncio.create_task(_vad_drain(), name="vad_drain")

        def stt_consumer(pcm: bytes) -> None:
            stt_frames.put_nowait(pcm)

        def vad_consumer(pcm: bytes) -> None:
            vad_frames.put_nowait(pcm)

        self.router.register("stt", stt_consumer)
        self.router.register("vad", vad_consumer)

        # Prepend the router's ring buffer to STT so the wake phrase +
        # start-of-utterance are captured (real STT streams have startup
        # latency; the ring buffer holds audio from before the stream existed).
        await self.router.replay_ring(stt_consumer)
        # VAD sees only live audio (replaying the wake phrase into VAD would
        # immediately fire speech_end at t=0).
        self.router.activate("stt", "vad")

        try:
            await asyncio.wait_for(
                speech_end.wait(),
                timeout=self.config.conversation.utterance_max_seconds,
            )
        except TimeoutError:
            log.warning("utterance_timeout",
                        seconds=self.config.conversation.utterance_max_seconds)

        # Stop routing audio and drain the queues cleanly.
        self.router.deactivate("stt", "vad")
        stt_frames.put_nowait(None)
        vad_frames.put_nowait(None)
        await stt_task
        await vad_task

        return await stt_session.close()

    def _make_on_tool_call(self):
        """Build the reasoner's on_tool_call callback bound to this turn's context.
        The callback bridges Reasoner.ToolCall/ToolResult (stage layer) to the
        ToolRegistry (tools layer)."""
        async def on_tool_call(call: ToolCall) -> ToolResult:
            start = time.time()
            self.event_log.log(
                "reasoner", "llm_tool_use",
                {"tool_name": call.name, "tool_input": call.input, "tool_use_id": call.id},
                conversation_id=self._conversation_id,
                turn_id=self._current_turn.id if self._current_turn else None,
            )
            provider_result: ProviderToolResult = await self.tool_registry.execute(
                call.name, call.input,
                ToolContext(
                    conversation_id=self._conversation_id,
                    turn_id=self._current_turn.id if self._current_turn else None,
                ),
            )
            self.event_log.log(
                "reasoner", "llm_tool_result",
                {
                    "tool_use_id": call.id,
                    "is_error": provider_result.is_error,
                    "latency_ms": int((time.time() - start) * 1000),
                    "content_preview": provider_result.content[:80],
                },
                conversation_id=self._conversation_id,
                turn_id=self._current_turn.id if self._current_turn else None,
            )
            return ToolResult(
                tool_use_id=call.id,
                content=provider_result.content,
                is_error=provider_result.is_error,
            )
        return on_tool_call

    async def _on_filler_needed(self, phrase: str) -> None:
        """Called by the reasoner when it emits a tool_use block; speaks a short
        filler phrase while the tool runs so there's no dead air."""
        await self.stages.tts.speak(phrase, self.router.push_output, self._tts_cancel)

    # ─── State + turn bookkeeping ────────────────────────────────

    def _transition(self, new_state: State, *, trigger: str) -> None:
        old = self.state
        self.state = new_state
        log.debug("state_transition", from_state=old.value, to_state=new_state.value,
                  trigger=trigger)
        self.event_log.log(
            "fsm", "state_transition",
            {"from": old.value, "to": new_state.value, "trigger": trigger},
            conversation_id=self._conversation_id,
            turn_id=self._current_turn.id if self._current_turn else None,
        )
        # Human-visible line for terminal watchers
        print(f"[FSM] {old.value} → {new_state.value}", flush=True)

    async def _start_conversation(self) -> None:
        conv_id = str(ulid.new())
        self._conversation_id = conv_id
        self._turn_index_by_conv[conv_id] = 0
        now = time.time()
        await self._db_execute(
            "INSERT INTO conversations (id, user_id, started_at, turn_count) VALUES (?, ?, ?, 0)",
            (conv_id, "default", now),
        )
        self.event_log.log("fsm", "conversation_start", {"conversation_id": conv_id},
                            conversation_id=conv_id)

    async def _start_turn(self) -> None:
        assert self._conversation_id is not None
        turn_id = str(ulid.new())
        turn_index = self._turn_index_by_conv[self._conversation_id]
        self._turn_index_by_conv[self._conversation_id] = turn_index + 1
        now = time.time()
        self._current_turn = TurnRecord(id=turn_id, turn_index=turn_index, started_at=now)
        await self._db_execute(
            "INSERT INTO turns (id, conversation_id, turn_index, started_at, interrupted) "
            "VALUES (?, ?, ?, ?, 0)",
            (turn_id, self._conversation_id, turn_index, now),
        )

    async def _end_turn(self) -> None:
        """Persist the current turn's final fields and clear self._current_turn.
        Safe to call on the error path — records whatever data we have."""
        if self._current_turn is None:
            return
        self._current_turn.ended_at = time.time()
        await self._db_execute(
            "UPDATE turns SET user_transcript = ?, assistant_text = ?, ended_at = ?, "
            "latency_ms_to_first_audio = ?, interrupted = ? WHERE id = ?",
            (
                self._current_turn.user_transcript,
                self._current_turn.assistant_text,
                self._current_turn.ended_at,
                self._current_turn.latency_ms_to_first_audio,
                1 if self._current_turn.interrupted else 0,
                self._current_turn.id,
            ),
        )
        await self._db_execute(
            "UPDATE conversations SET turn_count = turn_count + 1 WHERE id = ?",
            (self._conversation_id,),
        )
        self._current_turn = None

    def _log_session_start(self) -> None:
        self.event_log.log("system", "session_start", {
            "config_version": self.config.version,
            "providers": self.config.providers.model_dump(),
        })

    def _log_session_end(self) -> None:
        self.event_log.log("system", "session_end", {})

    async def _db_execute(self, sql: str, params: tuple) -> None:
        """Sync SQLite write dispatched to a worker thread — never blocks the loop."""
        def _do():
            conn = connect(self.config.storage.resolved_path())
            try:
                conn.execute("BEGIN")
                conn.execute(sql, params)
                conn.execute("COMMIT")
            finally:
                conn.close()
        await asyncio.to_thread(_do)
