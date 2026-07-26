"""End-to-end mock pipeline test. This is the Milestone 1 acceptance test in
code form: a wake event triggers the whole pipeline, we assert the state
transitions happened and the event log recorded them, then shutdown cleanly."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from carl.config import Config
from carl.events import EventLog
from carl.memory import SQLiteMemoryStore
from carl.pipeline import AudioRouter, Pipeline
from carl.pipeline.fsm import WAKE_DETECTED, PipelineEvent, State
from carl.stages import build_stages
from carl.storage.db import connect
from carl.tools import NativeToolProvider, ToolRegistry, clear_registry


@pytest.fixture(autouse=True)
def _reset_tool_registry():
    clear_registry()
    yield
    clear_registry()


def _build_pipeline(config: Config, db_path: Path) -> Pipeline:
    memory_store = SQLiteMemoryStore(db_path)
    stages = build_stages(config)
    tool_registry = ToolRegistry(providers=[NativeToolProvider()])
    event_log = EventLog(db_path)
    router = AudioRouter()
    return Pipeline(
        config=config,
        stages=stages,
        memory_store=memory_store,
        event_log=event_log,
        tool_registry=tool_registry,
        router=router,
    )


@pytest.mark.asyncio
async def test_mock_pipeline_full_turn(default_config: Config, tmp_db: Path) -> None:
    default_config.conversation.follow_up_ms = 100  # short for fast tests
    default_config.conversation.followup_guard_ms = 0  # skip TTS-echo guard in tests

    pipeline = _build_pipeline(default_config, tmp_db)

    # Skip real wake-word start (which would block on stdin); instead put
    # a wake_detected event on the queue directly.
    pipeline.event_log.start()
    await pipeline.tool_registry.initialize()
    pipeline._log_session_start()  # match run()'s side effect for realism

    await pipeline._event_queue.put(
        PipelineEvent(type=WAKE_DETECTED, payload={"keyword": "hey_carl_mock"})
    )

    # Run the dispatch loop for one event, then shutdown.
    async def _run_one():
        event = await pipeline._event_queue.get()
        await pipeline._dispatch(event)

    await asyncio.wait_for(_run_one(), timeout=5.0)

    # Post-conditions
    assert pipeline.state == State.IDLE, "should return to IDLE after follow-up window"
    assert pipeline._conversation_id is not None
    # _current_turn is cleared once _end_turn persists the row — verify via DB.
    assert pipeline._current_turn is None
    assert pipeline._history, "history should contain the user+assistant turn"
    assert any("You said:" in msg.content for msg in pipeline._history)

    await pipeline.memory_store.close()   # drain fire-and-forget _touch tasks
    await pipeline.event_log.stop(drain=True)

    # Event log should contain the full trace.
    conn = connect(tmp_db)
    rows = conn.execute(
        "SELECT stage, event_type FROM events ORDER BY id"
    ).fetchall()
    conn.close()
    event_types = [(r["stage"], r["event_type"]) for r in rows]

    # Not asserting exact order (state_transition + fsm are interleaved),
    # but the load-bearing ones must all appear.
    assert ("system", "session_start") in event_types
    assert ("fsm", "conversation_start") in event_types
    assert ("wake_word", "wake_detected") in event_types
    assert ("stt", "stt_final") in event_types
    assert ("memory", "memory_retrieve") in event_types
    assert ("reasoner", "llm_request") in event_types
    assert ("reasoner", "llm_response") in event_types
    assert ("tts", "tts_complete") in event_types

    # State transitions: IDLE→RECORDING, RECORDING→THINKING,
    # THINKING→SPEAKING, SPEAKING→RECORDING (follow-up), RECORDING→IDLE
    conn = connect(tmp_db)
    transitions = [
        json.loads(r["payload_json"])
        for r in conn.execute(
            "SELECT payload_json FROM events "
            "WHERE stage='fsm' AND event_type='state_transition' ORDER BY id"
        ).fetchall()
    ]
    conn.close()

    seen = [(t["from"], t["to"]) for t in transitions]
    assert ("IDLE", "RECORDING") in seen
    assert ("RECORDING", "THINKING") in seen
    assert ("THINKING", "SPEAKING") in seen
    assert ("SPEAKING", "RECORDING") in seen
    assert ("RECORDING", "IDLE") in seen


@pytest.mark.asyncio
async def test_turn_row_persisted(default_config: Config, tmp_db: Path) -> None:
    default_config.conversation.follow_up_ms = 50
    default_config.conversation.followup_guard_ms = 0
    pipeline = _build_pipeline(default_config, tmp_db)
    pipeline.event_log.start()
    await pipeline.tool_registry.initialize()

    await pipeline._event_queue.put(
        PipelineEvent(type=WAKE_DETECTED, payload={"keyword": "hey_carl_mock"})
    )
    async def _run_one():
        event = await pipeline._event_queue.get()
        await pipeline._dispatch(event)
    await asyncio.wait_for(_run_one(), timeout=5.0)
    await pipeline.memory_store.close()
    await pipeline.event_log.stop(drain=True)

    conn = connect(tmp_db)
    row = conn.execute("SELECT * FROM turns LIMIT 1").fetchone()
    conn.close()
    assert row is not None
    assert row["user_transcript"]
    assert row["assistant_text"]
    assert row["ended_at"] is not None
