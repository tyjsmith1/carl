"""Event log tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carl.events import EventLog
from carl.events.log import _sanitize
from carl.storage.db import connect


@pytest.mark.asyncio
async def test_log_and_flush(tmp_db: Path) -> None:
    log = EventLog(tmp_db)
    log.start()
    log.log("fsm", "state_transition", {"from": "IDLE", "to": "RECORDING"})
    log.log("stt", "stt_final", {"text": "hello"})
    await log.stop(drain=True)

    conn = connect(tmp_db)
    rows = conn.execute("SELECT stage, event_type, payload_json FROM events ORDER BY id").fetchall()
    conn.close()
    assert len(rows) == 2
    assert rows[0]["stage"] == "fsm"
    assert rows[1]["event_type"] == "stt_final"
    assert json.loads(rows[1]["payload_json"])["text"] == "hello"


@pytest.mark.asyncio
async def test_log_is_nonblocking(tmp_db: Path) -> None:
    """log() must not await; measured to be << 1ms even for 1000 events."""
    import time
    log = EventLog(tmp_db)
    log.start()
    start = time.perf_counter()
    for i in range(1000):
        log.log("fsm", "test", {"i": i})
    elapsed = time.perf_counter() - start
    await log.stop(drain=True)
    # 1000 puts should finish in well under 100ms; we allow 200ms for CI wiggle.
    assert elapsed < 0.2, f"log() enqueue took {elapsed*1000:.1f}ms for 1000 events"


def test_sanitizer_strips_openai_style_key() -> None:
    payload = {"authorization": "sk-abcdefghijklmnop1234567890XYZ"}
    sanitized = _sanitize(payload)
    assert "sk-abcdef" not in json.dumps(sanitized)
    assert "[REDACTED]" in json.dumps(sanitized)


def test_sanitizer_recurses() -> None:
    payload = {"headers": {"auth": ["Bearer sk-abcdefghijklmnop1234567890XYZ"]}}
    sanitized = _sanitize(payload)
    assert "sk-abcdef" not in json.dumps(sanitized)
