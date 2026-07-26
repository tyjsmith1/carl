# Event Log Spec

## Purpose

Structured, append-only record of everything that happened in every conversation. Serves three consumers:

1. **The nightly consolidation job** — reads yesterday's events to extract new memories and detect corrections.
2. **Debugging** — when Carl misbehaves, the event log is the flight recorder.
3. **Future behavioral learning** — trends across days (what tools are called most, which memories get retrieved, latency percentiles).

The rule: **if it happened, it's in the event log.** No exceptions. Stages log fire-and-forget; they never wait on the log write.

## Storage

SQLite table `events` in the same DB file (`~/.carl/carl.db`) as memories. Same file so consolidation can join memories ↔ events in a single transaction, and so backups are one file.

Why SQLite (not JSONL)?
- Queryable by conversation, by stage, by time range without writing custom scanners.
- Consolidation needs `WHERE created_at BETWEEN ... AND stage = 'reasoner'` — trivial in SQL, annoying against JSONL.
- Cross-file joins with `memories` and `conversations` tables.
- Single-writer + WAL mode handles our concurrency (event log writes serialize on one background task).

## Schema

```sql
CREATE TABLE conversations (
    id              TEXT PRIMARY KEY,         -- ULID for time-sortable
    user_id         TEXT NOT NULL DEFAULT 'default',
    started_at      TIMESTAMP NOT NULL,
    ended_at        TIMESTAMP,                -- NULL while active
    end_reason      TEXT,                     -- 'silence' | 'user_stop' | 'error' | ...
    turn_count      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE turns (
    id              TEXT PRIMARY KEY,         -- ULID
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    turn_index      INTEGER NOT NULL,         -- 0-based, per conversation
    user_transcript TEXT,                     -- final STT output
    assistant_text  TEXT,                     -- final spoken response (post-tool loop)
    started_at      TIMESTAMP NOT NULL,
    ended_at        TIMESTAMP,
    latency_ms_to_first_audio INTEGER,        -- for latency SLO tracking
    interrupted     BOOLEAN NOT NULL DEFAULT FALSE,  -- was TTS cut off by barge-in
    UNIQUE(conversation_id, turn_index)
);

CREATE TABLE events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    turn_id         TEXT REFERENCES turns(id),        -- NULL for pre-turn events (wake, VAD)
    stage           TEXT NOT NULL,                    -- see event-type enum
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,                    -- JSON string; shape varies by event_type
    created_at      TIMESTAMP NOT NULL DEFAULT (unixepoch('subsec'))
);

CREATE INDEX idx_events_conv ON events(conversation_id, created_at);
CREATE INDEX idx_events_type ON events(event_type, created_at);
CREATE INDEX idx_events_stage_time ON events(stage, created_at);  -- for consolidation queries
```

**Timestamp choice**: sub-second Unix epoch as a REAL. Cheap, sortable, timezone-free. Present to humans as ISO in queries.

**Payload as JSON blob**: yes, we lose SQL-level introspection into payload fields, but the shape varies enormously per event type. JSON keeps schema evolution painless. If a specific field becomes hot for querying, we add an extracted generated column.

## Event-type catalog

Each stage owns its event types. Payload shapes below are canonical.

### Wake-word stage
| event_type | payload |
|---|---|
| `wake_detected` | `{"keyword": "hey_carl", "score": 0.82, "audio_pre_ms": 500}` |
| `wake_rejected_low_score` | `{"score": 0.32}` (debug only, when logging.level=DEBUG) |

### VAD stage
| event_type | payload |
|---|---|
| `speech_start` | `{"timestamp_offset_ms": 340}` |
| `speech_end` | `{"timestamp_offset_ms": 2810, "silence_ms": 700}` |
| `bargein_triggered` | `{"during_playback_offset_ms": 1240}` |

### STT stage
| event_type | payload |
|---|---|
| `stt_partial` | `{"text": "when was chicag", "is_final": false}` (only if `logging.level=DEBUG`) |
| `stt_final` | `{"text": "when was chicago founded", "confidence": 0.94, "latency_ms": 380}` |
| `stt_error` | `{"error_class": "ConnectionError", "message": "...", "recoverable": true}` |

### Reasoner (LLM) stage
| event_type | payload |
|---|---|
| `llm_request` | `{"model": "claude-sonnet-4-6", "prompt_tokens": 812, "tools": ["web_search", "save_memory"], "messages_hash": "abc123"}` |
| `llm_response` | `{"stop_reason": "end_turn", "output_tokens": 47, "latency_ms": 620, "text_preview": "Chicago was..."}` |
| `llm_tool_use` | `{"tool_name": "web_search", "tool_input": {...}, "tool_use_id": "toolu_abc"}` |
| `llm_tool_result` | `{"tool_use_id": "toolu_abc", "is_error": false, "content_preview": "...", "latency_ms": 1420}` |
| `llm_error` | `{"error_class": "RateLimitError", "message": "...", "retried": true}` |

### TTS stage
| event_type | payload |
|---|---|
| `tts_request` | `{"voice": "nova", "char_count": 82, "text_preview": "Chicago was founded..."}` |
| `tts_first_byte` | `{"latency_ms": 340}` (used to compute time-to-first-audio) |
| `tts_complete` | `{"total_bytes": 48000, "duration_ms": 2100}` |
| `tts_interrupted` | `{"bytes_played": 12000, "text_played_preview": "Chicago was fou-"}` |

### Memory stage
| event_type | payload |
|---|---|
| `memory_retrieve` | `{"query": "chicago", "results_count": 3, "result_ids": ["01H...", "01H..."], "latency_ms": 8}` |
| `memory_save` | `{"memory_id": "01H...", "type": "fact", "content_preview": "...", "trigger": "llm_tool"}` |
| `memory_delete` | `{"memory_id": "01H...", "reason": "user_requested"}` |
| `memory_superseded` | `{"old_id": "01H...", "new_id": "01H...", "reason": "correction"}` |

### System / control
| event_type | payload |
|---|---|
| `session_start` | `{"config_version": 1, "providers": {...}}` |
| `session_end` | `{"uptime_seconds": 3600, "reason": "sigterm"}` |
| `provider_error` | `{"provider": "deepgram", "error_class": "...", "recovering": true}` |
| `consolidation_run` | `{"date": "2026-07-25", "memories_proposed": 4, "memories_saved": 3, "duration_ms": 12000}` |

## Write path

```python
class EventLog:
    """Async event log. Fire-and-forget from stages."""

    def __init__(self, db_path: Path):
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._writer_loop())

    def log(self, stage: str, event_type: str, payload: dict,
            conversation_id: str | None = None,
            turn_id: str | None = None) -> None:
        # Non-blocking put. If queue overflows (should never happen at our
        # rates), drop with a stderr warning — better than blocking a stage.
        try:
            self._queue.put_nowait(Event(stage, event_type, payload,
                                          conversation_id, turn_id, time.time()))
        except asyncio.QueueFull:
            log.warning("event_log queue overflow, dropping event", event_type=event_type)

    async def _writer_loop(self):
        while True:
            batch = [await self._queue.get()]
            # Drain up to 32 more events without blocking to amortize the INSERT
            for _ in range(32):
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._flush(batch)  # single transaction, WAL mode, ~sub-ms

    def _flush(self, batch: list[Event]) -> None:
        with sqlite_conn() as c:
            c.executemany(
                "INSERT INTO events (...) VALUES (?, ?, ?, ?, ?, ?)",
                [e.to_row() for e in batch],
            )
```

**Never** `await` on log writes from the hot path. Log writes must not add to time-to-first-audio latency.

## Read path

Consolidation and debug tools query directly. Common queries:

```sql
-- All turns from yesterday
SELECT t.*, c.user_id
FROM turns t JOIN conversations c ON t.conversation_id = c.id
WHERE date(t.started_at, 'unixepoch', 'localtime') = date('now', '-1 day', 'localtime');

-- All corrections in the last week (user said "no, that's wrong" or LLM detected one)
SELECT * FROM events
WHERE event_type = 'memory_superseded'
  AND created_at > unixepoch('now', '-7 days');

-- Latency percentiles for time-to-first-audio
SELECT
  json_extract(payload_json, '$.latency_ms') AS ms
FROM events
WHERE event_type = 'tts_first_byte'
  AND created_at > unixepoch('now', '-1 day');
```

## Retention

- **Events**: 30 days in the hot table. Nightly job archives older rows to `events_archive` table (same schema). Archive kept indefinitely (small).
- **Audio blobs**: not in the DB. If we ever store raw audio (for debug), it goes to `~/.carl/audio/{conversation_id}/{turn_id}.wav` with a 7-day retention. Off by default; enabled via `logging.store_audio: true`.

Rationale: 30 days of events is enough for the consolidation job (which runs on last-day) and any reasonable debug window. Beyond that, storage grows without corresponding value from the hot path.

## Payload PII policy

Payloads contain full transcripts and LLM responses — so they contain everything the user ever said to Carl. This is fine for single-user local deployment (the user's own machine), but:

- **Never log full audio bytes.** Payload previews are text-only.
- **Never log API keys or bearer tokens** in error payloads. Sanitizer in the write path strips known secret patterns.
- **`memory_save` payload** stores a `content_preview` (first 100 chars), not the full memory — the full memory is already in the `memories` table.
- If we ever move to multi-user or cloud sync, add per-user encryption at that point. For MVP local-only, filesystem permissions on `~/.carl/carl.db` are sufficient.

## Failure modes

| Failure | Behavior |
|---|---|
| SQLite locked (concurrent writer) | WAL mode + single writer task; shouldn't happen. If it does, retry once with backoff, then drop and log to stderr. |
| Disk full | Writer task logs to stderr, drops events. Health check on startup verifies free space > 1 GB. |
| Queue overflow (>10k pending) | Drop with stderr warning. Indicates something is wrong upstream (writer stalled). |
| Malformed payload (non-JSON-serializable) | Sanitizer catches at write time; substitutes `{"error": "payload_serialization_failed", "repr": "..."}`. |
| Schema migration needed | Startup runs `PRAGMA user_version` check; if behind, applies numbered migrations in `carl/storage/migrations/`. Never destructive without explicit flag. |

## Test strategy

- **Unit**: `EventLog.log()` is non-blocking (test that it returns in <100µs).
- **Unit**: batch flush groups events into single transaction.
- **Unit**: queue overflow drops gracefully.
- **Unit**: sanitizer strips known secret patterns from payloads.
- **Integration**: full pipeline run produces expected event sequence (`wake_detected` → `speech_start` → ... → `tts_complete`).
- **Integration**: DB corruption recovery — corrupt the file mid-run, verify restart handles it.
- **Property**: after N random events, event count matches, no ordering violations within a conversation.

## Open questions

- [OPEN] Do we ever want to expose the event log to the LLM as a tool? ("What did I ask you yesterday?") Cool but requires careful bounding of what gets returned. v2.
- [OPEN] Aggregation views (materialized daily rollups for the consolidation job)? Only if the raw scan gets slow — probably not before millions of events.
- [OPEN] Compression on old rows (payload_json can be gzipped in-place). Wait for pain.
