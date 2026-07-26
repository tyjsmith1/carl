-- Migration 001: Initial schema.
-- Corresponds to docs/specs/event-log.md and docs/specs/memory.md.
-- Applied at PRAGMA user_version = 0 → sets user_version = 1.

CREATE TABLE conversations (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default',
    started_at      REAL NOT NULL,               -- unixepoch subsec
    ended_at        REAL,
    end_reason      TEXT,
    turn_count      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_conversations_started ON conversations(started_at DESC);

CREATE TABLE turns (
    id                          TEXT PRIMARY KEY,
    conversation_id             TEXT NOT NULL REFERENCES conversations(id),
    turn_index                  INTEGER NOT NULL,
    user_transcript             TEXT,
    assistant_text              TEXT,
    started_at                  REAL NOT NULL,
    ended_at                    REAL,
    latency_ms_to_first_audio   INTEGER,
    interrupted                 INTEGER NOT NULL DEFAULT 0,   -- SQLite has no BOOL
    UNIQUE(conversation_id, turn_index)
);

CREATE INDEX idx_turns_conv ON turns(conversation_id, turn_index);
CREATE INDEX idx_turns_started ON turns(started_at DESC);

CREATE TABLE events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT REFERENCES conversations(id),
    turn_id         TEXT REFERENCES turns(id),
    stage           TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    created_at      REAL NOT NULL
);

CREATE INDEX idx_events_conv ON events(conversation_id, created_at);
CREATE INDEX idx_events_type ON events(event_type, created_at);
CREATE INDEX idx_events_stage_time ON events(stage, created_at);

CREATE TABLE memories (
    id                     TEXT PRIMARY KEY,
    user_id                TEXT NOT NULL DEFAULT 'default',
    type                   TEXT NOT NULL CHECK (type IN ('fact', 'preference', 'correction', 'routine')),
    content                TEXT NOT NULL,
    subject                TEXT,
    embedding              BLOB,
    embedding_model        TEXT,

    created_at             REAL NOT NULL,
    last_used_at           REAL,
    use_count              INTEGER NOT NULL DEFAULT 0,

    source_conversation_id TEXT REFERENCES conversations(id),
    source_turn_id         TEXT REFERENCES turns(id),
    origin                 TEXT NOT NULL CHECK (origin IN ('explicit', 'llm_inferred', 'consolidation')),

    superseded_by          TEXT REFERENCES memories(id),
    superseded_at          REAL,
    superseded_reason      TEXT,

    confidence             REAL NOT NULL DEFAULT 1.0,
    tombstoned             INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_memories_active
    ON memories(user_id)
    WHERE tombstoned = 0 AND superseded_by IS NULL;

CREATE INDEX idx_memories_type_active
    ON memories(type, user_id)
    WHERE tombstoned = 0 AND superseded_by IS NULL;

CREATE INDEX idx_memories_subject
    ON memories(subject, user_id)
    WHERE subject IS NOT NULL;

-- Short-id lookup: last 6 chars of ULID.
-- SQLite doesn't index on substr() cheaply, so consumers do the substr in Python.

PRAGMA user_version = 1;
