# Memory Spec

## Purpose

Persistent, structured knowledge about the user that survives across conversations and across restarts. This is *the* differentiator versus off-the-shelf assistants: Carl should build up a picture of who you are, what you prefer, and how you correct it, over time.

Three design commitments (established during the initial brainstorm):

1. **Designed in from day 1, implemented over time.** The three seams (event log, retrieval hook, save-memory tool) are in every version starting v1.
2. **Silent recall.** Carl does not narrate remembered facts. Retrieval is invisible; the user just notices Carl gets it right without having to be reminded.
3. **Correction-aware.** When a user says "no, that's wrong," the offending memory is *demoted*, not left in place. This is what makes Carl feel like it's learning *you*, not just accumulating trivia.

## Storage schema

Same SQLite file as `events` (`~/.carl/carl.db`).

```sql
CREATE TABLE memories (
    id                     TEXT PRIMARY KEY,              -- ULID (time-sortable)
    user_id                TEXT NOT NULL DEFAULT 'default',
    type                   TEXT NOT NULL,                 -- 'fact' | 'preference' | 'correction' | 'routine'
    content                TEXT NOT NULL,                 -- the memory itself, natural language
    subject                TEXT,                          -- optional short slug for grouping ("dog", "morning_routine")
    embedding              BLOB,                          -- optional; NULL until embed_topk retrieval enabled
    embedding_model        TEXT,                          -- which model produced the embedding

    created_at             TIMESTAMP NOT NULL,
    last_used_at           TIMESTAMP,                     -- updated when returned by retrieve()
    use_count              INTEGER NOT NULL DEFAULT 0,

    source_conversation_id TEXT REFERENCES conversations(id),  -- where this memory was born
    source_turn_id         TEXT REFERENCES turns(id),
    origin                 TEXT NOT NULL,                 -- 'explicit' | 'llm_inferred' | 'consolidation'

    superseded_by          TEXT REFERENCES memories(id),  -- non-NULL = this memory is stale
    superseded_at          TIMESTAMP,
    superseded_reason      TEXT,                          -- 'correction' | 'duplicate' | 'contradiction'

    confidence             REAL NOT NULL DEFAULT 1.0,     -- 0.0-1.0; consolidation may lower
    tombstoned             BOOLEAN NOT NULL DEFAULT FALSE -- user-requested "forget"
);

CREATE INDEX idx_memories_user_active ON memories(user_id, tombstoned, superseded_by)
    WHERE tombstoned = FALSE AND superseded_by IS NULL;

CREATE INDEX idx_memories_type ON memories(type, user_id)
    WHERE tombstoned = FALSE AND superseded_by IS NULL;

CREATE INDEX idx_memories_subject ON memories(subject, user_id)
    WHERE subject IS NOT NULL;
```

**Design notes:**

- **ULIDs for IDs.** Time-sortable, human-readable-ish, no coordination needed.
- **`user_id` from day 1** even in single-user mode. Adding voice-ID-based multi-user later is a runtime concern; the schema is already ready.
- **Tombstoning, not deletion.** "Forget that" sets `tombstoned = TRUE` rather than DELETE. Enables "actually, remember that again" undo, and audit trail if you want to know how Carl decided things over time.
- **Supersession, not overwrite.** Corrections create a NEW memory that points to the old one via `superseded_by`. Old memory stays for provenance. Retrieval filters superseded ones out.
- **`embedding` starts NULL.** MVP uses dump-all retrieval; embeddings only get populated when we switch to `embed_topk` strategy.

## Memory types

Typed memory is significantly more useful than a flat blob because retrieval can filter/weight by type.

| Type | Description | Examples |
|---|---|---|
| `fact` | Stable propositions about the user's world | "User has a dog named Ruby (golden retriever)." "User's spouse is Sam." "User lives in Chicago." |
| `preference` | How the user likes things done | "Report weather in Fahrenheit." "Speak concisely by default." "Morning briefing at 7am." |
| `correction` | Explicit override of a prior wrong assumption | "When user says 'the lights' they mean living room, not bedroom." "User's dog is 'Ruby' not 'Rudy'." |
| `routine` | Recurring behaviors/patterns | "Every weekday at 6:45am the user asks for a coffee timer." "On Fridays the user asks for weekend plans." |

Types drive:
- **Retrieval weighting**: preferences and corrections always injected; facts and routines injected only if relevant.
- **Consolidation heuristics**: nightly job knows to promote inferred preferences to explicit ones after N observations.
- **Presentation to LLM**: system prompt formats each type differently (see below).

## MemoryStore interface

The store is the seam. All memory operations go through this; the pipeline never touches SQL directly.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

MemoryType = Literal["fact", "preference", "correction", "routine"]

@dataclass(frozen=True)
class Memory:
    id: str
    user_id: str
    type: MemoryType
    content: str
    subject: str | None
    created_at: datetime
    confidence: float
    origin: Literal["explicit", "llm_inferred", "consolidation"]

class MemoryStore(Protocol):
    async def retrieve(
        self,
        query: str,
        user_id: str = "default",
        k: int = 8,
        types: list[MemoryType] | None = None,
    ) -> list[Memory]: ...

    async def save(
        self,
        content: str,
        type: MemoryType,
        user_id: str = "default",
        subject: str | None = None,
        origin: Literal["explicit", "llm_inferred", "consolidation"] = "explicit",
        source_conversation_id: str | None = None,
        source_turn_id: str | None = None,
    ) -> Memory: ...

    async def list(
        self,
        user_id: str = "default",
        type: MemoryType | None = None,
        include_superseded: bool = False,
    ) -> list[Memory]: ...

    async def supersede(
        self,
        old_id: str,
        new_content: str,
        reason: Literal["correction", "duplicate", "contradiction"],
    ) -> Memory: ...

    async def tombstone(self, memory_id: str, reason: str) -> None: ...

    async def get(self, memory_id: str) -> Memory | None: ...
```

Two implementations to build:

1. `SQLiteMemoryStore` — the real one.
2. `InMemoryMemoryStore` — for tests and dev-mode. Same interface.

## Retrieval strategies

Config selects one via `memory.retrieval_strategy`.

### Strategy: `dump_all` (MVP default)

For small memory counts (<200 items), just return all active memories, sorted by type priority and recency. The LLM's context handles it.

```
1. SELECT * FROM memories
   WHERE user_id = ? AND tombstoned = FALSE AND superseded_by IS NULL
   ORDER BY (CASE type WHEN 'correction' THEN 0
                       WHEN 'preference' THEN 1
                       WHEN 'routine'    THEN 2
                       WHEN 'fact'       THEN 3 END),
            created_at DESC
   LIMIT 200;
2. Group by type.
3. Format into the system prompt (see "Prompt injection" below).
```

Cost: single indexed query, ~ms. Zero infra.

### Strategy: `embed_topk` (when we cross ~200 memories)

Embedding-based semantic search.

```
1. Embed the user's current utterance (or the last N turns concatenated)
   using OpenAI text-embedding-3-small.
2. Cosine similarity vs. all active memories' embeddings.
3. Return top-K by similarity, plus ALWAYS all corrections and preferences
   (regardless of similarity) — those must always be in context.
```

Storage for embeddings: SQLite BLOB column, `sqlite-vec` extension for indexed similarity search. If sqlite-vec proves annoying, fall back to loading all embeddings into memory at startup (200-2000 items is tiny; ~6KB per embedding).

### The interface is the same

Both strategies implement the same `MemoryStore.retrieve()` signature. The pipeline doesn't care which is active. Swap via config.

## Save flow

Two paths for a memory to be born:

### Path 1: LLM tool call (during a conversation)

The reasoner has `save_memory` in its tool set:

```json
{
  "name": "save_memory",
  "description": "Save a durable fact, preference, correction, or routine about the user for future conversations. Use when the user tells you something worth remembering, when they correct you, or when you notice a pattern worth preserving.",
  "input_schema": {
    "type": "object",
    "properties": {
      "content": {"type": "string", "description": "The memory in natural language, from the assistant's perspective."},
      "type": {"enum": ["fact", "preference", "correction", "routine"]},
      "subject": {"type": "string", "description": "Optional short slug for grouping related memories, e.g. 'dog', 'morning_routine'."}
    },
    "required": ["content", "type"]
  }
}
```

When the LLM calls it:
- `origin = "llm_inferred"` (or `"explicit"` if triggered by "remember that")
- `source_conversation_id`, `source_turn_id` filled from current context
- Tool result to the LLM: `{"success": true, "memory_id": "01H..."}` — LLM knows it worked and can reference it in the reply.

### Path 2: Nightly consolidation job (see `nightly-consolidation.md`)

Opus 4.7 walks yesterday's event log, proposes new memories, dedupes against existing, writes them with `origin = "consolidation"`.

## Correction handling

The whole point of storing corrections as a first-class type: when the user says "no, that's wrong," we don't just save the correction and leave the old memory in place. We supersede.

### Detection

The LLM classifies each user turn for correction intent (implicitly, via the system prompt). When it detects one, it calls a second tool:

```json
{
  "name": "correct_memory",
  "description": "The user has corrected a previous fact or assumption. Find the offending memory and supersede it with the corrected version.",
  "input_schema": {
    "type": "object",
    "properties": {
      "old_memory_id": {"type": "string"},
      "corrected_content": {"type": "string"},
      "reason": {"type": "string"}
    },
    "required": ["old_memory_id", "corrected_content"]
  }
}
```

The LLM identifies `old_memory_id` from the memories injected in its context. `MemoryStore.supersede()` does the rest:

```python
async def supersede(self, old_id, new_content, reason):
    # 1. Fetch old memory (get its type + subject to preserve).
    # 2. Create new memory with same type/subject, origin='explicit', confidence=1.0.
    # 3. UPDATE old row: superseded_by = new_id, superseded_at = now, reason.
    # 4. Log memory_superseded event.
```

Retrieval already filters `WHERE superseded_by IS NULL`, so the old memory silently drops out of future prompts. But it's still in the DB for audit.

### What "sophisticated enough to be good" means

- Correction detection is LLM-driven, not rule-based. This works because the LLM already has the memories in context and understands natural language contradiction.
- Supersession is one-hop (new memory replaces old). We don't try to reason about chains ("your original memory came from a session two weeks ago, which had a different context...").
- Users can also correct via explicit voice: "forget that Ruby is a labrador — she's a golden." Handled the same way through `correct_memory`.
- We accept some false negatives (missed corrections). Consolidation catches many of them the next night by re-analyzing the transcript.

## Prompt injection format

Every reasoner turn, retrieved memories are formatted into the system prompt like this:

```
<memories>
<preferences>
- Prefers weather in Fahrenheit.
- Wants morning briefing at 7am.
- Speak concisely.
</preferences>

<corrections>
- When user says "the lights" they mean the living room, not the bedroom.
</corrections>

<facts>
- User has a dog named Ruby (golden retriever, 3 years old).
- User's spouse is Sam.
- User lives in Chicago.
</facts>

<routines>
- Weekday mornings at ~6:45am, user asks for a coffee timer.
</routines>
</memories>

Do not narrate that you remembered these things — the user prefers silent recall.
Use them naturally in your response only if they're relevant.
```

The `Do not narrate` line is critical. Per user preference: silent recall. See `feedback-brainstorm-style` and `project-carl` memory notes for rationale.

## Multi-user readiness

`user_id` is present in schema and interface but hardcoded to `"default"` in v1. To enable multi-user later:

1. Add a voice-ID stage between wake-word and STT that classifies the speaker.
2. Pass the resulting `user_id` through the pipeline into every memory call.
3. No schema migration needed.

## Failure modes

| Failure | Behavior |
|---|---|
| SQLite corruption | Startup integrity check; if broken, back up to `carl.db.corrupt-{timestamp}`, start fresh with empty tables. User notified via next spoken turn. |
| Retrieval returns malformed row | Skip that row, log, continue. Never fail a whole turn on one bad memory. |
| Save called during pipeline shutdown | Sync write to a JSONL "pending" file; replay on next startup. |
| LLM calls `save_memory` with invalid `type` | Pydantic validation on tool input; tool_result returns `{"error": "..."}`; LLM sees it and can retry. |
| LLM calls `correct_memory` with unknown `old_memory_id` | Same — tool_result error, LLM adjusts. |
| Embedding provider (OpenAI) down when using `embed_topk` | Fall back to `dump_all` for that turn; log warning. |
| User says "forget everything" | Explicit tool `tombstone_all_memories` (in `sensitive` tier — requires voice confirmation). Not exposed by default. |

## Test strategy

- **Unit**: `MemoryStore` interface — CRUD, supersede, tombstone, retrieve with type filter.
- **Unit**: retrieval query filters superseded and tombstoned rows.
- **Unit**: supersession preserves old memory row and links correctly.
- **Unit**: ULID ordering — `list()` returns creation-order for a stream of saves.
- **Integration**: full LLM turn with memories in context — system prompt contains the injection block.
- **Integration**: correction round-trip — user corrects a fact, next conversation's retrieval shows only the new version.
- **Property**: for any sequence of save/supersede/tombstone ops, `retrieve()` returns a set consistent with the final state (no orphan references, no superseded leaks).
- **Consolidation-adjacent**: given a fixture event log, consolidation dry-run produces expected candidate memories. (Detail in `nightly-consolidation.md`.)

## Open questions

- [OPEN] Confidence decay over time — should facts unused for months drift toward `confidence = 0.5` so consolidation reconsiders them? Neat but not urgent.
- [OPEN] Cross-memory reasoning — when consolidating, should we let the LLM merge two related facts into one summary? Yes probably; add to consolidation prompt when we get there.
- [OPEN] Explicit "what do you remember about X" tool for the user. Trivial to add; belongs in `tools.md` when we build memory-inspection tools.
- [OPEN] Export/import — one command to `carl memory export → memories.jsonl`. Useful for backup and for transferring Carl to a new machine. v2.
