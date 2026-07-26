# Nightly Consolidation Spec

## Purpose

Once a day, while Carl is otherwise idle, a batch job walks yesterday's event log and asks: **what happened today that I should remember tomorrow, and what did I have wrong that I should update?**

This is the mechanism behind "behavioral learning" — the thing that makes Carl feel like it's learning *you* over time, not just accumulating verbatim commands.

**The framing that resonates (from the design brainstorm):** this is analogous to sleep-driven memory consolidation in a real brain. The hippocampus replays the day's experiences and moves some of them into long-term storage in the neocortex. That's exactly what this job does: replay yesterday's transcript, decide what to promote to durable memory, and let the rest fade.

Keep this framing alive in prompts and comments. It shapes intuition about what the job should and shouldn't do.

## When it runs

- **Trigger**: cron, default `0 3 * * *` (3am local time). Config: `consolidation.cron`.
- **Scope**: previous local calendar day (00:00:00 → 23:59:59 in local time).
- **Duration budget**: 5 minutes soft cap, 15 minutes hard cap. If we blow it, something is wrong and we bail with an alert.
- **Concurrency**: single-instance. A file lock (`~/.carl/consolidation.lock`) prevents overlapping runs.
- **Skip conditions**: no conversations yesterday → no-op. Config `consolidation.enabled: false` → no-op.

## Model

Claude Opus 4.7 (`claude-opus-4-7`). Reasoning quality matters more than latency here — the job runs at 3am on last night's data; nobody's waiting on it.

## Input

The full event log for the previous day, plus current active memories.

```sql
-- Yesterday's turns with their transcripts and responses
SELECT t.id, t.turn_index, t.user_transcript, t.assistant_text,
       t.started_at, c.id AS conversation_id
FROM turns t JOIN conversations c ON t.conversation_id = c.id
WHERE date(t.started_at, 'unixepoch', 'localtime') = ?  -- yesterday
ORDER BY c.started_at, t.turn_index;

-- Yesterday's tool calls (for detecting patterns like "always weather")
SELECT * FROM events
WHERE event_type IN ('llm_tool_use', 'llm_tool_result', 'memory_save', 'memory_superseded')
  AND date(created_at, 'unixepoch', 'localtime') = ?;

-- Current active memories (so we can dedupe)
SELECT id, type, subject, content, created_at
FROM memories
WHERE user_id = ? AND tombstoned = FALSE AND superseded_by IS NULL;
```

## Process

```
1. Acquire file lock, or exit.
2. Query yesterday's turns, tool calls, and current memories.
3. If zero turns: log 'consolidation_run' event, exit.
4. Group turns into conversations (already grouped by conversation_id).
5. Format everything into a single prompt for Opus (see prompt template below).
6. Call Opus with structured output (tool-use style — Opus proposes memory
   changes as tool calls; the runner applies them).
7. For each proposed change:
     a. Validate it against current memories (dedup / conflict check).
     b. Apply: save / supersede / tombstone.
     c. Emit event log entries.
8. Log 'consolidation_run' with summary counts.
9. Release lock.
```

## The prompt

Kept in `config/prompts/consolidation.md` so it's iterable without code changes.

```markdown
# Carl's Nightly Consolidation

You are the reflection process for Carl, a personal voice assistant. Once
a day, while Carl rests, you replay yesterday's conversations and decide
what to remember for tomorrow. This is analogous to how a brain
consolidates memories during sleep — the hippocampus replays the day's
experiences and promotes selected ones into long-term storage.

## Your job

Read yesterday's conversations. Compare with what Carl already knows about
the user. Decide:

1. **New memories to save** — stable facts, preferences, corrections,
   or recurring routines that Carl should carry forward.
2. **Memories to supersede** — where the user corrected something, or
   where evidence contradicts an existing memory.
3. **Memories to tombstone** — where the user explicitly asked to forget
   something.

## Guidelines

- **Be conservative.** It is much better to miss a memory than to save
  something that will annoy the user later. When in doubt, don't save.
- **Prefer specific over vague.** "User has a dog named Ruby (golden
  retriever, 3 years old)" beats "User likes dogs."
- **Deduplicate.** If a proposed memory restates something already in
  the current-memories list, skip it. If it refines an existing memory
  (more detail on the same subject), supersede instead of adding a duplicate.
- **Preferences require repetition.** A one-off request in Fahrenheit is
  not a preference; asking in Fahrenheit three times across the day is.
- **Corrections take priority.** If the user pushed back on something,
  save the correction even if you'd otherwise skip.
- **Routines require pattern.** A single 6:45am coffee timer isn't a
  routine. Four days in a row is.
- **Don't save the raw transcript.** Save the *inference*, in Carl's own
  words, from the assistant's perspective.
- **Silent recall stays silent.** Do not add memories that instruct Carl
  to narrate ("mention Ruby by name in every response"). The user prefers
  invisible use of memory.

## Output

Use the tool interface. For each proposed change, call one of:
- `propose_new_memory(type, content, subject, confidence, evidence)`
- `propose_supersede(old_memory_id, new_content, reason, evidence)`
- `propose_tombstone(memory_id, reason, evidence)`

`evidence` is a 1-2 sentence quote-or-summary from the transcript that
justifies the change. This gets logged for audit.

When you're done reviewing everything, call `finalize()`.

## Current memories

{{current_memories_block}}

## Yesterday's conversations

{{transcript_block}}

## Yesterday's tool calls

{{tool_calls_summary}}
```

## The proposal tools

These are tools *only* the consolidation job sees; not part of the runtime assistant tool set.

```python
consolidation_tools = [
    {
        "name": "propose_new_memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"enum": ["fact", "preference", "correction", "routine"]},
                "content": {"type": "string"},
                "subject": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": ["type", "content", "confidence", "evidence"],
        },
    },
    {
        "name": "propose_supersede",
        "input_schema": {
            "type": "object",
            "properties": {
                "old_memory_id": {"type": "string"},
                "new_content": {"type": "string"},
                "reason": {"enum": ["correction", "duplicate", "contradiction", "refinement"]},
                "evidence": {"type": "string"},
            },
            "required": ["old_memory_id", "new_content", "reason", "evidence"],
        },
    },
    {
        "name": "propose_tombstone",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["memory_id", "reason", "evidence"],
        },
    },
    {"name": "finalize", "input_schema": {"type": "object", "properties": {}}},
]
```

The runner accumulates proposals across tool calls until `finalize` is called, then applies them in a single SQL transaction.

## Dry-run mode

`consolidation.dry_run: true` in config runs the whole flow but skips the SQL apply. Instead, it writes proposals to `~/.carl/consolidation-proposals-{date}.json` for human review.

Useful in the first few weeks of running Carl — you want to see what it *would* remember before actually letting it. Once you trust it, flip the flag.

## Safety: what gets rejected before apply

The runner validates each proposal before applying:

| Check | Action if fails |
|---|---|
| `type` is valid | Skip proposal, log warning |
| `old_memory_id` in supersede/tombstone actually exists and is active | Skip, log |
| Proposed `content` is not >500 chars | Skip (probably confused) |
| Proposed new memory duplicates an existing active one (case-insensitive substring match) | Convert to supersede if refinement, else skip |
| More than 20 proposals in one run | Suspicious — likely runaway. Log alarm, apply top 20 by confidence, skip rest |
| Any single memory's `content` contains PII patterns (SSN, credit-card-shaped strings, etc.) | Skip and alert |

## Cost budget

Rough per-run cost estimate:
- Yesterday's transcripts: ~20-50 turns × ~200 tokens each = ~10K input tokens
- Current memories: ~100 memories × ~30 tokens = ~3K tokens
- Prompt template + tool defs: ~1.5K tokens
- **Total input**: ~15K tokens
- Output: ~2K tokens (proposals + reasoning)
- **Opus 4.7 pricing**: input ~$15/M, output ~$75/M
- **Per run**: ~$0.30. Annual: ~$110.

Reasonable for the value delivered. If it grows past $1/day (i.e., input tokens explode), add a summarization pass before the consolidation prompt.

## Observability

Every run emits:

```json
{
  "event_type": "consolidation_run",
  "payload": {
    "date": "2026-07-25",
    "turns_scanned": 34,
    "conversations_scanned": 8,
    "memories_before": 47,
    "memories_proposed": 6,
    "memories_saved": 4,
    "memories_superseded": 1,
    "memories_tombstoned": 0,
    "proposals_rejected": 1,
    "reasons": {"duplicate": 1},
    "input_tokens": 14200,
    "output_tokens": 1840,
    "cost_usd": 0.35,
    "duration_ms": 8200
  }
}
```

Query this over time to see whether consolidation is doing useful work. A run that scans 30 turns and proposes zero memories for weeks is a signal something's off with the prompt.

## Failure modes

| Failure | Behavior |
|---|---|
| Lock file exists (previous run stuck) | Abort with alert. Human intervention required — safer than double-writing. |
| Opus API down | Retry with exponential backoff up to 3 times, then abort. Try again tomorrow. |
| Malformed proposal (tool arg validation fails) | Skip that proposal, continue with the rest. Log which one. |
| SQL apply transaction fails | Rollback, log, don't retry. Next night's job will re-scan the same day's events (since they're still in the log). |
| Job runs longer than hard cap (15 min) | Kill, log, alert. Next night resumes with a fresh session. |
| Zero turns yesterday | No-op, emit event with `turns_scanned: 0`. |
| Model returns text instead of tool call | Prompt failure — log the response, alert, don't apply anything. |

## Test strategy

- **Unit**: proposal validators (duplicate detection, PII check, cap enforcement).
- **Unit**: SQL apply is a single transaction (partial apply is impossible).
- **Integration (dry-run)**: fixture event log for a fake day; run consolidation; assert produced proposals match expected set. This is the "golden test" — a curated day with corrections, preferences, and noise.
- **Integration**: lock-file behavior — two overlapping runs, second aborts cleanly.
- **Integration**: `dry_run: true` writes proposals to file but changes zero DB rows.
- **Cost regression**: input tokens < 30K for a fixture of 50 turns (guard against prompt bloat).
- **Chaos**: kill mid-run, verify next run picks up the same day (or moves on, per resume policy — decide when we get there).

## Open questions

- [OPEN] Should consolidation re-analyze N days back (not just yesterday) to catch corrections it missed the first time? Adds cost and complexity; wait to see if we miss things. Probably a "weekly review" secondary job.
- [OPEN] Should proposals be batched into a review queue that the user approves via voice ("Carl, here's what I'd like to remember from today — approve or reject?") ? Neat UX, invisible-recall violation-adjacent. Discuss with user.
- [OPEN] Cross-user leak protection when multi-user lands: proposals from user A's transcripts must never end up in user B's memory table. Enforce at query time (`user_id` in every WHERE), plus a check at apply time.
- [OPEN] Consolidation-generated `origin = "consolidation"` memories should probably be flagged when injected so the LLM can distinguish "user told me directly" from "I inferred this while reflecting." Adds a paren in the prompt injection format. Minor.
