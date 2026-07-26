# Milestone 4 — Memory

## Goal

Wire real memory retrieval into every reasoner turn (auto-injected into the system prompt), expose `save_memory` / `correct_memory` / `recall_memory` / `list_memories` as tools, and implement correction handling via supersession. After this milestone, Carl remembers things across sessions.

## Prerequisites

- Milestone 3 complete (tools working; reasoner tool loop solid).
- `memory.md`, `tools.md`, `reasoner.md` read.
- SQLite `memories` table already exists from Milestone 1's migration.

## Non-goals

- Nightly consolidation (Milestone 6).
- Embedding-based retrieval (`embed_topk` strategy) — MVP uses `dump_all`.
- Multi-user (schema is ready but only `default` user in this milestone).
- Memory-inspection UI beyond the `list_memories` tool.

## What changes vs Milestone 3

1. `SQLiteMemoryStore.retrieve()` gains real content (was stubbed to return `[]`).
2. Reasoner injects memories into the system prompt every turn.
3. Four memory tools registered.
4. Correction handling via `MemoryStore.supersede()`.
5. `follow_up_ms` window is reused for sensitive-tier confirmation (piggyback on same state).

## Files to create/modify

### Modify: `carl/memory/sqlite_store.py`

Implement the full interface per `memory.md`. The retrieve method:

```python
async def retrieve(self, query, user_id="default", k=8, types=None):
    """dump_all strategy: return all active memories, ordered by type priority."""
    async with aiosqlite.connect(self.db_path) as db:
        db.row_factory = aiosqlite.Row
        clause = ("WHERE user_id = ? AND tombstoned = 0 AND superseded_by IS NULL"
                  + (f" AND type IN ({','.join('?' * len(types))})" if types else ""))
        rows = await db.execute_fetchall(
            f"""SELECT * FROM memories {clause}
                ORDER BY CASE type
                    WHEN 'correction'  THEN 0
                    WHEN 'preference'  THEN 1
                    WHEN 'routine'     THEN 2
                    WHEN 'fact'        THEN 3
                END, created_at DESC
                LIMIT 200""",
            [user_id, *(types or [])],
        )
        # Async side effect: update last_used_at + use_count (fire-and-forget)
        ids = [r["id"] for r in rows]
        if ids:
            asyncio.create_task(self._touch(ids))
        return [Memory(**self._row_to_dict(r)) for r in rows]
```

Plus `save`, `supersede`, `tombstone`, `list`, `get` methods per the interface.

### Modify: `carl/pipeline/orchestrator.py`

Before starting a reasoner turn, retrieve memories:

```python
async def _run_reasoner_turn(self, transcript):
    memories = await self.memory_store.retrieve(
        query=transcript,
        user_id="default",
    )
    self.event_log.log("memory", "memory_retrieve",
                       payload={"query": transcript, "results_count": len(memories),
                                "result_ids": [m.id for m in memories]},
                       conversation_id=self._conv_id, turn_id=self._turn_id)
    request = ReasonerTurn(
        user_transcript=transcript,
        conversation_id=self._conv_id,
        turn_id=self._turn_id,
        history=self._history,
        memories=memories,
    )
    # ... rest of turn
```

### Modify: `carl/stages/reasoner/claude.py`

The system prompt builder now injects memories per the format in `memory.md`:

```python
def _build_system_prompt(self, memories):
    base = self._base_prompt   # from config/prompts/carl_base.md
    memory_block = self._format_memories(memories)
    rules = self._behavior_rules
    context = f"Current time: {datetime.now().isoformat()}. User: default."
    return [
        {"type": "text", "text": base, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": memory_block},         # varies per turn, not cached
        {"type": "text", "text": rules + "\n\n" + context, "cache_control": {"type": "ephemeral"}},
    ]

def _format_memories(self, memories):
    if not memories:
        return "<memories>\n(no stored memories yet)\n</memories>"
    grouped = defaultdict(list)
    for m in memories:
        grouped[m.type].append(m)
    lines = ["<memories>"]
    for section, mtype in [("preferences", "preference"), ("corrections", "correction"),
                            ("facts", "fact"), ("routines", "routine")]:
        items = grouped.get(mtype, [])
        if not items: continue
        lines.append(f"<{section}>")
        for m in items:
            lines.append(f"- [{m.id[-6:]}] {m.content}")   # short id for LLM reference
        lines.append(f"</{section}>")
    lines.append("</memories>")
    lines.append("Do not narrate that you remembered these things. The user prefers silent recall.")
    lines.append("Use them naturally in your response only if they're relevant.")
    return "\n".join(lines)
```

**Short IDs**: expose the last-6-chars of the ULID to the LLM. The LLM uses these when calling `correct_memory(old_memory_id=...)`. Store maps them back to full ULIDs at execute time.

### New: memory tools

Four tool classes in `carl/tools/memory_tools.py`:

```python
@tool(
    name="save_memory",
    description="Save a durable fact/preference/correction/routine about the user for future conversations. Use when the user tells you something worth remembering or when you notice a pattern.",
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "type": {"enum": ["fact", "preference", "correction", "routine"]},
            "subject": {"type": "string"},
        },
        "required": ["content", "type"],
    },
    safety_tier="mutating",
)
class SaveMemoryTool:
    def __init__(self, config, memory_store, context):
        self.store = memory_store
        self.context = context  # gets conversation_id + turn_id from pipeline

    async def execute(self, content, type, subject=None):
        m = await self.store.save(
            content=content, type=type, subject=subject,
            origin="llm_inferred",
            source_conversation_id=self.context.conversation_id,
            source_turn_id=self.context.turn_id,
        )
        return ToolResult(content=json.dumps({"success": True, "memory_id": m.id[-6:]}))

@tool(
    name="correct_memory",
    description="The user has corrected a previous fact. Find the offending memory by short_id (last 6 chars of the id shown in <memories>) and supersede it with the corrected version.",
    input_schema={
        "type": "object",
        "properties": {
            "old_memory_short_id": {"type": "string"},
            "corrected_content": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["old_memory_short_id", "corrected_content"],
    },
    safety_tier="mutating",
)
class CorrectMemoryTool:
    async def execute(self, old_memory_short_id, corrected_content, reason=""):
        full_id = await self.store.find_by_short_id(old_memory_short_id)
        if not full_id:
            return ToolResult(is_error=True, content=f"No active memory with short id {old_memory_short_id}")
        new_m = await self.store.supersede(full_id, corrected_content, reason="correction")
        return ToolResult(content=json.dumps({"success": True, "new_memory_id": new_m.id[-6:]}))

@tool(
    name="list_memories",
    description="List memories Carl has stored about the user, optionally filtered by type. Use when the user asks what Carl knows.",
    input_schema={
        "type": "object",
        "properties": {"type": {"enum": ["fact", "preference", "correction", "routine"]}},
    },
    safety_tier="readonly",
)
class ListMemoriesTool: ...

@tool(
    name="forget",
    description="Tombstone a memory. Requires user confirmation because it's not easily undone.",
    input_schema={"type": "object", "properties": {"memory_short_id": {"type": "string"}}, "required": ["memory_short_id"]},
    safety_tier="sensitive",
)
class ForgetTool: ...
```

### Modify: tool registry — sensitive-tier confirmation flow

Per `tools.md`, sensitive-tier tools require voice confirmation. Add to the registry:

```python
class ToolRegistry:
    def __init__(self, ...):
        # ...
        self._pending_confirmations: dict[str, PendingCall] = {}  # per-conversation

    async def execute(self, name, input, *, conversation_id):
        tool_def = self._index[name]
        if tool_def.safety_tier == "sensitive":
            # Store pending, return awaiting_confirmation
            call_id = ulid.new().str
            self._pending_confirmations[conversation_id] = PendingCall(
                call_id=call_id, tool_name=name, tool_input=input,
                created_at=time.time(),
            )
            verb = self._describe_verb(name, input)
            return ToolResult(content=json.dumps({
                "awaiting_confirmation": True,
                "prompt": f"You want me to {verb}. Say yes to confirm.",
                "call_id": call_id,
            }))
        # else: execute normally
```

The reasoner's system prompt gets an addition:

```
If a tool_result contains "awaiting_confirmation": true, speak the exact
"prompt" text and end your turn. Do not add anything else. Do not make
further tool calls in this turn.
```

The pipeline, when starting a new turn, checks if there's a pending confirmation from the same conversation:

```python
async def _on_speech_end(self, event):
    transcript = await self._stt.close()
    pending = self.tool_registry.get_pending(self._conv_id)
    if pending:
        if _is_confirmation(transcript):
            result = await self.tool_registry.execute_pending(pending.call_id)
            # Speak the result via a synthesized "done" phrase, no LLM turn needed
            await self._speak("Done.")
            return
        else:
            self.tool_registry.discard_pending(self._conv_id)
            # Continue as normal turn
```

`_is_confirmation(transcript)` is rule-based first (short list: "yes", "confirm", "do it", "go ahead", "yep"), with an LLM classifier fallback (single Haiku call).

## Definition of done

- [ ] Memory retrieval returns real memories; `dump_all` strategy queries by user + active + type priority.
- [ ] System prompt shows the `<memories>` block with correct grouping.
- [ ] `save_memory` tool works end-to-end: user says "remember I have a dog named Ruby", next session, memory appears in the injection block.
- [ ] `correct_memory` tool works: user corrects a fact; the old memory is superseded (has `superseded_by`), new one appears in retrieval.
- [ ] `list_memories` tool works via voice: "hey Carl, what do you remember about me" → Carl reads relevant memories.
- [ ] `forget` triggers voice confirmation flow: says the prompt, waits, applies iff user confirms.
- [ ] Silent-recall behavior: Carl never says "I remember..." even when using memories (verify with 10 test conversations).
- [ ] `memory_save`, `memory_superseded` events in the log with correct payloads.
- [ ] `pytest tests/test_memory_store.py tests/test_memory_tools.py` passes.
- [ ] Integration test: two-session sequence — session 1 saves memory, session 2 uses it silently.

## Acceptance tests (manual)

**Test 1 — Save + recall across sessions:**
1. *"hey Carl, my dog is named Ruby, she's a golden retriever."*
   - Carl says something like "Got it." (Save memory tool fires; silent otherwise.)
2. Quit and restart Carl.
3. *"hey Carl, what should I feed my dog for dinner?"*
   - Carl gives dog food advice using Ruby's name naturally (e.g. "For a golden like Ruby..."). Never says "I remember Ruby is a golden."

**Test 2 — Correction:**
1. *"hey Carl, remember my dog is a labrador."* (Wrong on purpose.)
2. Later: *"hey Carl, actually Ruby's a golden retriever, not a labrador."*
   - Carl acknowledges without narrating the memory system. Old memory is `superseded_by` the new one.
3. Third turn: *"what breed is my dog"* → "golden retriever."

**Test 3 — Silent recall:**
Ask 5 different questions where Ruby is relevant. Carl uses the name naturally, never announces recall.

**Test 4 — Forget:**
1. *"hey Carl, forget everything about my dog."*
2. Carl: "You want me to forget the memory about your dog Ruby. Say yes to confirm."
3. *"yes"*
4. Carl: "Done."
5. Ask about the dog → Carl doesn't know.

## Traps / lessons

- **Short-id collisions**: last-6-chars of ULID collide at ~1 in a million active memories per user. Acceptable. If it happens, `find_by_short_id` returns multiple; log warning, refuse the correction, ask user to disambiguate.
- **LLM ignores silent-recall rule**: happens especially with Claude Sonnet on first roll-out. Iterate the system prompt. Consider a post-processor that flags/rewrites "I remember..." patterns.
- **Retrieval on huge memory sets**: `LIMIT 200` protects context window. If a user has >200 memories, upgrade to `embed_topk` strategy — but not this milestone.
- **Confirmation flow ambiguity**: user says "yeah go ahead why not" — rule list may miss it, LLM classifier should catch. Log every fallback to Haiku and iterate the rule list based on real usage.
- **Auto-inject slows first-token**: every turn now has memories in the system prompt. Prompt caching mitigates but doesn't eliminate. Measure; if p50 first-token >600ms, tune what gets injected.
