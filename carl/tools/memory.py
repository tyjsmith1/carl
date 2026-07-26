"""Memory tools — save/correct/list/forget. See docs/specs/memory.md.

The LLM auto-receives injected memories in every system prompt (see
reasoner.claude._build_system_prompt). These tools let the LLM *manage*
memory — save durable facts, correct wrong ones, list what's stored, and
forget on user request.

Every tool takes a `context: ToolContext` param so we can attach the
current conversation/turn as source_* on new/superseded memories, giving
the nightly consolidation job (M6) provenance to work from.

Design notes:
- `save_memory`: origin="llm_inferred" because the LLM decides what's worth
  saving from the conversation. Explicit "remember that" requests get the
  same origin — user intent is inferred from language.
- `correct_memory`: uses short_id (last 6 chars of ULID) because that's
  what's shown in the memory-injection block. Full IDs are 26 chars.
- `list_memories`: returns a compact JSON summary; the LLM speaks the
  natural-language version. Not for prod use ("what do you know about me").
- `forget`: safety_tier="sensitive" as a signal, but voice-confirmation
  flow isn't wired at the registry level yet — see docs/BACKLOG.md.
  Interim: the system prompt instructs the LLM to confirm verbally with
  the user before calling forget, and only call it after the user's next
  affirmative turn.
"""

from __future__ import annotations

import json

import structlog

from carl.memory import MemoryStore
from carl.tools import ToolContext, ToolResult, tool

log = structlog.get_logger()


@tool(
    name="save_memory",
    description=(
        "Save a durable fact, preference, correction, or routine about the "
        "user for future conversations. Call this when the user tells you "
        "something worth remembering (their dog's name, how they take their "
        "coffee, a recurring routine), OR when you notice a pattern across "
        "turns that would help you in future sessions. Do NOT call for "
        "one-off conversational context or for facts you'll only use in the "
        "current turn."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The memory in your own words, from the assistant's "
                    "perspective. E.g. 'User has a dog named Ruby, golden "
                    "retriever, 3 years old.' Not 'the user said X'."
                ),
            },
            "type": {
                "enum": ["fact", "preference", "correction", "routine"],
                "description": (
                    "fact = stable proposition about the user's world; "
                    "preference = how the user likes things done; "
                    "correction = explicit override of a prior wrong assumption; "
                    "routine = recurring behavior/pattern."
                ),
            },
            "subject": {
                "type": "string",
                "description": (
                    "Optional short slug for grouping related memories, "
                    "e.g. 'dog', 'morning_routine', 'spouse'."
                ),
            },
        },
        "required": ["content", "type"],
    },
    safety_tier="mutating",
    provider_group=None,   # always available (no external provider dep)
)
class SaveMemoryTool:
    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store

    async def execute(
        self,
        context: ToolContext,
        content: str,
        type: str,
        subject: str | None = None,
    ) -> ToolResult:
        m = await self._store.save(
            content=content,
            type=type,           # type: ignore[arg-type]
            subject=subject,
            user_id=context.user_id,
            origin="llm_inferred",
            source_conversation_id=context.conversation_id,
            source_turn_id=context.turn_id,
        )
        log.info("save_memory_ok", type=type, short_id=m.short_id,
                 content_preview=content[:60])
        return ToolResult(content=json.dumps({
            "success": True,
            "memory_id": m.short_id,
            "type": m.type,
        }))


@tool(
    name="correct_memory",
    description=(
        "The user has corrected something a previous memory got wrong. "
        "Find the offending memory by its short_id (shown in brackets in "
        "the <memories> block, e.g. '[abc123]') and supersede it with the "
        "corrected version. Use when the user explicitly contradicts a fact "
        "you're operating on."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "old_memory_short_id": {
                "type": "string",
                "description": (
                    "The 6-character short_id from the <memories> block "
                    "(the part inside brackets, e.g. 'abc123'). NOT the "
                    "full 26-char ULID."
                ),
            },
            "corrected_content": {
                "type": "string",
                "description": (
                    "The corrected memory in your own words, from the "
                    "assistant's perspective."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Brief description of what was wrong.",
            },
        },
        "required": ["old_memory_short_id", "corrected_content"],
    },
    safety_tier="mutating",
    provider_group=None,
)
class CorrectMemoryTool:
    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store

    async def execute(
        self,
        context: ToolContext,
        old_memory_short_id: str,
        corrected_content: str,
        reason: str = "correction",
    ) -> ToolResult:
        old = await self._store.find_by_short_id(
            old_memory_short_id, user_id=context.user_id,
        )
        if old is None:
            return ToolResult(
                content=f"No active memory found with short_id '{old_memory_short_id}'.",
                is_error=True,
            )
        new = await self._store.supersede(
            old.id,
            corrected_content,
            reason="correction",
            source_conversation_id=context.conversation_id,
            source_turn_id=context.turn_id,
        )
        log.info("correct_memory_ok", old_short_id=old.short_id,
                 new_short_id=new.short_id)
        return ToolResult(content=json.dumps({
            "success": True,
            "old_memory_id": old.short_id,
            "new_memory_id": new.short_id,
        }))


@tool(
    name="list_memories",
    description=(
        "Return everything Carl currently remembers about the user, optionally "
        "filtered by type. Use when the user asks 'what do you know about me' "
        "or wants to audit what's stored. Speak the results naturally — don't "
        "read out short_ids or JSON structure."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "type": {
                "enum": ["fact", "preference", "correction", "routine"],
                "description": "Optional filter — omit to return all types.",
            },
        },
    },
    safety_tier="readonly",
    provider_group=None,
)
class ListMemoriesTool:
    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store

    async def execute(
        self,
        context: ToolContext,
        type: str | None = None,
    ) -> ToolResult:
        memories = await self._store.list(
            user_id=context.user_id,
            type=type,           # type: ignore[arg-type]
        )
        summary = [
            {
                "short_id": m.short_id,
                "type": m.type,
                "content": m.content,
                "subject": m.subject,
            }
            for m in memories
        ]
        return ToolResult(content=json.dumps({
            "count": len(summary),
            "memories": summary,
        }))


@tool(
    name="forget",
    description=(
        "Tombstone a memory the user has asked to remove. Confirmation "
        "protocol: (1) On the FIRST request, do not call this tool. Instead "
        "say a short confirmation prompt ('You want me to forget X — say yes "
        "to confirm.') and end your turn. (2) On the very next user turn, if "
        "their message contains any affirmative ('yes', 'yeah', 'yep', "
        "'confirm', 'do it', 'go ahead', 'that's right', 'correct') — even "
        "if they also re-state the request or add extra words — call this "
        "tool immediately. Do NOT ask a second time. Only skip the call if "
        "the user clearly changed their mind ('no', 'wait', 'cancel') or "
        "switched topics."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "memory_short_id": {
                "type": "string",
                "description": "The 6-character short_id of the memory to forget.",
            },
        },
        "required": ["memory_short_id"],
    },
    safety_tier="sensitive",
    provider_group=None,
)
class ForgetTool:
    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store

    async def execute(
        self,
        context: ToolContext,
        memory_short_id: str,
    ) -> ToolResult:
        m = await self._store.find_by_short_id(
            memory_short_id, user_id=context.user_id,
        )
        if m is None:
            return ToolResult(
                content=f"No active memory found with short_id '{memory_short_id}'.",
                is_error=True,
            )
        await self._store.tombstone(m.id, reason="user_forget")
        log.info("forget_ok", short_id=m.short_id,
                 content_preview=m.content[:60])
        return ToolResult(content=json.dumps({
            "success": True,
            "forgotten_id": m.short_id,
        }))
