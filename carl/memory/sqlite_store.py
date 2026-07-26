"""SQLite-backed MemoryStore. See docs/specs/memory.md."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import ulid

from carl.memory.models import Memory, MemoryOrigin, MemoryType, SupersedeReason
from carl.storage.db import aconnect

if TYPE_CHECKING:
    from carl.config.schema import MemoryConfig


def _now_epoch() -> float:
    return time.time()


def _to_dt(epoch: float | None) -> datetime | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=UTC)


def _row_to_memory(row: aiosqlite.Row) -> Memory:
    return Memory(
        id=row["id"],
        user_id=row["user_id"],
        type=row["type"],
        content=row["content"],
        subject=row["subject"],
        created_at=_to_dt(row["created_at"]),  # type: ignore[arg-type]
        origin=row["origin"],
        confidence=row["confidence"],
        last_used_at=_to_dt(row["last_used_at"]),
        use_count=row["use_count"],
        source_conversation_id=row["source_conversation_id"],
        source_turn_id=row["source_turn_id"],
        superseded_by=row["superseded_by"],
        tombstoned=bool(row["tombstoned"]),
    )


_TYPE_PRIORITY = "CASE type WHEN 'correction' THEN 0 WHEN 'preference' THEN 1 WHEN 'routine' THEN 2 WHEN 'fact' THEN 3 ELSE 4 END"


class SQLiteMemoryStore:
    """Async SQLite-backed memory store. Implements MemoryStore protocol."""

    def __init__(self, db_path: Path, config: MemoryConfig | None = None):
        self._db_path = db_path
        # Config wires retrieval strategy, default k, etc. through from the
        # top-level YAML. Milestone-1: dump_all only, embed_topk lands later.
        self._config = config
        self._default_k: int | None = config.top_k if config is not None else 8
        # Strong references to fire-and-forget _touch tasks so they aren't
        # garbage-collected mid-flight (Python 3.11+ only weakly references
        # active tasks). Each task removes itself via done_callback.
        self._pending_touches: set[asyncio.Task] = set()

    async def retrieve(
        self,
        query: str,
        *,
        user_id: str = "default",
        k: int | None = None,
        types: list[MemoryType] | None = None,
    ) -> list[Memory]:
        """dump_all strategy: return all active memories ordered by type priority.
        `query` is currently unused; embed_topk strategy will use it later.
        Pass `k=None` (default) to use config.memory.top_k; `k=0` returns an empty list.
        """
        if k is None:
            k = self._default_k
        conn = await aconnect(self._db_path)
        try:
            sql = (
                "SELECT * FROM memories "
                "WHERE user_id = ? AND tombstoned = 0 AND superseded_by IS NULL"
            )
            params: list = [user_id]
            if types:
                placeholders = ",".join("?" * len(types))
                sql += f" AND type IN ({placeholders})"
                params.extend(types)
            sql += f" ORDER BY {_TYPE_PRIORITY}, created_at DESC LIMIT 200"

            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            memories = [_row_to_memory(r) for r in rows]

            # Slice FIRST — only touch the memories that actually reached the
            # LLM, so use_count / last_used_at reflect real usage (needed for
            # M6 recency ranking) instead of every row we happened to fetch.
            result = memories if k is None else memories[:k]
            if result:
                task = asyncio.create_task(self._touch([m.id for m in result]))
                self._pending_touches.add(task)
                task.add_done_callback(self._pending_touches.discard)
            return result
        finally:
            await conn.close()

    async def _touch(self, ids: list[str]) -> None:
        conn = await aconnect(self._db_path)
        try:
            placeholders = ",".join("?" * len(ids))
            now = _now_epoch()
            await conn.execute(
                f"UPDATE memories SET last_used_at = ?, use_count = use_count + 1 "
                f"WHERE id IN ({placeholders})",
                [now, *ids],
            )
            await conn.commit()
        finally:
            await conn.close()

    async def close(self) -> None:
        """Drain in-flight fire-and-forget tasks (e.g. _touch) before shutdown.
        Called from Pipeline._cleanup so pending UPDATEs complete and connections
        close cleanly instead of getting GC'd with the event loop."""
        if self._pending_touches:
            await asyncio.gather(*self._pending_touches, return_exceptions=True)

    async def save(
        self,
        *,
        content: str,
        type: MemoryType,
        user_id: str = "default",
        subject: str | None = None,
        origin: MemoryOrigin = "explicit",
        source_conversation_id: str | None = None,
        source_turn_id: str | None = None,
        confidence: float = 1.0,
    ) -> Memory:
        memory_id = str(ulid.new())
        now = _now_epoch()
        conn = await aconnect(self._db_path)
        try:
            await conn.execute(
                "INSERT INTO memories "
                "(id, user_id, type, content, subject, created_at, origin, confidence, "
                " source_conversation_id, source_turn_id, tombstoned, use_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)",
                (
                    memory_id, user_id, type, content, subject, now, origin, confidence,
                    source_conversation_id, source_turn_id,
                ),
            )
            await conn.commit()
        finally:
            await conn.close()

        return Memory(
            id=memory_id,
            user_id=user_id,
            type=type,
            content=content,
            subject=subject,
            created_at=_to_dt(now),  # type: ignore[arg-type]
            origin=origin,
            confidence=confidence,
            source_conversation_id=source_conversation_id,
            source_turn_id=source_turn_id,
        )

    async def list(
        self,
        *,
        user_id: str = "default",
        type: MemoryType | None = None,
        include_superseded: bool = False,
        include_tombstoned: bool = False,
    ) -> list[Memory]:
        conn = await aconnect(self._db_path)
        try:
            sql = "SELECT * FROM memories WHERE user_id = ?"
            params: list = [user_id]
            if not include_tombstoned:
                sql += " AND tombstoned = 0"
            if not include_superseded:
                sql += " AND superseded_by IS NULL"
            if type:
                sql += " AND type = ?"
                params.append(type)
            sql += f" ORDER BY {_TYPE_PRIORITY}, created_at DESC"

            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [_row_to_memory(r) for r in rows]
        finally:
            await conn.close()

    async def get(self, memory_id: str, *, user_id: str = "default") -> Memory | None:
        conn = await aconnect(self._db_path)
        try:
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            row = await cursor.fetchone()
            return _row_to_memory(row) if row else None
        finally:
            await conn.close()

    async def find_by_short_id(
        self, short_id: str, *, user_id: str = "default"
    ) -> Memory | None:
        """Look up an active memory by its last-6-char short_id.
        Returns None if not found, or if multiple match (ambiguous)."""
        conn = await aconnect(self._db_path)
        try:
            # UPPER on both sides — ULIDs are stored uppercase, but Claude
            # sometimes lowercases the short_id when echoing it back in a
            # tool call, and a case-sensitive equality would silently return
            # 'not found' for a memory that actually exists.
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND tombstoned = 0 "
                "AND superseded_by IS NULL AND UPPER(substr(id, -6)) = UPPER(?)",
                (user_id, short_id),
            )
            rows = await cursor.fetchall()
            if len(rows) != 1:
                return None
            return _row_to_memory(rows[0])
        finally:
            await conn.close()

    async def supersede(
        self,
        old_id: str,
        new_content: str,
        *,
        reason: SupersedeReason = "correction",
        source_conversation_id: str | None = None,
        source_turn_id: str | None = None,
    ) -> Memory:
        old = await self.get(old_id)
        if old is None:
            raise ValueError(f"memory {old_id} not found")
        if old.superseded_by is not None:
            raise ValueError(f"memory {old_id} is already superseded")

        new_id = str(ulid.new())
        now = _now_epoch()

        conn = await aconnect(self._db_path)
        try:
            await conn.execute("BEGIN")
            # Insert new memory carrying old type + subject
            await conn.execute(
                "INSERT INTO memories "
                "(id, user_id, type, content, subject, created_at, origin, confidence, "
                " source_conversation_id, source_turn_id, tombstoned, use_count) "
                "VALUES (?, ?, ?, ?, ?, ?, 'explicit', 1.0, ?, ?, 0, 0)",
                (
                    new_id, old.user_id, old.type, new_content, old.subject, now,
                    source_conversation_id, source_turn_id,
                ),
            )
            # Mark old as superseded
            await conn.execute(
                "UPDATE memories SET superseded_by = ?, superseded_at = ?, "
                "superseded_reason = ? WHERE id = ?",
                (new_id, now, reason, old_id),
            )
            await conn.commit()
        finally:
            await conn.close()

        return Memory(
            id=new_id,
            user_id=old.user_id,
            type=old.type,
            content=new_content,
            subject=old.subject,
            created_at=_to_dt(now),  # type: ignore[arg-type]
            origin="explicit",
            source_conversation_id=source_conversation_id,
            source_turn_id=source_turn_id,
        )

    async def tombstone(self, memory_id: str, *, reason: str = "user_requested") -> None:
        conn = await aconnect(self._db_path)
        try:
            await conn.execute(
                "UPDATE memories SET tombstoned = 1 WHERE id = ?",
                (memory_id,),
            )
            await conn.commit()
        finally:
            await conn.close()
