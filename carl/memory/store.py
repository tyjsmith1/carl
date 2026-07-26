"""MemoryStore protocol. Concrete implementations in sqlite_store.py, in_memory_store.py."""

from __future__ import annotations

from typing import Protocol

from carl.memory.models import Memory, MemoryOrigin, MemoryType, SupersedeReason


class MemoryStore(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        user_id: str = "default",
        k: int | None = 8,
        types: list[MemoryType] | None = None,
    ) -> list[Memory]: ...

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
    ) -> Memory: ...

    async def list(
        self,
        *,
        user_id: str = "default",
        type: MemoryType | None = None,
        include_superseded: bool = False,
        include_tombstoned: bool = False,
    ) -> list[Memory]: ...

    async def get(self, memory_id: str, *, user_id: str = "default") -> Memory | None: ...

    async def find_by_short_id(
        self, short_id: str, *, user_id: str = "default"
    ) -> Memory | None: ...

    async def supersede(
        self,
        old_id: str,
        new_content: str,
        *,
        reason: SupersedeReason = "correction",
        source_conversation_id: str | None = None,
        source_turn_id: str | None = None,
    ) -> Memory: ...

    async def tombstone(self, memory_id: str, *, reason: str = "user_requested") -> None: ...
