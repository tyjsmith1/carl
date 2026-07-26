"""Memory data model. See docs/specs/memory.md."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MemoryType = Literal["fact", "preference", "correction", "routine"]
MemoryOrigin = Literal["explicit", "llm_inferred", "consolidation"]
SupersedeReason = Literal["correction", "duplicate", "contradiction", "refinement"]


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    user_id: str
    type: MemoryType
    content: str
    subject: str | None
    created_at: datetime
    origin: MemoryOrigin
    confidence: float = 1.0
    last_used_at: datetime | None = None
    use_count: int = 0
    source_conversation_id: str | None = None
    source_turn_id: str | None = None
    superseded_by: str | None = None
    tombstoned: bool = False

    @property
    def short_id(self) -> str:
        """Last 6 chars of the ULID — safe to expose to the LLM for correction tools."""
        return self.id[-6:]
