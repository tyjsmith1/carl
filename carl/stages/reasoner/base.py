"""Reasoner stage interface. See docs/specs/reasoner.md."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Message:
    """Turn history entry passed to the reasoner."""
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class ReasonerTurn:
    user_transcript: str
    conversation_id: str
    turn_id: str
    history: list[Message] = field(default_factory=list)
    memories: list = field(default_factory=list)  # list[Memory] but avoiding circular import


@dataclass(frozen=True, slots=True)
class SpokenChunk:
    text: str
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False


OnToolCallCallback = Callable[[ToolCall], Awaitable[ToolResult]]
OnFillerNeededCallback = Callable[[str], Awaitable[None]]


class Reasoner(Protocol):
    def turn(
        self,
        request: ReasonerTurn,
        on_tool_call: OnToolCallCallback | None = None,
        on_filler_needed: OnFillerNeededCallback | None = None,
    ) -> AsyncIterator[SpokenChunk]:
        """Run one reasoning turn. Returns an async iterator of spoken chunks.

        NOTE: this is intentionally NOT `async def`. An `async def` function
        that returns `AsyncIterator[X]` is a coroutine — callers would need
        to `await` it before `async for`. Declaring it as a plain factory
        `def turn(...) -> AsyncIterator[X]` accepts both idioms:
          - `async def turn(...): yield ...`  (async generator)
          - `def turn(...): return _stream(...)`  (returns an iterator)
        The current EchoReasoner uses the generator form.
        """
        ...

    async def abort(self) -> None:
        """Cancel any in-flight streaming turn."""
        ...
