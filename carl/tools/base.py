"""Tool system data model + provider Protocol. See docs/specs/tools.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

SafetyTier = Literal["readonly", "mutating", "sensitive"]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    safety_tier: SafetyTier = "readonly"
    provider_name: str = "native"


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Injected into tool execute() calls. Lets tools reach back into the
    current turn/conversation for provenance (memory tools use this)."""
    conversation_id: str | None = None
    turn_id: str | None = None
    user_id: str = "default"


class ToolProvider(Protocol):
    """A source of tools. NativeToolProvider reads decorated Python classes;
    MCPToolProvider (v2) will wrap an MCP server session."""

    async def list_tools(self) -> list[ToolDefinition]: ...

    async def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult: ...
