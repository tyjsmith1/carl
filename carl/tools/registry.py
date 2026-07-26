"""Registry aggregating multiple ToolProviders into a single flat view."""

from __future__ import annotations

from typing import Any

import structlog

from carl.tools.base import ToolContext, ToolDefinition, ToolProvider, ToolResult

log = structlog.get_logger()


class ToolRegistry:
    def __init__(self, providers: list[ToolProvider]):
        self._providers = providers
        self._index: dict[str, ToolProvider] = {}
        self._definitions: list[ToolDefinition] = []

    async def initialize(self) -> None:
        """Discover tools across all providers. Fails on duplicate names."""
        for provider in self._providers:
            for defn in await provider.list_tools():
                if defn.name in self._index:
                    existing = self._index[defn.name].__class__.__name__
                    raise ValueError(
                        f"tool name collision: '{defn.name}' registered by "
                        f"both {existing} and {provider.__class__.__name__}"
                    )
                self._index[defn.name] = provider
                self._definitions.append(defn)
        log.info("tool_registry_initialized", tool_count=len(self._definitions),
                 tools=[d.name for d in self._definitions])

    def definitions(self) -> list[ToolDefinition]:
        return list(self._definitions)

    def definition(self, name: str) -> ToolDefinition | None:
        for d in self._definitions:
            if d.name == name:
                return d
        return None

    async def execute(
        self,
        name: str,
        tool_input: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        provider = self._index.get(name)
        if provider is None:
            return ToolResult(content=f"Unknown tool: {name}", is_error=True)
        return await provider.execute(name, tool_input, context)
