"""Wiring smoke-test tool. Real tools land in Milestone 3."""

from __future__ import annotations

from typing import Any

from carl.tools import ToolResult, tool


@tool(
    name="noop",
    description="Does nothing. Returns 'ok'. Used only to smoke-test the tool registry wiring; do not use this tool in a real conversation.",
    input_schema={"type": "object", "properties": {}},
    safety_tier="readonly",
)
class NoopTool:
    def __init__(self, config: Any = None):
        pass

    async def execute(self) -> ToolResult:
        return ToolResult(content="ok")
