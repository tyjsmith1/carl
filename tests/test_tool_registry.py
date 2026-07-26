"""Tool registry tests."""

from __future__ import annotations

import pytest

from carl.tools import (
    NativeToolProvider,
    ToolContext,
    ToolRegistry,
    ToolResult,
    clear_registry,
    tool,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.mark.asyncio
async def test_decorator_registers_and_executes() -> None:
    @tool(
        name="hello",
        description="Says hi.",
        input_schema={"type": "object", "properties": {}},
        safety_tier="readonly",
    )
    class _HelloTool:
        def __init__(self, config=None):
            pass
        async def execute(self) -> ToolResult:
            return ToolResult(content="hi")

    provider = NativeToolProvider()
    registry = ToolRegistry([provider])
    await registry.initialize()

    definitions = registry.definitions()
    assert len(definitions) == 1
    assert definitions[0].name == "hello"

    result = await registry.execute("hello", {}, ToolContext())
    assert result.content == "hi"
    assert not result.is_error


@pytest.mark.asyncio
async def test_duplicate_name_across_providers_fails() -> None:
    @tool(name="dup", description="d", input_schema={"type": "object", "properties": {}})
    class _A:
        def __init__(self, config=None): pass
        async def execute(self) -> ToolResult: return ToolResult(content="a")

    provider1 = NativeToolProvider()
    provider2 = NativeToolProvider()
    registry = ToolRegistry([provider1, provider2])
    with pytest.raises(ValueError, match="collision"):
        await registry.initialize()


@pytest.mark.asyncio
async def test_unknown_tool_returns_error() -> None:
    provider = NativeToolProvider()
    registry = ToolRegistry([provider])
    await registry.initialize()
    result = await registry.execute("nonexistent", {}, ToolContext())
    assert result.is_error
    assert "Unknown tool" in result.content


@pytest.mark.asyncio
async def test_dependency_injection() -> None:
    @tool(name="uses_dep", description="d", input_schema={"type": "object", "properties": {}})
    class _WithDep:
        def __init__(self, memory_store):
            self.store = memory_store
        async def execute(self) -> ToolResult:
            return ToolResult(content=f"got={type(self.store).__name__}")

    provider = NativeToolProvider(injected={"memory_store": "fake_store_object"})
    registry = ToolRegistry([provider])
    await registry.initialize()
    result = await registry.execute("uses_dep", {}, ToolContext())
    assert "str" in result.content   # our fake_store is a str


@pytest.mark.asyncio
async def test_tool_with_kwargs_does_not_break_di() -> None:
    """Regression: _resolve_deps must skip **kwargs / *args params."""
    @tool(
        name="varkw",
        description="d",
        input_schema={"type": "object", "properties": {}},
    )
    class _VarKw:
        def __init__(self, config=None, **kwargs):
            self.extra = kwargs
        async def execute(self) -> ToolResult:
            return ToolResult(content="ok")

    provider = NativeToolProvider()   # must not raise
    registry = ToolRegistry([provider])
    await registry.initialize()
    result = await registry.execute("varkw", {}, ToolContext())
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_execute_catches_exceptions() -> None:
    @tool(name="crashes", description="d", input_schema={"type": "object", "properties": {}})
    class _Crashes:
        def __init__(self, config=None): pass
        async def execute(self) -> ToolResult:
            raise RuntimeError("boom")

    provider = NativeToolProvider()
    registry = ToolRegistry([provider])
    await registry.initialize()
    result = await registry.execute("crashes", {}, ToolContext())
    assert result.is_error
    assert "boom" in result.content
