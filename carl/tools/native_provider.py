"""Native tool provider: reads classes decorated with @tool from a Python
package. See docs/specs/tools.md.

Adding a tool:
    from carl.tools import tool, ToolResult
    @tool(name=..., description=..., input_schema=..., safety_tier=...)
    class MyTool:
        def __init__(self, config, **injected): ...
        async def execute(self, **tool_input) -> ToolResult: ...

At startup, `NativeToolProvider.discover(package_name)` imports every module
under that package, which triggers the decorators, populating an internal
registry.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import structlog

from carl.tools.base import SafetyTier, ToolContext, ToolDefinition, ToolResult

log = structlog.get_logger()

# Module-level registry populated by the @tool decorator.
_REGISTRY: dict[str, tuple[type, ToolDefinition]] = {}


def tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    safety_tier: SafetyTier = "readonly",
):
    """Decorator that registers a tool class."""

    def decorate(cls: type) -> type:
        if name in _REGISTRY:
            existing_cls = _REGISTRY[name][0]
            if existing_cls is cls:
                return cls  # idempotent re-import
            raise ValueError(
                f"tool name collision: '{name}' registered by "
                f"{existing_cls.__module__}.{existing_cls.__name__} and "
                f"{cls.__module__}.{cls.__name__}"
            )
        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            safety_tier=safety_tier,
            provider_name="native",
        )
        _REGISTRY[name] = (cls, definition)
        return cls

    return decorate


def clear_registry() -> None:
    """For tests. Do not call in production."""
    _REGISTRY.clear()


class NativeToolProvider:
    """Reads tools from the module-level @tool registry.

    Instantiates each tool class once at construction, passing:
      - `config`: whatever config subsection the caller hands over
      - Any keyword deps whose parameter names match keys in `injected`
        (used to hand a MemoryStore into memory tools, etc.)
    """

    def __init__(self, *, injected: dict[str, Any] | None = None):
        self._injected = injected or {}
        self._instances: dict[str, Any] = {}
        for name, (cls, _defn) in _REGISTRY.items():
            kwargs = self._resolve_deps(cls)
            self._instances[name] = cls(**kwargs)

    @staticmethod
    def discover(package_name: str) -> None:
        """Import every module under `package_name` so decorators run."""
        pkg = importlib.import_module(package_name)
        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return
        for module_info in pkgutil.walk_packages(pkg_path, prefix=pkg.__name__ + "."):
            importlib.import_module(module_info.name)

    def _resolve_deps(self, cls: type) -> dict[str, Any]:
        """Match constructor parameter names to keys in self._injected.
        Extra injected deps are ignored; missing required ones raise.
        Skips *args and **kwargs — they don't need injection."""
        sig = inspect.signature(cls.__init__)
        kwargs: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if pname in self._injected:
                kwargs[pname] = self._injected[pname]
            elif param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"tool {cls.__name__} requires constructor arg '{pname}' "
                    f"but no matching key in NativeToolProvider(injected=...). "
                    f"Provided keys: {list(self._injected)}"
                )
        return kwargs

    async def list_tools(self) -> list[ToolDefinition]:
        return [defn for _, defn in _REGISTRY.values()]

    async def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        if tool_name not in self._instances:
            return ToolResult(content=f"Unknown tool: {tool_name}", is_error=True)
        instance = self._instances[tool_name]
        try:
            # Prefer execute(context=..., **tool_input); fall back to execute(**tool_input).
            sig = inspect.signature(instance.execute)
            if "context" in sig.parameters:
                return await instance.execute(context=context, **tool_input)
            return await instance.execute(**tool_input)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool_execute_failed", tool_name=tool_name)
            return ToolResult(content=f"Tool error: {exc!s}", is_error=True)
