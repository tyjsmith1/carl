# Tools Spec

## Purpose

Tools are how Carl acts on the world beyond talking: searching the web, saving/recalling memories, setting timers, controlling smart-home devices (later). This spec covers the orchestration model (LLM-native tool-use), the registry, per-tool contracts, and safety.

Design commitment (from brainstorm): **the LLM is the orchestrator.** No router layer, no classifier stage, no LangChain-style planner. Claude's native tool-calling picks the right tool; we execute and hand back results; the LLM decides if it's done.

## ToolProvider interface

Tools come from providers. Two provider implementations planned:

```python
class ToolProvider(Protocol):
    async def list_tools(self) -> list[ToolDefinition]:
        """Return the tools this provider exposes."""

    async def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Execute a tool call."""

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict              # JSON schema
    safety_tier: Literal["readonly", "mutating", "sensitive"]
    provider_name: str              # for tie-breaking + logging

@dataclass(frozen=True)
class ToolResult:
    content: str | list[dict]       # what gets passed back as tool_result
    is_error: bool = False
    metadata: dict = field(default_factory=dict)  # cost, latency, etc.
```

**Provider implementations:**

1. `NativeToolProvider` — Python tools decorated in `carl/tools/*.py`, auto-discovered at startup. This is the MVP.
2. `MCPToolProvider` — connects to MCP servers over stdio/HTTP, exposes their tools. **Not built for MVP**, but the interface is here so adding it later is a new class, not a refactor.

The `ToolRegistry` aggregates all providers into a single flat view for the LLM:

```python
class ToolRegistry:
    def __init__(self, providers: list[ToolProvider]):
        self._providers = providers
        self._index: dict[str, ToolProvider] = {}    # tool_name → provider

    async def all_definitions(self) -> list[ToolDefinition]: ...
    async def execute(self, name: str, input: dict) -> ToolResult:
        provider = self._index[name]
        return await provider.execute(name, input)
```

## Native tool registration

Plugin-style. Each tool is a class in `carl/tools/`. A decorator registers it:

```python
# carl/tools/web_search.py

from carl.tools import tool, ToolResult

@tool(
    name="web_search",
    description=(
        "Search the web for current information. Use when the user asks "
        "about facts you're not sure about, current events, prices, or "
        "anything that could have changed recently. Do not use for basic "
        "knowledge you already have."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query, natural language."},
        },
        "required": ["query"],
    },
    safety_tier="readonly",
)
class WebSearchTool:
    def __init__(self, config):
        from tavily import AsyncTavilyClient
        self.client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.config = config

    async def execute(self, query: str) -> ToolResult:
        r = await self.client.search(
            query=query,
            search_depth=self.config.search_depth,       # "basic" or "advanced"
            max_results=self.config.max_results,
            include_answer=self.config.include_answer,
        )
        # Compact shape for the LLM
        payload = {
            "answer": r.get("answer"),
            "results": [
                {"title": x["title"], "url": x["url"], "snippet": x["content"]}
                for x in r.get("results", [])
            ],
        }
        return ToolResult(content=json.dumps(payload), metadata={"credits_used": ...})
```

Startup: `carl/tools/__init__.py` imports all files in the directory, decorators register into a module-level registry, `NativeToolProvider` reads from that.

**No manifest files.** The decorator IS the registration. Drop a file in the dir, restart, tool is live.

## The description field is prompt engineering

The `description` string is *the* thing the LLM uses to decide when to reach for the tool. It matters more than any other part of the tool. Rules:

- **Say when to use it AND when not to.** "Use for current events. Do not use for basic knowledge you already have." Both halves.
- **Speak in imperatives to the model.** ("Search the web for...", not "This tool searches the web.")
- **Give examples in the description if a class of query is ambiguous.** ("For weather questions like 'is it raining', prefer get_weather over web_search.")
- **Iterate on it.** If Carl reaches for a tool when it shouldn't (or fails to when it should), the fix is in the description, not the surrounding code.

## Built-in tools for MVP

Only web_search + memory tools ship on day one. Everything else is v2+ per user's earlier answer.

### `web_search` — see above. Safety: `readonly`.

### `save_memory` — Persist a new memory. Safety: `mutating`.

```python
@tool(
    name="save_memory",
    description=(
        "Save a durable fact, preference, correction, or routine about the "
        "user for future conversations. Use when the user tells you "
        "something worth remembering, when they correct you, or when you "
        "notice a pattern worth preserving. Do not use for one-off "
        "conversational context."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory, in Carl's own words, from the assistant's perspective."},
            "type": {"enum": ["fact", "preference", "correction", "routine"]},
            "subject": {"type": "string", "description": "Optional short slug for grouping related memories, e.g. 'dog', 'morning_routine'."},
        },
        "required": ["content", "type"],
    },
    safety_tier="mutating",
)
class SaveMemoryTool:
    def __init__(self, memory_store): ...
    async def execute(self, content, type, subject=None, *, context) -> ToolResult:
        m = await self.memory_store.save(
            content=content, type=type, subject=subject,
            origin="llm_inferred",
            source_conversation_id=context.conversation_id,
            source_turn_id=context.turn_id,
        )
        return ToolResult(content=json.dumps({"success": True, "memory_id": m.id}))
```

### `correct_memory` — Supersede an existing memory. Safety: `mutating`.

See `memory.md` for the schema. Registered analogously.

### `recall_memory` — Explicitly query memory. Safety: `readonly`.

Mostly a fallback since memories are auto-injected. Useful for "what do you remember about my dog?" queries.

### `forget` — Tombstone a memory. Safety: `sensitive`.

Requires voice confirmation before applying. See "Safety tiers" below.

### `list_memories` — Return all active memories, optionally filtered. Safety: `readonly`.

Debug + user-facing ("what do you remember about me?").

## Safety tiers

Every tool has a `safety_tier`. The registry enforces different execution semantics per tier:

| Tier | Semantics |
|---|---|
| `readonly` | Execute silently. No confirmation. Errors don't surface to the user unless they materially affect the answer. |
| `mutating` | Execute, then announce briefly ("Noted." or via reasoner's natural response). Error surfaces if it fails. |
| `sensitive` | Two-turn flow: model calls tool → registry intercepts → speaks confirmation prompt → waits for next user turn → applies iff user confirms. |

### Sensitive-tier flow

```
1. LLM emits tool_use for a sensitive tool.
2. Registry sees safety_tier == "sensitive". Instead of executing:
     a. Constructs a confirmation prompt from a template:
        "Okay, you want me to {tool_verb}. Say yes to confirm."
     b. Returns a special ToolResult(content=..., metadata={"awaiting_confirmation": true, "pending_call": {...}}).
3. Reasoner receives that ToolResult. Its system prompt has a rule:
     "If a tool_result indicates awaiting_confirmation, speak the
      prompt exactly and end your turn without further tool calls."
4. Pipeline stores the pending_call in conversation state.
5. Next user turn:
     - If user says a confirming phrase ("yes", "confirm", "do it", "go ahead"):
         → registry executes the pending_call as if freshly received.
     - Otherwise:
         → pending_call discarded; conversation continues normally.
6. Confirming phrase detection: rule-based on the transcript (short list),
   plus fallback to LLM classification on the same turn.
```

Config `tools.require_confirmation` lists which tools force sensitive tier even if not tagged (belt+suspenders for tools that turn out to be more dangerous than expected).

## Parallel tool execution

Claude's tool-use API can emit multiple `tool_use` blocks in one response ("turn off the lights AND set a timer AND tell me the weather"). By default we execute in parallel:

```python
results = await asyncio.gather(*[
    self.registry.execute(tc.name, tc.input) for tc in tool_calls
])
```

Exception: `sensitive`-tier tools serialize. If any block is sensitive, execute all `readonly` and `mutating` calls first, then present the sensitive one for confirmation.

Config `tools.parallel_execution: false` disables the whole thing if we ever hit issues.

## Filler speech during tool execution

Coordinated with the reasoner (see `reasoner.md`):
1. Reasoner emits a `tool_use` block.
2. Pipeline gets `on_filler_needed` callback.
3. Filler phrase picked (rotating, keyed to tool name).
4. TTS speaks filler while tool executes.

Filler phrases live in `config/prompts/filler_phrases.yaml`:

```yaml
web_search:
  - Let me check.
  - One sec.
  - Looking that up.
get_weather:
  - Checking the weather.
save_memory:
  - Noted.
  - Got it.
default:
  - Hold on.
  - One sec.
```

Rules:
- Never same phrase twice in a row.
- If a tool completes in <300ms, skip filler (we'd finish first anyway).
- If tools run in parallel, one filler covers the whole batch.

## MCP integration (v2 seam)

The `MCPToolProvider` is not built for MVP but the interface exists. When we add it:

```python
class MCPToolProvider(ToolProvider):
    def __init__(self, server_command: list[str] | str):
        # spawn MCP server subprocess (stdio) or connect over HTTP
        self.session = MCPSession(server_command)

    async def list_tools(self) -> list[ToolDefinition]:
        raw_tools = await self.session.list_tools()
        return [self._translate(t) for t in raw_tools]

    async def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        result = await self.session.call_tool(tool_name, tool_input)
        return ToolResult(content=result.content, is_error=result.is_error)
```

Config addition when we get there:

```yaml
tools:
  mcp_providers:
    - command: ["uvx", "mcp-server-home-assistant"]
      safety_tier_map:                # override the server's declared tiers
        turn_off_lights: mutating
        unlock_door: sensitive
```

The `safety_tier_map` is important — MCP tools don't have a native safety tier concept; we assign them at config time.

## Tool observability

Every tool call emits event log entries:
- `llm_tool_use` when the LLM requests the tool.
- `llm_tool_result` when it completes (with latency, error, cost if provider reports it).

This lets consolidation notice patterns ("every morning at 7am the user asks for weather → propose a morning briefing routine").

## Failure modes

| Failure | Behavior |
|---|---|
| Tool name unknown | Return `ToolResult(is_error=True, content="Unknown tool: X")` — LLM sees it, can adjust |
| Tool input fails schema validation | Same |
| Tool raises unhandled exception | Same, but log full traceback. Tool result contains a redacted message. |
| Tool times out (`tool_timeout_seconds`) | Cancel task, return timeout error to LLM |
| Sensitive tool: user says nothing after confirmation prompt | After `follow_up_ms`, pending_call is discarded silently |
| Sensitive tool: user asks a different question instead | Pending_call discarded; new turn proceeds normally |
| Multiple providers export same tool name | Startup fails loudly with clear message about conflict |
| Tavily API 5xx / rate limit | Tool returns is_error; LLM should apologize and offer to retry |
| Tavily returns zero results | Tool returns success with empty results; LLM decides what to say |

## Test strategy

- **Unit**: `@tool` decorator registers correctly; duplicate names fail loudly.
- **Unit**: schema validation rejects bad inputs.
- **Unit**: safety-tier flow — readonly executes, mutating executes+announces, sensitive returns awaiting_confirmation.
- **Unit**: filler-phrase rotation never repeats consecutively.
- **Integration**: full turn with `web_search` — LLM emits tool call, Tavily hit, results returned to LLM, spoken reply generated.
- **Integration**: sensitive-tier round-trip — LLM emits `forget`, user says "yes", memory tombstoned; second run where user says "actually no", memory untouched.
- **Integration**: parallel tools — two `readonly` calls execute concurrently and both results are in the follow-up turn.
- **Chaos**: kill Tavily connection mid-request, verify tool returns error and LLM handles gracefully.

## Open questions

- [OPEN] Tool cost/rate-limit accounting — Tavily has monthly credit limits. Track per-tool spend so we can warn user. v2.
- [OPEN] "Undo my last action" — for mutating tools, keep an undo stack in memory. Nice UX; not v1.
- [OPEN] Should the LLM be allowed to call `save_memory` inside a sensitive-tier confirmation prompt? Probably no — confirmation flow should be clean, no side effects until confirmed.
- [OPEN] Voice confirmation via wake-word replay ("hey Carl, yes") vs. just "yes" in the follow-up window. Design says follow-up window is enough; revisit if it feels wrong.
- [OPEN] Tool-result truncation — Tavily raw results can be large. Cap tool_result content to ~2KB before passing to Claude to save tokens. Do it.
- [OPEN] Dynamic tool loading — hot-reload for tool definitions like we have for prompts? Not urgent; restart is fine.
