# Milestone 3 — Web Search Tool

## Goal

Carl can answer questions that require current information ("what's the weather in Chicago", "who won the game last night") by calling Tavily via the tool registry. This is the first real exercise of the tool-orchestration loop from `reasoner.md`, including filler speech during tool execution.

## Prerequisites

- Milestone 2 complete (basic voice loop works).
- `TAVILY_API_KEY` in `.env`.
- Specs read: `tools.md`, `reasoner.md` (tool loop section).

## Non-goals

- Memory (Milestone 4).
- Barge-in (Milestone 5).
- Additional tools beyond `web_search` — timer/weather/smart-home wait.
- MCP integration.

## What changes vs Milestone 2

1. Tool registry gets a real tool.
2. Reasoner's turn loop upgrades from "just stream text" to "handle tool_use blocks in the streaming response."
3. Pipeline coordinates filler speech during tool execution.

## Files to create/modify

### New: `carl/tools/web_search.py`

Per `tools.md`, using Tavily's Python SDK:

```python
import json, os
from tavily import AsyncTavilyClient
from carl.tools import tool, ToolResult

@tool(
    name="web_search",
    description=(
        "Search the web for current information. Use when the user asks "
        "about facts you might not know or that could have changed "
        "recently: current events, weather, prices, sports scores, "
        "recent news, or anything time-sensitive. Do NOT use for basic "
        "knowledge you already have (historical facts, famous people, "
        "math, definitions)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query in natural language."},
        },
        "required": ["query"],
    },
    safety_tier="readonly",
)
class WebSearchTool:
    def __init__(self, config):
        self.client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.config = config

    async def execute(self, query: str) -> ToolResult:
        r = await self.client.search(
            query=query,
            search_depth=self.config.search_depth,
            max_results=self.config.max_results,
            include_answer=self.config.include_answer,
        )
        payload = {
            "answer": r.get("answer"),
            "results": [
                {"title": x["title"], "url": x["url"], "snippet": x["content"]}
                for x in r.get("results", [])
            ],
        }
        # Cap to ~2KB per tools.md open question
        content = json.dumps(payload)
        if len(content) > 2000:
            payload["results"] = payload["results"][:3]
            content = json.dumps(payload)
        return ToolResult(content=content, metadata={"provider": "tavily"})
```

Config toggle: `providers.search: tavily` (was `mock`).

### Modify: `carl/stages/reasoner/claude.py`

Extend the turn method to handle tool-use per the pattern in `reasoner.md`:

```python
async def turn(self, request, on_tool_call, on_filler_needed):
    messages = self._build_messages(request)
    system = self._build_system_prompt(request.memories)
    tool_defs = self._tool_definitions()

    for iteration in range(self.config.max_iterations):
        text_buffer = ""
        tool_uses = []
        filler_fired = False

        async with self.client.messages.stream(
            model=self.config.model,
            system=system,
            messages=messages,
            tools=tool_defs,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        text_buffer += event.delta.text
                        for s in extract_complete_sentences(text_buffer):
                            yield SpokenChunk(text=s, is_final=False)
                            text_buffer = text_buffer[len(s):]
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use" and not filler_fired:
                        asyncio.create_task(
                            on_filler_needed(self._pick_filler(event.content_block.name))
                        )
                        filler_fired = True

            final_msg = await stream.get_final_message()

        if final_msg.stop_reason == "end_turn":
            if text_buffer:
                yield SpokenChunk(text=text_buffer, is_final=True)
            return

        # Extract tool_use blocks, execute in parallel, feed back
        tool_uses = [b for b in final_msg.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": final_msg.content})
        results = await asyncio.gather(*[
            on_tool_call(ToolCall.from_block(tc)) for tc in tool_uses
        ])
        messages.append({
            "role": "user",
            "content": [self._to_result_block(tc, r) for tc, r in zip(tool_uses, results)],
        })

    yield SpokenChunk(text="Sorry, that got too complicated. Let's try again.", is_final=True)
```

The `_to_result_block` helper produces:

```python
{"type": "tool_result", "tool_use_id": tc.id, "content": r.content, "is_error": r.is_error}
```

### Modify: `carl/pipeline/orchestrator.py`

The pipeline is what actually calls `tool_registry.execute()` when the reasoner requests. Wire `on_tool_call` and `on_filler_needed`:

```python
async def _run_reasoner_turn(self, transcript):
    async def on_tool_call(call: ToolCall) -> ToolResult:
        self.event_log.log("reasoner", "llm_tool_use",
                           payload={"tool_name": call.name, "tool_input": call.input, ...},
                           conversation_id=self._conv_id, turn_id=self._turn_id)
        start = time.monotonic()
        result = await self.tool_registry.execute(call.name, call.input)
        latency = (time.monotonic() - start) * 1000
        self.event_log.log("reasoner", "llm_tool_result",
                           payload={"tool_use_id": call.id, "is_error": result.is_error,
                                    "latency_ms": latency, ...},
                           conversation_id=self._conv_id, turn_id=self._turn_id)
        return result

    async def on_filler_needed(phrase: str):
        # Stream filler through TTS immediately
        cancel = asyncio.Event()
        await self._speak(phrase, cancel)

    async for chunk in self.stages.reasoner.turn(request,
                                                  on_tool_call=on_tool_call,
                                                  on_filler_needed=on_filler_needed):
        await self._speak(chunk.text, cancel_event=self._tts_cancel)
```

### Modify: `config/prompts/filler_phrases.yaml`

```yaml
web_search:
  - Let me check.
  - One sec.
  - Looking that up.
default:
  - Hold on.
  - One sec.
```

### Modify: `carl/stages/reasoner/claude.py` — filler rotation

Track last-picked-per-tool in the reasoner instance so we never repeat consecutively:

```python
def _pick_filler(self, tool_name: str) -> str:
    options = FILLER_PHRASES.get(tool_name, FILLER_PHRASES["default"])
    last = self._last_filler.get(tool_name)
    choice = random.choice([o for o in options if o != last]) if len(options) > 1 else options[0]
    self._last_filler[tool_name] = choice
    return choice
```

## Pseudocode: turn with a tool call

```
User: "hey Carl, what's the weather in Chicago right now"
├─ (wake, STT, transcript ready)
├─ Reasoner streams:
│    stream event: text_delta "" (no text yet)
│    stream event: content_block_start (tool_use: web_search)
│    ├─ filler_needed callback fires → "Let me check."
│    ├─ TTS speaks "Let me check." (starts within ~800ms of speech_end)
│    stream event: content_block_stop (tool_use complete)
│    final message: stop_reason=tool_use, tool_use_id=abc, input={"query": "weather in Chicago right now"}
├─ pipeline: on_tool_call → registry.execute("web_search", {"query": ...})
│    ├─ Tavily API call (2-3s typical)
│    ├─ Returns compact JSON with answer + top 3 results
├─ Second reasoner iteration: messages now include assistant tool_use + user tool_result
│    stream event: text_delta "It's currently 32 degrees and lightly snowing in Chicago."
│    ├─ Sentence-boundary emit: SpokenChunk(is_final=False)
│    ├─ TTS speaks (may overlap tail of "Let me check.")
│    final message: stop_reason=end_turn
├─ FSM: SPEAKING → RECORDING (follow-up)
```

## Definition of done

- [ ] `web_search` tool registered and discoverable via `carl tools list` CLI subcommand (add this — one line for debug).
- [ ] Reasoner's tool loop handles tool_use responses (unit test with mocked Anthropic).
- [ ] Filler speech fires for tool calls; never repeats same phrase consecutively.
- [ ] "hey Carl, what's the weather in Chicago" produces a spoken answer with a Tavily hit visible in the event log.
- [ ] Event log for such a turn contains: `stt_final`, `llm_request`, `llm_tool_use`, `llm_tool_result`, `tts_first_byte` (for filler), `llm_response`, `tts_first_byte` (for real reply), `tts_complete`.
- [ ] Tool-result size cap works (verify with a huge query — payload never exceeds ~2KB).
- [ ] Tool timeout (`tool_timeout_seconds`) causes graceful failure — LLM sees error, apologizes.
- [ ] `pytest tests/test_tool_registry.py tests/test_reasoner_tool_loop.py` passes.

## Acceptance test (manual)

*"hey Carl, what's the weather in Chicago right now"*

Response:
1. ~800ms after you stop: hear "Let me check." (nova voice)
2. ~2-3s later: hear "It's currently 32 degrees and lightly snowing in Chicago." (or similar, per current weather)
3. Event log shows the full trace.

Then:

*"hey Carl, when was Chicago founded"*

Response:
1. Straight to answer within ~1.5s ("Chicago was founded in 1833..."). No filler, no tool call — the LLM knows this without searching. This is the "prefer knowledge over search" behavior working correctly.

If the LLM tools-up for known historical facts, tighten the tool description with a "do NOT use for" clause.

## Traps / lessons

- **Filler cadence**: if the tool completes in <300ms, you may end up with filler-then-immediately-answer sounding weird. Add a `tool_timeout_before_filler=250ms` — wait 250ms before firing filler; if tool completes first, skip filler entirely.
- **Streaming + tool_use event order**: Anthropic streams `content_block_start` for a tool_use block, then `content_block_delta` events with `input_json_delta` (accumulating the tool input JSON), then `content_block_stop`. Use `stream.get_final_message()` for the assembled tool_use blocks; don't try to parse the incremental deltas.
- **Parallel tool calls**: Claude may emit multiple tool_use blocks in one response. `asyncio.gather` them. Filler fires once for the batch, not once per tool.
- **Tool description iteration**: the "do NOT use for basic knowledge" clause is load-bearing. Watch a week of tool-use logs; if the LLM over-tools, refine the description.
- **Tavily rate limit on free tier is ~1 RPS** [VERIFY] — voice UX serializes tool calls per turn so this isn't a hot issue, but a test suite hammering it will 429. Rate-limit the test runner.
