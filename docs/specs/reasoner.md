# Reasoner Spec

## Purpose

The Reasoner is Carl's brain — the LLM that takes a final user transcript, retrieves memories, decides which tools to call, and streams a natural-language reply to the TTS stage. It's the biggest single component and the one where latency, cost, and quality trade against each other.

## Model choice

Default: **Claude Sonnet 4.6** (`claude-sonnet-4-6`). Rationale:
- Native tool-use support with streaming.
- Fast enough for voice UX (first-token latency ~300-500ms typical).
- Cheap enough for interactive use (~$3/M input, $15/M output).
- Excellent instruction-following for the "silent recall" and memory-usage discipline.

Alt (via config): **Haiku 4.5** for a "cheap mode" where we don't need deep reasoning. Not used by default.

**Not** for the reasoner: Opus 4.7. It's overkill for turn latency and cost. Opus is reserved for the nightly consolidation job.

## Interface

```python
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Awaitable

@dataclass
class ReasonerTurn:
    user_transcript: str
    conversation_id: str
    turn_id: str
    history: list[Message]              # last N turns from this conversation
    memories: list[Memory]              # from MemoryStore.retrieve()

@dataclass
class SpokenChunk:
    text: str                           # sentence-ish segment to hand to TTS
    is_final: bool                      # last chunk of this turn

class Reasoner(Protocol):
    async def turn(
        self,
        request: ReasonerTurn,
        on_tool_call: Callable[[ToolCall], Awaitable[ToolResult]],
        on_filler_needed: Callable[[str], Awaitable[None]],
    ) -> AsyncIterator[SpokenChunk]:
        """
        Run a full reasoning turn. Yields spoken chunks as they're generated.
        Handles the tool-use loop internally, calling on_tool_call for each
        tool. Emits filler speech via on_filler_needed when a tool call
        starts, so TTS can play "let me check" while tools run.
        """
        ...
```

The reasoner *owns* the tool loop. The pipeline hands it a request and callbacks; it drives the loop to completion.

## The turn loop

```python
async def turn(self, request, on_tool_call, on_filler_needed):
    messages = [
        *[msg for turn in request.history for msg in turn.as_messages()],
        {"role": "user", "content": request.user_transcript},
    ]

    system = self._build_system_prompt(request.memories)
    tools = self._active_tools()

    iterations = 0
    text_buffer = ""

    while iterations < self.config.max_iterations:
        iterations += 1

        # Stream the LLM response.
        async with self.client.messages.stream(
            model=self.config.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ) as stream:
            tool_calls: list[ToolUseBlock] = []
            async for event in stream:
                if isinstance(event, TextDelta):
                    text_buffer += event.text
                    # Emit sentence-boundary chunks to TTS as we go.
                    for sentence in extract_complete_sentences(text_buffer):
                        yield SpokenChunk(text=sentence, is_final=False)
                        text_buffer = text_buffer[len(sentence):]
                elif isinstance(event, ContentBlockStop):
                    if event.content_block.type == "tool_use":
                        tool_calls.append(event.content_block)
                        # Fire filler speech for the first tool call.
                        if len(tool_calls) == 1:
                            asyncio.create_task(
                                on_filler_needed(self._pick_filler(event.content_block.name))
                            )

            final_message = await stream.get_final_message()

        # No tools requested → we're done.
        if final_message.stop_reason == "end_turn":
            if text_buffer:
                yield SpokenChunk(text=text_buffer, is_final=True)
            self._log_turn_complete(final_message)
            return

        # Tools requested → execute them and loop.
        messages.append({"role": "assistant", "content": final_message.content})
        tool_results = await asyncio.gather(*[
            on_tool_call(ToolCall.from_block(tc)) for tc in tool_calls
        ])
        messages.append({
            "role": "user",
            "content": [tr.as_content_block() for tr in tool_results],
        })

    # Iteration cap hit — safety fallback.
    yield SpokenChunk(text="Sorry, that got too complicated. Let's try again.",
                      is_final=True)
```

## System prompt structure

The system prompt is built per turn with memories injected. Kept small and disciplined — long system prompts eat cache benefits and slow first-token latency.

```
{base_personality_prompt}       # from config/prompts/carl_base.md, hot-reloaded

<memories>
{formatted_memory_block}         # see memory.md — grouped by type
</memories>

# Behavior rules
- Silent recall: never say "I remember...", just use the memory naturally.
- Voice UX: reply in complete, spoken sentences. Keep answers short unless
  asked for detail. Sentences are what get streamed to TTS.
- Tools: prefer tools over guessing for facts that could be wrong (current
  events, prices, weather). Use save_memory when the user tells you
  something durable.

# Current context
- Current time: {now}
- User: {user_id}
- Conversation ID: {conversation_id} (turn {turn_index})
```

`{base_personality_prompt}` example (`config/prompts/carl_base.md`):

```
You are Carl, a personal voice assistant. You are direct, warm, and dry
in the way a competent friend is — never obsequious, never chirpy. You
value the user's time. You don't hedge. If you don't know, say so and
offer to look it up.

Your voice is spoken, not written: no bullet lists, no headings, no
markdown, no emoji. One idea per sentence. Complete sentences.
```

## Streaming to TTS — the sentence-boundary trick

Naive: buffer the whole LLM response, then hand to TTS. Adds 500-2000ms of dead air.

Better: as LLM streams, buffer text until we see a sentence-ending boundary (`. ` or `? ` or `! ` followed by capital or newline), then flush that sentence to TTS. TTS starts speaking sentence 1 while LLM is still generating sentence 2.

```python
_SENTENCE_END = re.compile(r'([.!?])\s+(?=[A-Z])|([.!?])$|\n{2,}')

def extract_complete_sentences(buffer: str) -> list[str]:
    """Yield complete sentences; leaves incomplete tail in buffer."""
    sentences = []
    while match := _SENTENCE_END.search(buffer):
        end = match.end()
        sentences.append(buffer[:end])
        buffer = buffer[end:]
    return sentences  # caller reconstructs remaining buffer
```

Time-to-first-audio budget:
- LLM first token: ~400ms
- First sentence completed: +200-500ms (~10-15 tokens)
- TTS first byte: ~340ms (per TTS research brief)
- **Total: ~1-1.2s from user finishing to Carl starting.** Meets the "feels responsive" bar.

## Filler speech during tool execution

When the reasoner emits a `tool_use` block, the pipeline is about to wait for the tool to execute. Rather than dead air:

```python
FILLER_PHRASES = {
    "web_search":    ["Let me check.", "One sec.", "Looking that up."],
    "get_weather":   ["Checking the weather.", "One moment."],
    "set_timer":     ["Got it.", "Setting that up."],
    "save_memory":   ["Noted.", "Got it."],
    "default":       ["Hold on.", "One sec.", "Working on it."],
}

def pick_filler(tool_name: str) -> str:
    options = FILLER_PHRASES.get(tool_name, FILLER_PHRASES["default"])
    return random.choice(options)
```

Fires asynchronously — TTS speaks filler while tool executes. Filler and tool-result-driven speech may briefly overlap in the playback queue; that's fine, TTS is streaming into the same output.

Stored in `config/prompts/filler_phrases.yaml` for easy iteration.

## Tool-loop safety cap

`max_iterations: 5` in config. If the LLM keeps requesting tools past that, something has gone wrong (loop, confused planning). Bail with a graceful fallback message.

## Prompt caching

The system prompt (base personality + memory block + behavior rules + tool defs) is largely stable across turns within a conversation. Use Anthropic's prompt caching to avoid re-billing for it every turn:

```python
system = [
    {"type": "text", "text": base_personality, "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": memory_block},   # changes turn-to-turn; not cached
    {"type": "text", "text": behavior_rules, "cache_control": {"type": "ephemeral"}},
]

tools = [
    # Tool defs are stable → cache
    *tool_definitions,
]
# Anthropic SDK: add cache_control to the *last* item you want cached; everything
# up to that point becomes part of the cache prefix.
```

Cache hit rate should be ~90%+ within a single conversation. Meaningful cost drop for anything multi-turn.

## History window

Config `conversation.history_turns: 10`. Last 10 turns of (user, assistant) pairs are included in `messages`.

Older turns are *not* automatically summarized or dropped further — if the conversation goes beyond that, the reasoner just doesn't see them. Rationale: at 10 turns, most conversations are already at their natural end (Carl isn't ChatGPT; it's a voice interlocutor). If we ever see long conversations, add summarization.

Cross-conversation "history" is exactly what the memory system is for. Don't leak conversation-1 turns into conversation-2's `messages`; that's what memories are.

## Assistant text truncation on barge-in

When the user interrupts (see `conversation-state.md`), the reasoner may still be streaming. The pipeline calls `reasoner.abort()`, which cancels the streaming context. The `history` for the next turn records what was *actually spoken*, not what the LLM had produced. Estimation: `played_bytes / 24000 / 2 → played_seconds → played_chars @ ~15 chars/sec`.

The reasoner exposes:

```python
async def abort(self, played_bytes: int) -> str:
    """Cancel current stream. Return the truncated assistant_text
    that reflects what the user actually heard."""
```

## Failure modes

| Failure | Behavior |
|---|---|
| Anthropic API 429 (rate limit) | SDK retries with backoff (built-in). If sustained, surface a spoken fallback. |
| Anthropic API 5xx | Retry once, then spoken fallback. Log `provider_error`. |
| Network timeout | `llm_timeout_seconds` fires from pipeline; reasoner is aborted; fallback text speaks. |
| Streaming stalls mid-response | Watchdog: if no tokens for 5s, abort and speak what we have. |
| Tool call requires unknown tool name | Tool result: `{"is_error": true, "content": "tool 'X' not registered"}`. LLM sees it and adjusts. |
| Tool execution raises | Same — wrapped as `is_error: true` result. |
| Iteration cap hit | Fallback spoken text; log `llm_error` with `reason: iteration_cap`. |
| LLM produces markdown/lists (violates voice-UX rule) | Post-process: strip markdown before TTS. Log so we can iterate the prompt. |

## Test strategy

- **Unit**: `extract_complete_sentences` on adversarial inputs (URLs with dots, ellipses, abbreviations like "Dr.").
- **Unit**: system-prompt builder produces cache-friendly structure (base + rules cached; memories not).
- **Unit**: iteration cap enforced.
- **Unit**: filler phrases rotate; never same one twice in a row.
- **Integration (with mock Anthropic)**: full turn with tool call → filler emitted → tool executed → final text streamed.
- **Integration**: abort mid-stream returns truncated text estimate within 20% of actual played chars.
- **Integration**: prompt-cache metrics — after 3 turns in a conversation, cache read tokens should be > 80% of system+tools tokens.
- **Property**: LLM output is always spoken-form (no markdown makes it past post-processor).
- **Golden**: fixture conversations produce expected reasoner behavior against a recorded/stubbed API (VCR-style).

## Open questions

- [OPEN] Structured output for reasoning steps ("I'm thinking about X") — do we ever surface that to the user? Probably not, but debug tooling could show it.
- [OPEN] Interruption during tool execution — user interrupts while a web search is in flight. Do we cancel the tool call? Probably yes; add a cancel token to `on_tool_call`.
- [OPEN] Turn-level cost accounting — emit per-turn token/cost to event log for tracking. Small addition; do it.
- [OPEN] Fallback model chain (Sonnet → Haiku on 5xx) — nice for resilience. v2.
- [OPEN] Ambient re-planning — if the tool result changes the answer substantially, should the LLM be allowed to interrupt its own spoken output? Very "AI assistant of the future," but complicated. Not now.
