"""Claude reasoner stage. See docs/specs/reasoner.md.

Milestone 2 scope: streaming text generation only. Tool-use loop lands in
Milestone 3; the interface accepts on_tool_call callbacks now but this
implementation doesn't emit any tool_use blocks (empty tools list).

Streaming pattern uses AsyncAnthropic's messages.stream() context manager.
Text deltas are buffered per sentence-boundary (see reasoner.md sentence
rule) and yielded to the pipeline for TTS.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import structlog

from carl.config.schema import ClaudeReasonerConfig
from carl.stages.reasoner.base import (
    OnFillerNeededCallback,
    OnToolCallCallback,
    ReasonerTurn,
    SpokenChunk,
    ToolCall,
)

log = structlog.get_logger()

# Sentence-boundary detector: end-of-sentence punctuation followed by
# whitespace + capital letter, OR at end of string, OR double newline.
_SENTENCE_END = re.compile(r"([.!?])(\s+)(?=[A-Z])|([.!?])$")


def _extract_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remaining_buffer)."""
    sentences: list[str] = []
    remaining = buffer
    while True:
        m = _SENTENCE_END.search(remaining)
        if m is None:
            break
        end = m.end()
        sentences.append(remaining[:end])
        remaining = remaining[end:]
    return sentences, remaining


class ClaudeReasoner:
    def __init__(self, config: ClaudeReasonerConfig):
        # Lazy import — anthropic SDK is heavy on first import.
        from anthropic import AsyncAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set in environment. "
                "Add it to .env — see .env.example."
            )
        self._client = AsyncAnthropic(api_key=api_key)
        self._config = config

        # Load system prompt from file if configured, else fall back to a
        # minimal default so first-run works even without the prompt file.
        self._base_prompt = self._load_system_prompt()

        self._current_stream = None
        self._aborted = False
        # Filler-phrase state, loaded lazily on first tool call.
        self._filler_phrases: dict[str, list[str]] | None = None
        self._last_filler: dict[str, str] = {}

    def _load_system_prompt(self) -> str:
        if self._config.system_prompt_path is None:
            return (
                "You are Carl, a personal voice assistant. Direct, warm, dry. "
                "Voice-only output: no lists, no markdown, one idea per sentence."
            )
        path = Path(self._config.system_prompt_path)
        if not path.exists():
            log.warning("system_prompt_missing", path=str(path))
            return "You are Carl, a voice assistant."
        return path.read_text().strip()

    async def turn(
        self,
        request: ReasonerTurn,
        on_tool_call: OnToolCallCallback | None = None,
        on_filler_needed: OnFillerNeededCallback | None = None,
    ) -> AsyncIterator[SpokenChunk]:
        """Stream a Claude response, yielding sentence-boundary chunks.

        Handles the tool-use loop when `request.tools` is non-empty and Claude
        emits tool_use content blocks: fires `on_filler_needed` once per turn
        as the first tool starts, executes tools via `on_tool_call` in
        parallel, feeds results back, and iterates. Caps at max_iterations to
        prevent runaway planning loops.
        """
        self._aborted = False
        messages = [
            *self._format_history(request.history),
            {"role": "user", "content": request.user_transcript},
        ]
        tools = request.tools or []
        filler_fired_this_turn = False
        max_iterations = 5   # Safety cap; also on backlog to make configurable.

        try:
            for _iteration in range(max_iterations):
                if self._aborted:
                    return
                buffer = ""
                async with self._client.messages.stream(
                    model=self._config.model,
                    system=self._base_prompt,
                    messages=messages,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    tools=tools if tools else None,
                ) as stream:
                    self._current_stream = stream

                    # Walk the raw event stream so we can detect tool_use
                    # content blocks as they start (for filler speech) and
                    # stream text_deltas with sentence-boundary chunking.
                    async for event in stream:
                        if self._aborted:
                            break
                        et = getattr(event, "type", None)
                        if et == "content_block_start" and on_filler_needed \
                                and not filler_fired_this_turn:
                            block = getattr(event, "content_block", None)
                            if block is not None and getattr(block, "type", None) == "tool_use":
                                phrase = self._pick_filler(getattr(block, "name", ""))
                                asyncio.create_task(on_filler_needed(phrase))
                                filler_fired_this_turn = True
                        elif et == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is not None and getattr(delta, "type", None) == "text_delta":
                                buffer += getattr(delta, "text", "") or ""
                                sentences, buffer = _extract_complete_sentences(buffer)
                                for s in sentences:
                                    yield SpokenChunk(text=s, is_final=False)

                    # Flush trailing text (e.g. "Sure.") from this iteration.
                    if buffer and not self._aborted:
                        yield SpokenChunk(text=buffer, is_final=False)

                    final_msg = await stream.get_final_message()

                # Not tool_use → we're done.
                if getattr(final_msg, "stop_reason", None) != "tool_use":
                    return

                # Tools requested: execute all tool_use blocks in parallel,
                # feed results back, iterate.
                if on_tool_call is None:
                    log.warning("tool_use_but_no_callback — cannot execute tools")
                    return

                tool_uses = [
                    b for b in final_msg.content
                    if getattr(b, "type", None) == "tool_use"
                ]
                results = await asyncio.gather(*[
                    on_tool_call(ToolCall(id=b.id, name=b.name, input=b.input))
                    for b in tool_uses
                ])

                # Append assistant's tool-use message + user's tool_result message.
                messages.append({
                    "role": "assistant",
                    "content": [b.model_dump() for b in final_msg.content],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.tool_use_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        }
                        for r in results
                    ],
                })
                # Loop back for next iteration — Claude sees the tool results
                # and either responds with text (end_turn) or calls more tools.

            log.warning("tool_use_loop_hit_max_iterations", cap=max_iterations)
            yield SpokenChunk(text="Sorry, that got too complicated. Try again?",
                              is_final=True)
        finally:
            self._current_stream = None

    def _pick_filler(self, tool_name: str) -> str:
        """Load filler phrases from config/prompts/filler_phrases.yaml on first
        use, rotate through them so we never repeat back-to-back per tool."""
        if self._filler_phrases is None:
            self._load_filler_phrases()
        options = self._filler_phrases.get(tool_name) or self._filler_phrases.get("default")
        if not options:
            return "One moment."
        # Rotate: pick anything != last picked for this tool.
        last = self._last_filler.get(tool_name)
        pool = [p for p in options if p != last] if len(options) > 1 else options
        import random
        choice = random.choice(pool)
        self._last_filler[tool_name] = choice
        return choice

    def _load_filler_phrases(self) -> None:
        """Load config/prompts/filler_phrases.yaml. Tolerates missing file
        with a minimal built-in fallback."""
        import yaml
        path = Path("config/prompts/filler_phrases.yaml")
        if not path.exists():
            self._filler_phrases = {"default": ["One moment.", "Hold on."]}
            return
        try:
            self._filler_phrases = yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001
            log.exception("filler_phrases_load_failed", path=str(path))
            self._filler_phrases = {"default": ["One moment."]}

    def _format_history(self, history: list) -> list[dict[str, Any]]:
        """Convert Message dataclass list to Anthropic messages format.
        Drop the very first message if it's an assistant (Anthropic requires
        conversations start with user)."""
        formatted = []
        for msg in history:
            if not formatted and msg.role == "assistant":
                continue
            formatted.append({"role": msg.role, "content": msg.content})
        return formatted

    async def abort(self) -> None:
        """Cancel any in-flight streaming turn. Sets a flag; the async-for
        loop above checks it after each chunk."""
        self._aborted = True
        # Note: Anthropic SDK's stream doesn't have an explicit .cancel(),
        # but exiting the `async with` (which happens when the generator
        # is aborted/closed) closes the underlying HTTP connection.
