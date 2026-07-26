"""Claude reasoner stage. See docs/specs/reasoner.md.

Milestone 2 scope: streaming text generation only. Tool-use loop lands in
Milestone 3; the interface accepts on_tool_call callbacks now but this
implementation doesn't emit any tool_use blocks (empty tools list).

Streaming pattern uses AsyncAnthropic's messages.stream() context manager.
Text deltas are buffered per sentence-boundary (see reasoner.md sentence
rule) and yielded to the pipeline for TTS.
"""

from __future__ import annotations

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
        """Stream a Claude response, yielding sentence-boundary chunks."""
        self._aborted = False

        # Milestone 2: no memories injected yet (that's M4), no tools (M3).
        messages = [
            *self._format_history(request.history),
            {"role": "user", "content": request.user_transcript},
        ]

        buffer = ""
        try:
            async with self._client.messages.stream(
                model=self._config.model,
                system=self._base_prompt,
                messages=messages,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            ) as stream:
                self._current_stream = stream
                async for text in stream.text_stream:
                    if self._aborted:
                        break
                    buffer += text
                    sentences, buffer = _extract_complete_sentences(buffer)
                    for s in sentences:
                        yield SpokenChunk(text=s, is_final=False)
                # Flush any trailing partial (e.g. "Sure.") as final chunk.
                if buffer and not self._aborted:
                    yield SpokenChunk(text=buffer, is_final=True)
        finally:
            self._current_stream = None

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
