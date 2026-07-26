"""Web search tool backed by Tavily. See docs/specs/tools.md.

Tavily is designed for LLM tool-use: it returns cleaned snippets + an optional
pre-synthesized answer, which lets Claude respond with grounded facts without
having to read raw HTML.

The `description` field is the load-bearing prompt-engineering surface — it's
what Claude uses to decide when to call this tool. Iterate on it based on
real observed behavior in the event log (see docs/BACKLOG.md for the
tool-description iteration item).
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

from carl.config.schema import TavilyConfig
from carl.tools import ToolResult, tool

log = structlog.get_logger()

_RESULT_BUDGET_BYTES = 2000    # cap the tool_result content going back to Claude
_ANSWER_TRUNCATE_TO = 1200     # cap the pre-synthesized answer if it alone exceeds budget


@tool(
    name="web_search",
    description=(
        "Search the web for current information. Use when the user asks about "
        "things you might not know or that could have changed recently: current "
        "events, weather, prices, sports scores, recent news, or anything "
        "time-sensitive. Do NOT use for basic knowledge you already have "
        "(historical facts, famous people, math, definitions) — answer those "
        "directly. Also do not use for personal questions about the user; "
        "check memory instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query in natural language.",
            },
        },
        "required": ["query"],
    },
    safety_tier="readonly",
)
class WebSearchTool:
    def __init__(self, tavily_config: TavilyConfig | None = None):
        # Config comes via the NativeToolProvider's `injected` dict — main.py
        # populates {"tavily_config": config.search.tavily}. Fallback to
        # defaults if nothing was injected.
        self._config = tavily_config or TavilyConfig()
        # Client is constructed lazily on first execute() so `carl tools list`
        # (a read-only inspection command) works even when TAVILY_API_KEY is
        # unset — critical for a fresh clone where the dev hasn't signed up
        # for Tavily yet.
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from tavily import AsyncTavilyClient  # lazy — httpx transport cost

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set in environment. Add it to .env — "
                "see .env.example."
            )
        self._client = AsyncTavilyClient(api_key=api_key)

    async def close(self) -> None:
        """Called on Pipeline shutdown to release the httpx pool. No-op if
        the client was never lazy-constructed."""
        if self._client is None:
            return
        aclose = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if aclose is not None:
            try:
                result = aclose()
                if hasattr(result, "__await__"):
                    await result
            except Exception:  # noqa: BLE001
                log.exception("tavily_close_failed")

    async def execute(self, query: str) -> ToolResult:
        log.info("web_search_query", query=query[:80],
                 search_depth=self._config.search_depth)
        try:
            self._ensure_client()
            r = await self._client.search(
                query=query,
                search_depth=self._config.search_depth,
                max_results=self._config.max_results,
                include_answer=self._config.include_answer,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("web_search_failed")
            # Plain-text error so Claude can speak it naturally instead of
            # trying to render JSON braces in a spoken reply.
            return ToolResult(
                content=f"Web search failed: {exc!s}",
                is_error=True,
            )

        # Compact payload for Claude: pre-synthesized answer + top hits with
        # snippets. Drop raw_content unless we ever need it.
        answer = r.get("answer")
        payload: dict[str, Any] = {
            "answer": answer,
            "results": [
                {
                    "title": x.get("title", ""),
                    "url": x.get("url", ""),
                    "snippet": (x.get("content") or "")[:400],
                }
                for x in r.get("results", [])
            ],
        }
        content = json.dumps(payload, ensure_ascii=False)

        # If we blew the budget: first truncate the answer (which can be
        # multi-KB on advanced search_depth), then trim results one at a time.
        oversize = len(content.encode("utf-8")) > _RESULT_BUDGET_BYTES
        if oversize and answer and len(answer) > _ANSWER_TRUNCATE_TO:
            payload["answer"] = answer[:_ANSWER_TRUNCATE_TO].rstrip() + "…"
            content = json.dumps(payload, ensure_ascii=False)
        while len(content.encode("utf-8")) > _RESULT_BUDGET_BYTES and payload["results"]:
            payload["results"].pop()
            content = json.dumps(payload, ensure_ascii=False)

        log.info("web_search_ok",
                 result_count=len(payload["results"]),
                 has_answer=payload["answer"] is not None,
                 bytes=len(content))
        return ToolResult(content=content, metadata={"provider": "tavily"})
