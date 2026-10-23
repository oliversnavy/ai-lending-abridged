"""LangChain callback handler that sums usage_metadata across every model call in an
episode, including calls made inside deepagents subagents.

Subagent invocations are opaque to the main graph's own message stream: deepagents'
task()/atask() tools (deepagents/middleware/subagents.py) discard the subagent's
internal messages, keeping only a single synthesized ToolMessage — so walking the main
graph's own final message list (the previous approach) undercounts any episode that used
subagent delegation, worse the more an episode delegates. LangChain callbacks are
attached differently: they propagate through langgraph's config merging regardless of
graph nesting (deepagents/middleware/subagents.py's own comment: "the parent's
callbacks... reach the subagent automatically"), so a handler attached once at the
top-level graph invocation sees every model call the same way, whether it came from the
main agent or a delegated subagent.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenCountingCallbackHandler(BaseCallbackHandler):
    """Accumulates total_tokens from every on_llm_end event it observes."""

    def __init__(self) -> None:
        self.total_tokens = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) if message is not None else None
                if usage:
                    self.total_tokens += usage.get("total_tokens", 0) or 0
