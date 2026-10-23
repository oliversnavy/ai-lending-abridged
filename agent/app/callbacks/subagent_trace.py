"""LangChain callback handler that captures full trace content for every LLM and tool
call in an episode, including calls made inside deepagents subagents.

TokenCountingCallbackHandler (token_counter.py) already proved that callbacks propagate
through subagent invocations regardless of graph nesting -- this reuses the same
propagation guarantee to persist actual call content, not just a running token total.

Attribution note: deepagents tags subagent runs via `ls_agent_type="subagent"` through
LangSmith's tracing_context() mechanism (langsmith.run_helpers) -- verified directly that
this does NOT reach a plain BaseCallbackHandler's `metadata`/`tags` kwargs at all (it's
LangSmith-backend-only plumbing). So attribution here is done the reliable way: every
call's run_id/parent_run_id is recorded, and a post-hoc walk (see attribute_to_delegations
below) reconstructs which calls are descendants of which `task`/`atask` tool invocation,
purely from the run tree LangChain already builds internally.

Verified directly (see scratchpad probe scripts from the design session) that a leaf
event's parent_run_id does NOT point directly at the owning task()'s tool_start run_id --
LangGraph interposes intermediate "chain" runs (one per internal Runnable/graph-node
invocation) between them. Walking parent_run_id from a leaf event straight up therefore
hits a dead end (a parent_run_id that's never itself the run_id of any LLM/tool event) one
hop too early. Fix: also hook on_chain_start/on_chain_end to build a COMPLETE run_id ->
parent_run_id map spanning every intermediate hop, kept separate from `events` (chain runs
carry no content worth persisting -- they're scaffolding for the walk, not trace data).
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

_DELEGATION_TOOL_NAMES = ("task", "atask")


class SubagentTraceCallbackHandler(BaseCallbackHandler):
    """Accumulates every chat-model and tool call in an episode as flat, timestamped
    events carrying run_id/parent_run_id, plus a separate run_tree map (including
    intermediate chain-level runs) needed to walk from any leaf event up to its owning
    task() delegation. Call attribute_to_delegations(handler.events, handler.run_tree)
    after the episode finishes to group events by delegation.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        # run_id -> parent_run_id, covering EVERY observed run (chain/llm/tool) -- more
        # complete than what `events` alone would give, since chain runs aren't persisted
        # as trace content but are required to walk the full parent chain (see module
        # docstring for why leaf-to-leaf parent_run_id alone has gaps).
        self.run_tree: dict[str, str | None] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.run_tree[str(run_id)] = str(parent_run_id) if parent_run_id else None

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.run_tree[str(run_id)] = str(parent_run_id) if parent_run_id else None
        flat = [m for batch in messages for m in batch]
        self.events.append({
            "kind": "llm_start",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags or [],
            # Only the newest message is the actual turn input; the rest is history
            # already captured by earlier events (or, for the very first call in a
            # subagent, the task() description -- captured separately as tool input).
            "content": str(flat[-1].content) if flat else "",
        })

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.run_tree[str(run_id)] = str(parent_run_id) if parent_run_id else None
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                entry: dict[str, Any] = {
                    "kind": "llm_end",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "tags": tags or [],
                    "content": str(getattr(message, "content", "")),
                }
                tool_calls = getattr(message, "tool_calls", None) if isinstance(message, AIMessage) else None
                if tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                        for tc in tool_calls
                    ]
                self.events.append(entry)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.run_tree[str(run_id)] = str(parent_run_id) if parent_run_id else None
        self.events.append({
            "kind": "tool_start",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags or [],
            "name": serialized.get("name"),
            "input": inputs if inputs is not None else input_str,
        })

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.run_tree[str(run_id)] = str(parent_run_id) if parent_run_id else None
        self.events.append({
            "kind": "tool_end",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags or [],
            "output": str(output),
        })


def attribute_to_delegations(
    events: list[dict[str, Any]],
    run_tree: dict[str, str | None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Group events by which task()/atask() delegation they occurred under.

    Returns {delegation_run_id: [events...]} for every `task`/`atask` tool_start found,
    plus a "main" bucket for everything that isn't a descendant of any delegation. A
    delegation's own tool_start/tool_end events (the task() call itself, as seen by the
    main graph) are included in ITS OWN bucket, not "main" -- they're the entry/exit
    point of that subagent invocation.

    run_tree should be SubagentTraceCallbackHandler.run_tree (includes intermediate
    chain-level runs) -- a leaf event's own parent_run_id is NOT sufficient on its own,
    since LangGraph interposes chain runs between a leaf call and its true delegation
    ancestor (verified directly; see module docstring). Falls back to deriving parent_of
    from `events` alone if run_tree isn't provided, but that walk will underattribute
    (dead-ends one hop early into "main") for anything nested under a subagent.
    """
    parent_of: dict[str, str | None] = (
        dict(run_tree) if run_tree is not None
        else {e["run_id"]: e["parent_run_id"] for e in events}
    )
    delegation_run_ids = {
        e["run_id"] for e in events
        if e["kind"] == "tool_start" and e.get("name") in _DELEGATION_TOOL_NAMES
    }

    def _owning_delegation(run_id: str) -> str | None:
        """Walk parent_run_id up until hitting a delegation run_id or the root."""
        seen: set[str] = set()
        current = run_id
        while current is not None and current not in seen:
            if current in delegation_run_ids:
                return current
            seen.add(current)
            current = parent_of.get(current)
        return None

    grouped: dict[str, list[dict[str, Any]]] = {rid: [] for rid in delegation_run_ids}
    grouped["main"] = []

    for e in events:
        rid = e["run_id"]
        owner = rid if rid in delegation_run_ids else _owning_delegation(rid)
        grouped[owner if owner is not None else "main"].append(e)

    return grouped
