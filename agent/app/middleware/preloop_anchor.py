"""Pre-loop strategy anchor -- the "starting critic" in this harness's three-critic
lifecycle (see also preloop_anchor's siblings: periodic_critic.py mid-episode,
post_mortem.py end-of-episode).

Motivation: the source research project found a real ADHERENCE gap, not a content or
access problem -- agents given specific, actionable prior-approach text (via the
ALL_PRIOR loop-injection mechanism, which every loop iteration already receives)
consistently did not act on it. Trace inspection across many loop iterations found the
agent re-deriving its own model/feature/formula from scratch nearly every time, never
attempting to build on or approximate the best prior result already sitting in its own
injected context, even when that prior result was concrete (exact formula, exact
coefficients) rather than vague.

This middleware does NOT change what information is available (that was never the
problem) -- it changes how prominently the single best prior approach is surfaced, once,
before the agent's own first turn, using the same bounded-single-LLM-call pattern as
periodic_critic.py's PeriodicCriticMiddleware (a lightweight cousin of it, not a copy --
this fires exactly ONCE per episode, unconditionally on the first before_model call, never
again, and only if there is any learnings text to anchor on at all).

Deliberately NOT paired with any instruction to fall back to the anchored approach if the
agent's own attempt doesn't beat it -- an earlier version of this idea tried exactly that
and was rejected: it would let a real performance gain be explained by either genuine
memory-driven reasoning OR a plain ratchet/selection effect (submit a known-good prior
result when your own attempt is worse), which is a real confound the source project hit
and documented. This version surfaces the anchor as a non-binding suggested starting
point only -- the agent is explicitly told it is free to deviate -- so any adherence
improvement this produces is attributable to the agent actually engaging with the
suggestion, not to a safety net swallowing failed attempts. See ARCHITECTURE.md.
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware, hook_config

log = logging.getLogger(__name__)

_ANCHOR_MAX_SECONDS = 180            # 3 minutes -- single bounded call, before the episode starts
_ANCHOR_WARN_FRACTION = 0.6          # soft nudge at 60% elapsed, hard stop at 100%
_ANCHOR_RECURSION_LIMIT = 20         # backstop only -- see periodic_critic.py's identical constant
_ANCHOR_MARKER_PREFIX = "STRATEGY ANCHOR (from best prior result)"

_RETRY_NUDGE = (
    "You did not call submit_anchor. Call it now with your best read of the past "
    "learnings so far -- do not investigate further, do not explain why, just call the tool."
)

ANCHOR_SYSTEM_PROMPT = """You are reviewing what worked best in PAST, INDEPENDENT attempts at this
same credit risk ranking task, before a NEW attempt begins -- the new agent has not started yet.

You are given PAST LEARNINGS: positive/negative points distilled from prior episodes,
including their approaches and results (scored on validation data only -- never holdout).

Identify the SINGLE strongest prior approach mentioned, and describe it concretely enough
that a reader could actually implement it -- name the specific model type, features,
and parameter values if they are stated. This is a recommended STARTING
POINT for the new agent to consider, not a mandate -- it is free to deviate if it has a
good reason to.

Do not tell the agent to fall back to this approach if its own attempt doesn't beat it --
that is not your job here. Your only job is to make sure a concrete, specific starting
point is clearly on the table before the agent begins, instead of the agent having to
reconstruct this from a long list of past learnings on its own.

Call submit_anchor(recommended_approach=..., rationale=...) exactly once. If past learnings
don't clearly point to one strong approach (e.g. this is an early attempt), say so plainly.
"""


def _build_anchor_tool():
    from langchain_core.tools import tool

    @tool
    def submit_anchor(recommended_approach: str, rationale: str) -> str:
        """Call this exactly once with the single strongest prior approach you found
        (concrete enough to implement) and why you picked it."""
        return "Anchor recorded."

    return submit_anchor


def _find_submit_anchor_call(messages: list) -> dict | None:
    """Same status-aware extraction as periodic_critic.py's _find_submit_guidance_call --
    only a call whose paired ToolMessage isn't status="error" counts, first successful one
    wins."""
    tool_results_by_id = {
        getattr(msg, "tool_call_id", None): msg
        for msg in messages
        if msg.__class__.__name__ == "ToolMessage"
    }
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") != "submit_anchor":
                continue
            result = tool_results_by_id.get(tc.get("id"))
            if result is not None and getattr(result, "status", "success") == "error":
                continue
            return tc.get("args", {})
    return None


def _write_anchor_trace(path: pathlib.Path, messages: list) -> None:
    """Persist this episode's single anchor-firing conversation for diagnosability --
    same rationale as periodic_critic.py's _write_critic_trace. Overwrites (not appends):
    unlike the periodic critic, this fires at most once per episode, so there is never a
    second firing to accumulate alongside the first."""
    from agent.app.agent import _serialize_message  # local: avoids a circular import
    # (agent.py imports graph.py, which imports this module at module level)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            for i, msg in enumerate(messages):
                entry = _serialize_message(msg, "preloop_anchor", i)
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        log.warning("[PreLoopAnchor] failed to write anchor trace to %s: %s", path, exc)


def _stream_collect(graph, messages: list, config: dict, out: list) -> None:
    """Same streaming-collect pattern as periodic_critic.py's _stream_collect -- a
    partial conversation survives even if graph.stream() raises partway through."""
    for chunk in graph.stream({"messages": messages}, config=config, stream_mode="updates"):
        for _node, delta in chunk.items():
            if not delta:
                continue
            out.extend(delta.get("messages", []))


class _AnchorTimeAwareness(AgentMiddleware):
    """Scoped to this ONE anchor invocation -- same two-stage soft-nudge-then-hard-stop
    design as periodic_critic.py's _CriticTimeAwareness (see that class's docstring for
    the full rationale); duplicated rather than reused because its injected wording
    names submit_guidance specifically, which would be wrong here."""

    def __init__(self, max_seconds: float) -> None:
        self.max_seconds = max_seconds
        self._start_time = time.time()
        self._warned = False

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check()

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check()

    def _check(self) -> dict[str, Any] | None:
        elapsed = time.time() - self._start_time
        fraction = elapsed / self.max_seconds if self.max_seconds > 0 else 1.0

        if fraction >= 1.0:
            return {
                "messages": [HumanMessage(content=(
                    "Time is up for this strategy review -- stopping now."
                ))],
                "jump_to": "end",
            }
        if fraction >= _ANCHOR_WARN_FRACTION and not self._warned:
            self._warned = True
            remaining = max(0.0, self.max_seconds - elapsed)
            return {"messages": [HumanMessage(content=(
                f"~{remaining:.0f} seconds left for this strategy review -- call "
                "submit_anchor now with your best read so far, even if brief."
            ))]}
        return None


class PreLoopAnchorMiddleware(AgentMiddleware):
    """Fires exactly ONCE, on the very first before_model call of the episode -- before
    the main agent's own first turn. No-op on every subsequent call, and no-op entirely
    if there is no learnings text to anchor on (e.g. an instance's own iteration 1)."""

    def __init__(self, llm, learnings_text: str, anchor_trace_path: pathlib.Path) -> None:
        self.llm = llm
        self.learnings_text = learnings_text
        self.anchor_trace_path = anchor_trace_path
        self._fired = False

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._fired or not self.learnings_text:
            return None
        self._fired = True
        return self._run_anchor()

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._fired or not self.learnings_text:
            return None
        self._fired = True
        return self._run_anchor()

    def _run_anchor(self) -> dict[str, Any] | None:
        try:
            anchor = self._invoke_anchor()
        except Exception as exc:
            log.warning("[PreLoopAnchor] failed, skipping: %s", exc)
            return None
        if anchor is None:
            log.warning("[PreLoopAnchor] no anchor produced, skipping")
            return None

        content = (
            f"{_ANCHOR_MARKER_PREFIX}:\n"
            f"Recommended approach: {anchor.get('recommended_approach', '')}\n"
            f"Rationale: {anchor.get('rationale', '')}"
        )
        return {"messages": [HumanMessage(content=content, id=str(uuid.uuid4()))]}

    def _build_anchor_graph(self, max_seconds: float):
        submit_anchor = _build_anchor_tool()
        return create_agent(
            model=self.llm,
            tools=[submit_anchor],
            system_prompt=ANCHOR_SYSTEM_PROMPT,
            middleware=[_AnchorTimeAwareness(max_seconds=max_seconds)],
        )

    def _invoke_anchor(self) -> dict | None:
        initial_message = (
            f"=== PAST LEARNINGS (from prior, independent episodes) ===\n"
            f"{self.learnings_text or '(none yet)'}\n\n"
            "Identify the single strongest prior approach and describe it now."
        )
        messages: list = [HumanMessage(content=initial_message)]
        all_messages: list = list(messages)
        start = time.time()
        invoke_config = {"recursion_limit": _ANCHOR_RECURSION_LIMIT}
        try:
            graph = self._build_anchor_graph(max_seconds=_ANCHOR_MAX_SECONDS)
            try:
                _stream_collect(graph, messages, invoke_config, all_messages)
            except Exception as exc:
                log.warning("[PreLoopAnchor] stream raised (%s) -- checking for a "
                            "salvageable call from what was collected before that", exc)
            anchor = _find_submit_anchor_call(all_messages)

            remaining = _ANCHOR_MAX_SECONDS - (time.time() - start)
            if anchor is None and remaining > 0:
                nudge = HumanMessage(content=_RETRY_NUDGE)
                retry_seed = list(all_messages) + [nudge]
                all_messages.append(nudge)
                retry_graph = self._build_anchor_graph(max_seconds=remaining)
                try:
                    _stream_collect(retry_graph, retry_seed, invoke_config, all_messages)
                except Exception as exc:
                    log.warning("[PreLoopAnchor] retry stream raised (%s) -- checking "
                                "for a salvageable call from what was collected before that", exc)
                anchor = _find_submit_anchor_call(all_messages)

            return anchor
        finally:
            _write_anchor_trace(self.anchor_trace_path, all_messages)
