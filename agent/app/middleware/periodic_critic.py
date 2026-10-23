"""Periodic in-episode critic. The `RunConfig.use_periodic_critic=True` path leaves
`use_learnings_reinjection` False, so LearningsReinjectionMiddleware is never constructed
alongside this middleware for the same run.

Motivation: an earlier design tried in the source research project (static, every-turn
reinjection of learnings -- see learnings_reinjection.py) produced BYTE-IDENTICAL
risk_model.pkl artifacts across 3 consecutive iterations -- the every-turn "Keep doing: X
-- APPLY this now" framing appears to have caused the agent to stop exploring and just
copy the recipe verbatim, rather than robustly re-deriving a good solution each time. This
middleware bets that (a) lower frequency (every _CRITIC_FIRE_INTERVAL turns, not every
turn) preserves context/attention budget for actual work, and (b) dynamic, trace-aware,
ADVISORY (not imperative) guidance gets adherence without collapsing exploration into
copying. See ARCHITECTURE.md for the full writeup of this tradeoff.

Architecturally this is a lightweight cousin of post_mortem.py's design, not a copy of it:
post-mortem reviews a FINISHED episode's trace from disk, can afford a slow multi-turn
tool-using agent (bounded subagent-trace tools, up to 10 minutes) since it only runs once,
after the episode is already over. This middleware reviews the LIVE, in-progress episode's
trace directly from LangGraph state (no disk read needed) and must be fast and cheap since
it fires multiple times DURING an episode, competing with the main agent's own
run_label-configured episode budget (60 or 90 minutes depending on run_label -- see
RunConfig.max_episode_seconds) -- so it's a single structured LLM call, not a
multi-turn agent, and deliberately
scoped to the main trace only (not subagent traces -- matches the original design
discussion; revisit only if v1 proves insufficient).

Injection reuses learnings_reinjection.py's state-scanning remove-then-refresh pattern (not
id-remembering -- see that module's docstring for why remembering an id across independent
conversations is unsafe), but matched by a fixed PREFIX marker rather than exact content
equality, since the critic's suggestion text is different every firing.
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain.agents.middleware import AgentMiddleware, hook_config

log = logging.getLogger(__name__)

_CRITIC_FIRE_INTERVAL = 10          # fire every Nth before_model call, no-op otherwise
_CRITIC_MAX_SECONDS = 180           # 3 minutes -- must stay small, this runs mid-episode
_CRITIC_WARN_FRACTION = 0.6         # soft nudge at 60% elapsed, hard stop at 100%
_CRITIC_TRACE_TOKEN_BUDGET = 20_000  # bounded excerpt of the LIVE trace, most-recent-first
_GUIDANCE_MARKER_PREFIX = "IN-EPISODE GUIDANCE"  # scan target for remove-then-refresh
# Backstop only, not the primary bound -- a real incident in the source research
# project's dry run showed why this needs to be a backstop, not the primary bound: a
# recursion_limit alone (denominated in steps, not time) is a poor proxy for what we
# actually care about (wall-clock cost against the main episode's budget), and a purely
# soft time-based nudge has no teeth if the model just doesn't comply (same risk we moved
# away from with the mechanical-gate discussion earlier). _CriticTimeAwareness below is
# the PRIMARY bound (soft nudge at _CRITIC_WARN_FRACTION, hard jump_to=end at 100% of
# _CRITIC_MAX_SECONDS); this recursion_limit exists only in case that mechanism itself
# fails for some reason (defense in depth, not the main line of defense) -- generous
# rather than tight, since time should realistically cut things off first.
_CRITIC_RECURSION_LIMIT = 20

_RETRY_NUDGE = (
    "You did not call submit_guidance. Call it now with your best read of the trace so "
    "far -- do not investigate further, do not explain why, just call the tool."
)

CRITIC_SYSTEM_PROMPT = """You are a brief check-in on an agent that is CURRENTLY, RIGHT NOW, in the middle of
working on a credit risk ranking task. It has NOT finished. Your job is not to grade it or
write a retrospective -- it's to give one short, useful observation and one concrete
suggestion for what to do NEXT, based on what it's done SO FAR and what similar past
attempts (given below as "past learnings") found worked or didn't.

This is advice, not a command. The agent may already be doing something reasonable that
just looks unfamiliar from a short excerpt -- if the trace shows steady, sensible
progress and nothing stands out, it is completely fine to say so plainly and suggest it
simply continue. Do not manufacture a suggestion just to have one, and do not tell it to
just repeat a past approach verbatim unless the trace excerpt genuinely shows it's stuck,
drifting off-task, or about to repeat a specific documented mistake.

You are given:
- PAST LEARNINGS: positive/negative points distilled from PRIOR episodes (not this one).
- RECENT TRACE EXCERPT: the tail end of THIS episode's own conversation so far (may be
  truncated to the most recent portion if long -- a truncation notice will say so).

Call submit_guidance(observation=..., suggestion=...) exactly once:
- observation: one sentence, what you actually notice in the recent trace (progress,
  drift, a repeated mistake, or "on track" if nothing stands out).
- suggestion: one sentence, a concrete next action or consideration -- phrased as advice
  ("consider...", "you may want to...", "worth double-checking..."), never as an order.

Keep both fields short -- this is a quick check-in, not an essay. Do not restate the
entire past-learnings list; reference at most the one or two most relevant to what you
actually observed.

Once you call submit_guidance, you are done: do not re-analyze, second-guess, or call it
again. If you find yourself wanting to reconsider after already submitting, stop -- your
first answer stands.
"""


def _build_guidance_tool():
    from langchain_core.tools import tool

    @tool
    def submit_guidance(observation: str, suggestion: str) -> str:
        """Call this exactly once with your one-sentence observation and one-sentence
        advisory suggestion. This is how your check-in is recorded."""
        return "Guidance recorded."

    return submit_guidance


def _find_submit_guidance_call(messages: list) -> dict | None:
    """Same status-aware extraction as post_mortem.py's _find_submit_learnings_call --
    only a call whose paired ToolMessage isn't status="error" counts, and the first
    successful one wins over a later unprompted re-submission."""
    tool_results_by_id = {
        getattr(msg, "tool_call_id", None): msg
        for msg in messages
        if msg.__class__.__name__ == "ToolMessage"
    }
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") != "submit_guidance":
                continue
            result = tool_results_by_id.get(tc.get("id"))
            if result is not None and getattr(result, "status", "success") == "error":
                continue
            return tc.get("args", {})
    return None


def _build_bounded_trace_excerpt(messages: list, max_tokens: int = _CRITIC_TRACE_TOKEN_BUDGET) -> str:
    """Format the LIVE episode's recent messages as readable text, keeping the most
    RECENT entries (drop oldest first) if over budget -- the critic cares about what's
    happening now, not the full episode history (episode_injector.py's initial context
    already covers cross-episode history; this is in-episode only)."""
    real_messages = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage)) and getattr(m, "content", None)
    ]
    if not real_messages:
        return "(no trace yet)"

    trimmed = list(real_messages)
    dropped = 0
    while count_tokens_approximately(trimmed) > max_tokens and len(trimmed) > 1:
        trimmed.pop(0)
        dropped += 1

    lines = []
    if dropped:
        lines.append(f"[TRACE EXCERPT TRUNCATED: {dropped} earlier entries omitted, showing most recent only]")
    for m in trimmed:
        lines.append(f"[{m.__class__.__name__}] {m.content}")
    return "\n".join(lines)


def _write_critic_trace(path: pathlib.Path, messages: list, firing_turn: int) -> None:
    """Persist one critic firing's conversation for diagnosability -- previously a
    failure only logged a bare exception message with no way to see WHY after the fact
    (real gap, 2026-08-13: a GraphRecursionError gave zero insight into what the model
    actually did across those turns). Appends (not overwrites) since multiple firings
    share one episode's critic trace file."""
    from agent.app.agent import _serialize_message  # local: avoids a circular import
    # (agent.py imports graph.py, which imports this module at module level)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a") as f:
            for i, msg in enumerate(messages):
                entry = _serialize_message(msg, "periodic_critic", i)
                entry["firing_turn"] = firing_turn
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        log.warning("[PeriodicCritic] failed to write critic trace to %s: %s", path, exc)


def _stream_collect(graph, messages: list, config: dict, out: list) -> None:
    """Streams into `out` (appending each new message as it happens) rather than
    returning a value on completion -- so a partial conversation survives even if
    graph.stream() raises partway through (e.g. GraphRecursionError). Returning a
    value only helps on success; the whole point here is capturing what happened
    when it DOESN'T succeed."""
    for chunk in graph.stream({"messages": messages}, config=config, stream_mode="updates"):
        for _node, delta in chunk.items():
            if not delta:
                continue
            out.extend(delta.get("messages", []))


class _CriticTimeAwareness(AgentMiddleware):
    """Soft nudge + hard force-stop, scoped to ONE critic invocation -- NOT a reuse of
    the main TimeAwarenessMiddleware, deliberately: that class's injected wording is
    specific to the main episode's deliverables ("Save risk_model.pkl..."), which makes
    no sense for a critic whose only job is calling submit_guidance, and its
    exit_on_expiry defaults to False (never actually tested with a forced jump_to=end in
    production) whereas the critic needs exit_on_expiry ALWAYS on -- unlike the main
    episode (where ResultsGuardMiddleware provides a downstream recovery step after a
    soft timeout), the critic has no such follow-up: a hard stop here IS the recovery
    mechanism, not an escalation past it.

    Two-stage, not three like the main episode's 50/75/90% (a 3-minute budget doesn't
    need that granularity): one nudge at _CRITIC_WARN_FRACTION giving the model a real
    chance to see it and respond on its NEXT turn, then a hard jump_to=end at 100%.
    Injecting a message in the SAME call as jump_to=end (as the expiry branch below
    does) means the model never actually gets to act on that final message -- the graph
    stops immediately -- so it's a record of why we stopped, not a message expected to
    change behavior; the earlier nudge is what actually gives the model a chance.
    """

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
                    "Time is up for this check-in -- stopping now."
                ))],
                "jump_to": "end",
            }
        if fraction >= _CRITIC_WARN_FRACTION and not self._warned:
            self._warned = True
            remaining = max(0.0, self.max_seconds - elapsed)
            return {"messages": [HumanMessage(content=(
                f"~{remaining:.0f} seconds left for this check-in -- call submit_guidance "
                "now with your best read so far, even if brief."
            ))]}
        return None


class PeriodicCriticMiddleware(AgentMiddleware):
    """Fires every _CRITIC_FIRE_INTERVAL before_model calls; no-op otherwise. On a firing
    turn, spawns a bounded single-call critic review of the live trace + past learnings
    and injects its suggestion, replacing any previously-injected guidance.
    """

    def __init__(self, llm, learnings_text: str, critic_trace_path: pathlib.Path) -> None:
        self.llm = llm
        self.learnings_text = learnings_text
        self.critic_trace_path = critic_trace_path
        self._turn_count = 0

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._turn_count += 1
        if self._turn_count % _CRITIC_FIRE_INTERVAL != 0:
            return None
        return self._run_critic(state)

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._turn_count += 1
        if self._turn_count % _CRITIC_FIRE_INTERVAL != 0:
            return None
        return self._run_critic(state)

    def _run_critic(self, state: Any) -> dict[str, Any] | None:
        try:
            guidance = self._invoke_critic(state)
        except Exception as exc:
            log.warning("[PeriodicCritic] failed, skipping this firing: %s", exc)
            return None
        if guidance is None:
            log.warning("[PeriodicCritic] no guidance produced, skipping this firing")
            return None

        content = (
            f"{_GUIDANCE_MARKER_PREFIX} (turn {self._turn_count}):\n"
            f"Observation: {guidance.get('observation', '')}\n"
            f"Suggestion: {guidance.get('suggestion', '')}"
        )
        existing_ids = [
            m.id for m in state.get("messages", [])
            if getattr(m, "id", None) is not None
            and isinstance(getattr(m, "content", None), str)
            and m.content.startswith(_GUIDANCE_MARKER_PREFIX)
        ]
        from langchain_core.messages import RemoveMessage
        import uuid
        removals = [RemoveMessage(id=mid) for mid in existing_ids]
        msg = HumanMessage(content=content, id=str(uuid.uuid4()))
        return {"messages": [*removals, msg]}

    def _build_critic_graph(self, max_seconds: float):
        """Fresh graph per attempt (not reused across the initial + retry attempt) so
        _CriticTimeAwareness's clock starts fresh, sized to whatever budget THIS
        attempt actually has left -- see _invoke_critic for how max_seconds is computed
        for the retry so the two attempts together never exceed _CRITIC_MAX_SECONDS."""
        submit_guidance = _build_guidance_tool()
        return create_agent(
            model=self.llm,
            tools=[submit_guidance],
            system_prompt=CRITIC_SYSTEM_PROMPT,
            middleware=[_CriticTimeAwareness(max_seconds=max_seconds)],
        )

    def _invoke_critic(self, state: Any) -> dict | None:
        trace_excerpt = _build_bounded_trace_excerpt(state.get("messages", []))
        initial_message = (
            f"=== PAST LEARNINGS (from prior episodes) ===\n{self.learnings_text or '(none yet)'}\n\n"
            f"=== RECENT TRACE EXCERPT (this episode, in progress) ===\n{trace_excerpt}\n\n"
            "Give your one-sentence observation and one-sentence suggestion now."
        )
        messages: list = [HumanMessage(content=initial_message)]
        all_messages: list = list(messages)
        start = time.time()
        # recursion_limit is a BACKSTOP here, not the primary bound -- see the constant's
        # own comment. _CriticTimeAwareness (built fresh per attempt, below) is what
        # actually enforces _CRITIC_MAX_SECONDS: a soft nudge partway through, then a
        # hard jump_to=end at 100% of budget, enforced mid-call via LangGraph's own
        # control flow (not a post-hoc "time.time() - start" check between calls, which
        # can't interrupt a call already in progress -- the real gap in the original,
        # step-only version of this fix).
        invoke_config = {"recursion_limit": _CRITIC_RECURSION_LIMIT}
        try:
            # Real incident, 2026-08-13 re-validation: the model called submit_guidance
            # SUCCESSFULLY on its very first turn (good, usable content), then kept
            # calling it again anyway (7 total) instead of stopping, until the
            # recursion_limit backstop raised -- discarding the perfectly good first
            # answer along with the wasted repeats, since the old code never re-checked
            # all_messages after an exception. Extraction always takes the FIRST
            # successful call (_find_submit_guidance_call's own contract), so a stream
            # that raises partway through must still be checked for one before giving
            # up -- the exception means "something went wrong eventually," not
            # "nothing usable happened at all."
            graph = self._build_critic_graph(max_seconds=_CRITIC_MAX_SECONDS)
            try:
                _stream_collect(graph, messages, invoke_config, all_messages)
            except Exception as exc:
                log.warning("[PeriodicCritic] stream raised (%s) -- checking for a "
                            "salvageable call from what was collected before that", exc)
            guidance = _find_submit_guidance_call(all_messages)

            remaining = _CRITIC_MAX_SECONDS - (time.time() - start)
            if guidance is None and remaining > 0:
                nudge = HumanMessage(content=_RETRY_NUDGE)
                retry_seed = list(all_messages) + [nudge]
                all_messages.append(nudge)
                retry_graph = self._build_critic_graph(max_seconds=remaining)
                try:
                    _stream_collect(retry_graph, retry_seed, invoke_config, all_messages)
                except Exception as exc:
                    log.warning("[PeriodicCritic] retry stream raised (%s) -- checking "
                                "for a salvageable call from what was collected before that", exc)
                guidance = _find_submit_guidance_call(all_messages)

            return guidance
        finally:
            # Always persists, success or failure -- streaming (not invoke()) means
            # all_messages already holds whatever happened even if an exception
            # (e.g. GraphRecursionError) propagates out of this block.
            _write_critic_trace(self.critic_trace_path, all_messages, self._turn_count)
