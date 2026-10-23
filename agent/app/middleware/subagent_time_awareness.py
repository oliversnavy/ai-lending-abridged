"""
SubagentTimeAwarenessMiddleware — gives a delegated subagent its own time-budget
awareness, independent of the main agent's.

Background: TimeAwarenessMiddleware warns the main graph at 50/75/90% of the episode
budget, but that hook only fires when the *main* graph invokes its own model — while the
main graph is blocked awaiting a `task()` subagent call, it never runs, so a subagent can
silently consume the entire episode budget with no time signal at all (confirmed
empirically in the source research project: one subagent call ran for hours with zero
warnings, consuming well over its episode's nominal time budget, entirely inside
delegation).

The fix: the primary agent is prompted (SUBAGENT_GUIDANCE in
agent/app/prompts/system.py) to give the subagent the amount of time it thinks it can
afford for the given task (e.g. "complete this task within 20 minutes"), and this
middleware then gives the subagent a real sense of how much of that stated budget has
elapsed. This module is a reasonable first pass at that: the primary is prompted
(SUBAGENT_GUIDANCE in agent/app/prompts/system.py) to state a time budget in its `task()`
call's freeform `description` argument; this middleware parses that budget out of the
subagent's own first message (deepagents sets the subagent's initial state to exactly
`[HumanMessage(content=description)]` — see deepagents/middleware/subagents.py's
`_validate_and_prepare_state`) and warns the subagent at the same 50/75/90% cadence as the
main graph.

Known limitation, flagged rather than solved: deepagents compiles a subagent's graph
(and its middleware list) ONCE per episode in agent/graph.py's build_graph(), and every
`task()` call to that subagent type within the episode reuses the same compiled graph —
there is no per-invocation middleware instance. This middleware works around that by
treating "the subagent's message list contains exactly one HumanMessage" as a signal that
a new delegation has just started, and resets its own clock/budget at that point. That
heuristic is correct for the common case (delegations run one at a time), but two
genuinely concurrent `task()` calls to the same subagent type in one turn would race on
this shared instance's state. Not addressed here — a per-invocation-safe design (e.g.
threading budget/start-time through the subagent's own graph state instead of instance
attributes) is a reasonable next step if the user confirms this overall direction.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)

_WARN_THRESHOLDS = (0.50, 0.75, 0.90)
_DEFAULT_BUDGET_SECONDS = 1200  # 20 min fallback, if the primary didn't state a budget

_BUDGET_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"(\d+)\s*-?\s*hour", re.IGNORECASE), 3600),
    (re.compile(r"(\d+)\s*-?\s*min(?:ute)?s?\b", re.IGNORECASE), 60),
    (re.compile(r"(\d+)\s*-?\s*sec(?:ond)?s?\b", re.IGNORECASE), 1),
]


def _parse_stated_budget(text: str) -> int | None:
    """Best-effort extraction of a time budget from freeform task() description text,
    e.g. "...complete this task within 20 minutes" -> 1200. Returns None if no time
    expression is found (caller should fall back to a default)."""
    for pattern, unit_seconds in _BUDGET_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1)) * unit_seconds
    return None


class SubagentTimeAwarenessMiddleware(AgentMiddleware):
    """Warns a subagent at 50/75/90% of a time budget parsed from its own first message."""

    def __init__(self, default_seconds: int = _DEFAULT_BUDGET_SECONDS) -> None:
        self.default_seconds = default_seconds
        self._start_time: float | None = None
        self._budget_seconds: int = default_seconds
        self._injected_at: set[float] = set()

    def _maybe_reset(self, state: Any) -> None:
        """A subagent invocation always starts with exactly one HumanMessage (the
        task() call's description) — treat that shape as "a new delegation just
        started" and (re)parse the budget from it."""
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            content = messages[0].content
            if not isinstance(content, str):
                content = str(content)
            parsed = _parse_stated_budget(content)
            self._start_time = time.time()
            self._budget_seconds = parsed if parsed is not None else self.default_seconds
            self._injected_at = set()
            log.info(
                "[SubagentTimeAwareness] new delegation started, budget=%ds (%s)",
                self._budget_seconds,
                "stated by primary" if parsed is not None else "default fallback",
            )

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check(state)

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check(state)

    def _check(self, state: Any) -> dict[str, Any] | None:
        self._maybe_reset(state)
        if self._start_time is None:
            return None

        elapsed = time.time() - self._start_time
        fraction = elapsed / self._budget_seconds
        remaining_min = max(0.0, self._budget_seconds - elapsed) / 60

        threshold_hit = None
        for t in _WARN_THRESHOLDS:
            if fraction >= t and t not in self._injected_at:
                threshold_hit = t
        if threshold_hit is None:
            return None
        self._injected_at.add(threshold_hit)

        log.info(
            "[SubagentTimeAwareness] %.0f%% of %ds subtask budget elapsed",
            fraction * 100, self._budget_seconds,
        )

        if threshold_hit >= 0.90:
            msg = (
                f"SUBTASK TIME WARNING (90%): only ~{remaining_min:.0f} min left of the time budget "
                "for this delegated subtask. Wrap up now — return your best result to the primary "
                "agent rather than continuing to iterate."
            )
        elif threshold_hit >= 0.75:
            msg = (
                f"SUBTASK TIME CHECK (75%): ~{remaining_min:.0f} min left of the budget for this "
                "subtask. Start converging on a result to hand back."
            )
        else:
            msg = (
                f"SUBTASK TIME CHECK (50%): ~{remaining_min:.0f} min left of the budget for this "
                "subtask. If the current approach isn't working, simplify rather than keep tuning."
            )
        return {"messages": [HumanMessage(content=msg)]}
