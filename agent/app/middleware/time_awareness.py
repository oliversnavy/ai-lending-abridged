"""
TimeAwarenessMiddleware — injects wall-clock budget messages before model calls.

The agent has no intrinsic sense of elapsed time. Without reminders, it
over-invests in model iteration and runs out of context before saving its
deliverable (risk_model.pkl). This middleware injects a
brief HumanMessage at 50%, 75%, and 90% of the episode budget so the agent
can prioritise accordingly.

At 90%+ with can_jump_to=["end"], it can also hard-terminate if needed
(currently just warns — flip exit_on_expiry=True to enable hard stop).

Deliberately does not ask the agent to compute/report final metrics under time
pressure — that combination (time pressure + "just write a number") is what
produced a fabricated result in an early pilot episode. c_stat/etc. are computed
independently from the saved artifact by agent/evaluator/evaluate_episode.py.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)

# Thresholds at which we inject a time message (fraction of max_seconds elapsed).
# Chosen to give the agent meaningful decision points without spamming context.
_WARN_THRESHOLDS = (0.50, 0.75, 0.90)


class TimeAwarenessMiddleware(AgentMiddleware):
    """
    Injects time-remaining messages into the agent conversation at budget thresholds.

    Args:
        max_seconds: Total episode wall-clock budget. Default 3600s (60 min).
        exit_on_expiry: If True, jump to "end" once the budget is exhausted.
                        If False (default), just warn — ResultsGuardMiddleware
                        handles the no-results-json case after the agent stops.
    """

    def __init__(self, max_seconds: int = 3600, exit_on_expiry: bool = False) -> None:
        self.max_seconds = max_seconds
        self.exit_on_expiry = exit_on_expiry
        self._start_time = time.time()
        self._injected_at: set[float] = set()  # thresholds already injected

    # ------------------------------------------------------------------
    # Sync path
    # ------------------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check(state)

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check(state)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check(self, state: Any) -> dict[str, Any] | None:
        elapsed = time.time() - self._start_time
        fraction = elapsed / self.max_seconds
        remaining_s = max(0.0, self.max_seconds - elapsed)
        remaining_min = remaining_s / 60

        # Find the highest threshold we've passed but not yet injected
        threshold_hit = None
        for t in _WARN_THRESHOLDS:
            if fraction >= t and t not in self._injected_at:
                threshold_hit = t

        # Hard expiry: past 100%
        expired = fraction >= 1.0 and 1.0 not in self._injected_at

        if threshold_hit is None and not expired:
            return None

        if expired:
            self._injected_at.add(1.0)
            log.warning("[TimeAwareness] Episode budget exhausted (%.0fs elapsed)", elapsed)
            msg = (
                f"EPISODE TIME EXPIRED: Your {self.max_seconds // 60}-minute budget is used up. "
                "Save risk_model.pkl IMMEDIATELY using whatever you have. "
                "Do not run any more experiments — just save it."
            )
            if self.exit_on_expiry:
                return {"messages": [HumanMessage(content=msg)], "jump_to": "end"}
            return {"messages": [HumanMessage(content=msg)]}

        # Mark threshold injected
        self._injected_at.add(threshold_hit)
        log.info("[TimeAwareness] %.0f%% elapsed (%.0fs) — injecting time warning", fraction * 100, elapsed)

        if threshold_hit >= 0.90:
            msg = (
                f"TIME WARNING (90%): Only {remaining_min:.0f} minutes remaining in this episode. "
                "Stop optimising. Save risk_model.pkl NOW with your "
                "current best model. A saved artifact beats no artifact."
            )
        elif threshold_hit >= 0.75:
            msg = (
                f"TIME WARNING (75%): {remaining_min:.0f} minutes remaining. "
                "You should have a working pipeline by now. "
                "Save risk_model.pkl with your current best approach, "
                "then iterate only if time clearly permits."
            )
        else:  # 50%
            msg = (
                f"TIME CHECK (50%): {remaining_min:.0f} minutes remaining in this episode. "
                "If you don't yet have a working end-to-end pipeline, simplify now. "
                "Logistic regression beats no model; a simple saved risk_model.pkl "
                "beats a sophisticated one that never gets saved."
            )

        return {"messages": [HumanMessage(content=msg)]}
