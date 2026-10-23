"""
TokenAwarenessMiddleware — injects token-budget messages before model calls, mirroring
TimeAwarenessMiddleware's wall-clock pattern (agent/app/middleware/time_awareness.py) but
tracking cumulative token usage instead of elapsed wall-clock time.

A model could burn its wall-clock budget on a small number of very long,
token-heavy turns without actually running out of "thinking," or vice
versa -- so token usage is tracked as its own, separate budget rather than inferred from
elapsed time. Token budget is primary; wall-clock stays a loose backstop (see
TimeAwarenessMiddleware's max_seconds).

CALIBRATION NOTE: DEFAULT_MAX_TOKENS below is a starting placeholder, not a calibrated
value -- recalibrate it from a real pilot's observed token-usage distribution for this
task (model/backend-dependent) rather than trusting the number shipped here.

KNOWN GAP, not addressed here: same class of problem as
agent/app/middleware/subagent_time_awareness.py's documented gap -- this middleware's
before_model hook only fires on the MAIN graph, so while the main graph is blocked
awaiting a delegated task() subagent call, it never fires, even though token_counter.
total_tokens keeps accumulating from the subagent's own calls during that time (confirmed
by TokenCountingCallbackHandler's own docstring: it sees every model call the same way,
main or subagent). Unlike time (a subagent naturally gets its own fresh clock via
time.time()), tokens are a single shared running total across the whole episode, so a
subagent-side equivalent would need delta-tracking ("tokens used since this delegation
started"), not just a copy of this class -- a genuinely different, untested design. Given
SubagentTimeAwarenessMiddleware (the closest precedent) is itself still explicitly marked
tentative/unreviewed by the user, building a new speculative mechanism for tokens wasn't
in scope here -- flagged as a follow-up, not solved.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)

_WARN_THRESHOLDS = (0.50, 0.75, 0.90)

# Placeholder only -- see module docstring. Recalibrate against this task's own real
# token-usage distribution once a pilot batch exists.
DEFAULT_MAX_TOKENS = 500_000


class TokenAwarenessMiddleware(AgentMiddleware):
    """
    Injects token-remaining messages into the agent conversation at budget thresholds,
    reading the live running total from a TokenCountingCallbackHandler
    (agent/app/callbacks/token_counter.py) -- the same handler ThermalPauseMiddleware
    already reads `total_tokens` from, so this needs no new tracking mechanism, only a
    new consumer of the existing one.

    Args:
        token_counter: a TokenCountingCallbackHandler instance (shared with the episode's
            other callbacks/middleware, e.g. ThermalPauseMiddleware) -- read-only here.
        max_tokens: total episode token budget. See DEFAULT_MAX_TOKENS's docstring
            warning -- treat as provisional until the pilot recalibrates it.
        exit_on_expiry: If True, jump to "end" once the budget is exhausted.
                        If False (default), just warn -- ResultsGuardMiddleware
                        handles the no-results-json case after the agent stops.
    """

    def __init__(self, token_counter: Any, max_tokens: int = DEFAULT_MAX_TOKENS, exit_on_expiry: bool = False) -> None:
        self.token_counter = token_counter
        self.max_tokens = max_tokens
        self.exit_on_expiry = exit_on_expiry
        self._injected_at: set[float] = set()

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check()

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check()

    def _check(self) -> dict[str, Any] | None:
        tokens_so_far = self.token_counter.total_tokens
        fraction = tokens_so_far / self.max_tokens
        remaining = max(0, self.max_tokens - tokens_so_far)

        threshold_hit = None
        for t in _WARN_THRESHOLDS:
            if fraction >= t and t not in self._injected_at:
                threshold_hit = t

        expired = fraction >= 1.0 and 1.0 not in self._injected_at

        if threshold_hit is None and not expired:
            return None

        if expired:
            self._injected_at.add(1.0)
            log.warning("[TokenAwareness] Episode token budget exhausted (%s tokens used)", tokens_so_far)
            msg = (
                f"EPISODE TOKEN BUDGET EXPIRED: Your {self.max_tokens:,}-token budget is used up. "
                "Save risk_model.pkl IMMEDIATELY using whatever you have. "
                "Do not run any more experiments — just save it."
            )
            if self.exit_on_expiry:
                return {"messages": [HumanMessage(content=msg)], "jump_to": "end"}
            return {"messages": [HumanMessage(content=msg)]}

        self._injected_at.add(threshold_hit)
        log.info("[TokenAwareness] %.0f%% of token budget used (%s/%s) — injecting warning",
                  fraction * 100, tokens_so_far, self.max_tokens)

        if threshold_hit >= 0.90:
            msg = (
                f"TOKEN WARNING (90%): Only ~{remaining:,} tokens remaining in this episode's budget. "
                "Stop optimising. Save risk_model.pkl NOW with your "
                "current best model. A saved artifact beats no artifact."
            )
        elif threshold_hit >= 0.75:
            msg = (
                f"TOKEN WARNING (75%): ~{remaining:,} tokens remaining. "
                "You should have a working pipeline by now. "
                "Save risk_model.pkl with your current best approach, "
                "then iterate only if your remaining budget clearly permits."
            )
        else:  # 50%
            msg = (
                f"TOKEN CHECK (50%): ~{remaining:,} tokens remaining in this episode's budget. "
                "If you don't yet have a working end-to-end pipeline, simplify "
                "now. A simple saved risk_model.pkl beats a sophisticated one that never "
                "gets saved."
            )

        return {"messages": [HumanMessage(content=msg)]}
