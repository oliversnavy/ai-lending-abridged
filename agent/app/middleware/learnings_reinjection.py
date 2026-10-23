"""
LearningsReinjectionMiddleware -- re-emphasizes post_mortem.py's learnings at every turn
instead of relying solely on the one-time initial-context injection
(episode_injector.py's build_initial_human_message, still the primary history view).

Rationale: the source research project's traces showed agents DO distill prior learnings
well when asked -- post-mortem output references specific episodes, patterns, common
failure modes -- but don't reliably ADHERE to them mid-episode (a later episode re-hit
the exact pickle/__main__ bug an earlier one had already flagged as a common failure
mode). This tests whether keeping learnings "front and center" (most-recent message,
refreshed every turn) closes that adherence gap. Deliberately decoupled from a new
distillation step: reuses EpisodeRecord.learnings_positive/negative exactly as
post_mortem.py already produces them -- zero new LLM calls.

Kept in this repo but NOT wired into either shipped RUN_CONFIGS entry -- see
ARCHITECTURE.md for why: a real, documented incident (several consecutive episodes
produced byte-identical artifacts once this every-turn reinjection was tried) led to
periodic_critic.py instead, which is what ltm_full actually uses for in-episode
guidance. Kept here as a tried-and-rejected alternative worth showing, not hiding.

Uses a remove-then-append pattern (RemoveMessage by id) so the reminder never
accumulates: at any point there is at most ONE reminder message in state. Each turn
just repositions it to stay near the end of the transcript (closest to the model's
attention, and to whatever ContextEditingMiddleware/SummarizationMiddleware compress
next) rather than growing unboundedly with turn count.

Must be instantiated separately for the main agent and for the general-purpose
subagent (agent/graph.py) -- deepagents subagents do NOT inherit the main agent's
message history ("messages" is in deepagents' _EXCLUDED_STATE_KEYS), so each needs
its own before_model hook in its own middleware list, and each needs its own
_last_injected_id tracking since they're independent message threads. Both instances
are built from the same reminder_text (cheap to construct twice).
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, RemoveMessage

from ..models.episode import EpisodeRecord

# Aggregated across the whole episode_index, raw learnings can otherwise grow to
# num_episodes * post_mortem._MAX_LEARNINGS_PER_CATEGORY (up to 90 items by iteration
# 30) -- capped here, most-recent-first, after exact-string dedup (a learning repeated
# verbatim across episodes -- e.g. the recurring pickle/__main__ warning -- should only
# count once, not once per episode that re-flagged it).
_MAX_REMINDER_ITEMS_PER_CATEGORY = 8


def build_learnings_reminder(episode_index: list[EpisodeRecord]) -> str:
    """Compact, deduplicated positive/negative learnings reminder for reinjection.

    Scans episode_index most-recent-first so the cap favors current operational
    state over stale learnings that stopped recurring. Returns "" if there's
    nothing to show yet (e.g. instance 0's first episode, before any post-mortem
    has run) -- callers treat "" as "middleware is a no-op this episode".
    """
    positive: list[str] = []
    negative: list[str] = []
    seen_positive: set[str] = set()
    seen_negative: set[str] = set()

    for ep in reversed(episode_index):
        for item in ep.learnings_positive:
            if item not in seen_positive:
                seen_positive.add(item)
                positive.append(item)
        for item in ep.learnings_negative:
            if item not in seen_negative:
                seen_negative.add(item)
                negative.append(item)
        if len(positive) >= _MAX_REMINDER_ITEMS_PER_CATEGORY and len(negative) >= _MAX_REMINDER_ITEMS_PER_CATEGORY:
            break

    positive = positive[:_MAX_REMINDER_ITEMS_PER_CATEGORY]
    negative = negative[:_MAX_REMINDER_ITEMS_PER_CATEGORY]

    if not positive and not negative:
        return ""

    lines = [
        "OPERATING PRINCIPLES (distilled from prior episodes in this run -- "
        "APPLY these now, don't just note them):"
    ]
    if positive:
        lines.append("Keep doing:")
        lines.extend(f"  - {item}" for item in positive)
    if negative:
        lines.append("Avoid repeating:")
        lines.extend(f"  - {item}" for item in negative)
    return "\n".join(lines)


class LearningsReinjectionMiddleware(AgentMiddleware):
    """Re-injects a fixed learnings reminder as the most-recent message before
    every model call. No-op (returns None) when reminder_text is empty.

    Removal target is determined by scanning CURRENT state for a message whose
    content matches reminder_text, not by remembering the last-injected id on
    self. The latter goes stale across a subagent's independent delegations: the
    subagent's "general-purpose" SubAgent spec builds ONE middleware instance
    that's reused across every task() call within an episode, but each
    delegation gets a fresh, empty message list ("messages" is in deepagents'
    _EXCLUDED_STATE_KEYS -- no state carries over between delegations). A
    remembered id from delegation N's conversation doesn't exist in delegation
    N+1's fresh state, so RemoveMessage(id=<stale>) crashes the episode with
    "Attempting to delete a message with an ID that doesn't exist" (a real incident in
    the source research project's production batch). Scanning current
    state sidesteps this entirely -- there's nothing to desync from.
    """

    def __init__(self, reminder_text: str) -> None:
        self.reminder_text = reminder_text

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._inject(state)

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._inject(state)

    def _inject(self, state: Any) -> dict[str, Any] | None:
        if not self.reminder_text:
            return None
        existing_ids = [
            m.id for m in state.get("messages", [])
            if getattr(m, "id", None) is not None and getattr(m, "content", None) == self.reminder_text
        ]
        removals = [RemoveMessage(id=mid) for mid in existing_ids]
        msg = HumanMessage(content=self.reminder_text, id=str(uuid.uuid4()))
        return {"messages": [*removals, msg]}
