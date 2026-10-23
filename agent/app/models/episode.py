from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class LoopMode(str, Enum):
    NONE = "none"          # one independent episode, no cross-episode memory at all
    SINGLE_PRIOR = "single"  # each episode sees only the immediately preceding one
    ALL_PRIOR = "all"      # each episode sees the full history of this loop instance


class RunConfig(BaseModel):
    """Configuration for one named run. See `RUN_CONFIGS` below for the three
    illustrative configs this repo ships, and ARCHITECTURE.md for what each
    demonstrates."""

    run_label: str
    loop_mode: LoopMode = LoopMode.NONE

    # False only for a plain ReAct agent (langchain.agents.create_agent, no subagents).
    # True routes through deepagents.create_deep_agent with a sandboxed general-purpose
    # subagent -- see agent/graph.py.
    use_deep_agent: bool = True

    # Model backend -- see agent/app/clients/llm.py. "anthropic" needs ANTHROPIC_API_KEY;
    # "baseten" needs BASETEN_API_KEY (used for the GLM-5.3-via-Baseten open-weight path).
    backend: str = "anthropic"
    reasoning_effort: str = "low"           # main agent's effort level
    critic_reasoning_effort: str = "medium"  # periodic critic / post-mortem's effort level

    max_episode_seconds: int = 1800   # 30-minute wall-clock budget per episode
    max_episode_tokens: int = 500_000  # starting placeholder -- recalibrate from a real
    # pilot's observed token-usage distribution once one exists (see
    # agent/app/middleware/token_awareness.py's calibration note).

    # Advisor-consultation pattern -- agent escalates to a senior-reviewer LLM at defined
    # checkpoints (start / decision points / stuck / end). See
    # agent/app/tools/advisor_consult.py + agent/app/prompts/advisor.py. The "advisory"
    # point on the advisory-to-embedded delivery-range spectrum this repo demonstrates.
    use_advisor: bool = False

    # Three-critic lifecycle -- see ARCHITECTURE.md for the full writeup:
    #   starting critic  = PreLoopAnchorMiddleware (fires once, before turn 1)
    #   mid-episode critic = PeriodicCriticMiddleware (fires every N turns)
    #   end-of-episode critic = post_mortem.py (runs once, after the episode ends,
    #     and is what makes loop_mode=ALL_PRIOR's cross-episode memory useful rather
    #     than just a raw metric history)
    use_preloop_anchor: bool = False
    use_periodic_critic: bool = False
    use_post_mortem: bool = False

    # Deliberately ported but never set True by any RUN_CONFIGS entry below: an earlier
    # design (re-inject the same learnings as the most-recent message on EVERY turn)
    # caused episodes to collapse into copying a known-good prior artifact verbatim
    # instead of robustly re-deriving a solution each time -- a real, documented
    # incident in the source research project. Kept in the codebase and described in
    # ARCHITECTURE.md as a tried-and-rejected alternative to PeriodicCriticMiddleware,
    # not hidden.
    use_learnings_reinjection: bool = False

    @property
    def is_baseline(self) -> bool:
        """Runs with no loop -- each episode is fully independent."""
        return self.loop_mode == LoopMode.NONE


# Three illustrative run configs. Together they demonstrate this harness's range from
# "plain deep agent" through "advisory escalation" to "fully embedded, long-term-memory
# loop with all three critics" -- the advisory-to-embedded delivery-range spectrum, not
# research findings about which config performs best.
RUN_CONFIGS: dict[str, RunConfig] = {
    "baseline": RunConfig(
        run_label="baseline",
        use_deep_agent=True,
    ),
    "advisor": RunConfig(
        run_label="advisor",
        use_deep_agent=True,
        use_advisor=True,
    ),
    "ltm_full": RunConfig(
        run_label="ltm_full",
        use_deep_agent=True,
        loop_mode=LoopMode.ALL_PRIOR,
        use_post_mortem=True,
        use_periodic_critic=True,
        use_preloop_anchor=True,
    ),
}
# Loop-level settings (max_iterations, plateau detection) are NOT part of RunConfig --
# they're global, read by agent/loop.py from configs/base.yaml's `loop` block, same for
# every looped run.


class EpisodeRecord(BaseModel):
    episode_id: int
    run_label: str

    # Primary, reported metric -- scored against holdout.parquet, which the agent never
    # touches during the episode.
    c_stat: float
    # Diagnostic-only -- scored against val.parquet, the same sample the agent's own
    # local testing was tuned against during the episode. Expected to run a bit
    # optimistic relative to holdout; used for within-loop/within-run analysis only.
    c_stat_val: float = 0.0

    eval_status: str = "unknown"  # "ok" | "partial" | "failed"
    approach: str
    hypothesis: str
    skill_path: str
    tokens_used: int = 0
    duration_s: float = 0.0

    # Post-mortem output (empty for runs with use_post_mortem=False). Capped at 3 items
    # each (prompt instruction + mechanical truncation backstop) -- see post_mortem.py.
    learnings_positive: list[str] = []
    learnings_negative: list[str] = []
    # Mechanical ids assigned by the harness (never the model) to this episode's own
    # learnings_negative items, one per entry, e.g. "ep0003-neg0" -- lets a LATER
    # episode's post-mortem reference a specific still-open item without needing to
    # match free text across episodes.
    learnings_negative_ids: list[str] = []
    # ids (from EARLIER episodes' learnings_negative_ids, in this same instance) that
    # THIS episode's post-mortem reported as directly attempted -- validated against the
    # known open-id set before being trusted, so a hallucinated/unknown id from the
    # model can never silently "resolve" something.
    learnings_resolved_ids: list[str] = []
    # Set mechanically (hash comparison of risk_model.pkl against EVERY prior episode in
    # this instance, not just the immediately preceding one) -- None if this episode's
    # deliverable is novel. Ground truth, independent of what the model says about it.
    duplicate_of_episode_id: int | None = None

    # The episode-minutes value actually rendered into this episode's "Time Budget"
    # prompt section -- lets a later audit confirm what budget an episode was actually
    # told it had, without relying on config-file history.
    prompt_time_budget_minutes: int | None = None

    def to_index_line(self) -> str:
        # Deliberately shows ONLY val-scored metrics, never holdout -- this text is
        # injected into FUTURE episodes' context (episode_injector.py's SINGLE_PRIOR/
        # ALL_PRIOR modes) and feeds the post-mortem step. Holdout is the harness's sole
        # source of truth for the FINAL reported statistic and must stay that way;
        # repeatedly exposing it as per-iteration feedback the agent adapts on would be
        # a real adaptive-data-analysis / holdout-reuse risk. val.parquet is the sample
        # the agent's own local testing is already tuned against during the episode
        # anyway, so this doesn't newly expose anything the episode's own trace didn't
        # already touch.
        base = (
            f"Episode {self.episode_id:03d} | "
            f"C-stat (val): {self.c_stat_val:.3f} | "
            f"Approach: {self.approach} | "
            f"Hypothesis: {self.hypothesis} | "
            f"Skill: {self.skill_path}"
        )
        if not self.learnings_positive and not self.learnings_negative:
            return base
        if self.duplicate_of_episode_id is not None:
            positive = f"[DUPLICATE of episode {self.duplicate_of_episode_id:04d} -- resubmitted, no code changed] " + \
                ("; ".join(self.learnings_positive) or "none")
        else:
            positive = "; ".join(self.learnings_positive) or "none"
        negative = "; ".join(self.learnings_negative) or "none"
        return f"{base} | Worked: {positive} | Didn't work: {negative}"
