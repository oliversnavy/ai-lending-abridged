from __future__ import annotations
import pathlib
from collections.abc import Callable

import yaml

from agent.app.agent import get_skill_dir, run_episode
from agent.app.clients.llm import get_post_mortem_max_tokens, get_primary_client
from agent.app.models.episode import EpisodeRecord, LoopMode, RunConfig
from agent.app.post_mortem import run_post_mortem

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")
TRACES_DIR = PROJECT_ROOT / "results" / "traces"
LOOP_STATE_DIR = PROJECT_ROOT / "results" / "loop_state"


def _load_loop_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)["loop"]


def _loop_state_path(run_label: str) -> pathlib.Path:
    return LOOP_STATE_DIR / f"run_{run_label}.jsonl"


def _load_loop_state(run_label: str) -> list[EpisodeRecord]:
    """Reconstruct an in-progress instance's episode_index from disk, if one exists --
    the resumability mechanism this function enables is what lets a crash mid-instance
    resume from the last COMPLETED iteration instead of discarding a potentially
    multi-iteration instance's entire progress. Returns [] if no in-progress state
    exists (i.e. this is a fresh instance start, the normal case for the first launch)."""
    path = _loop_state_path(run_label)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.model_validate_json(line))
    return records


def _append_loop_state(run_label: str, record: EpisodeRecord) -> None:
    """Persist one completed iteration (post-mortem already run) immediately -- called
    after EVERY iteration, not just at instance end, so at most one iteration's worth of
    compute is ever at risk to a crash, not the whole instance."""
    LOOP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_loop_state_path(run_label), "a") as f:
        f.write(record.model_dump_json() + "\n")


def _clear_loop_state(run_label: str) -> None:
    """Called once an instance finishes (max_iterations or plateau) -- the NEXT launch
    for this run should start a brand-new instance, not resume the just-finished one."""
    path = _loop_state_path(run_label)
    if path.exists():
        path.unlink()


def _check_plateau(
    episode_index: list[EpisodeRecord],
    window: int,
    threshold: float,
) -> bool:
    if len(episode_index) < window + 1:
        return False
    recent = [ep.c_stat_val for ep in episode_index[-(window + 1):]]
    improvements = [recent[i + 1] - recent[i] for i in range(window)]
    return all(abs(imp) / max(abs(recent[-1]), 1e-9) < threshold for imp in improvements)


def run_loop(
    run_config: RunConfig,
    start_episode_id: int = 0,
    on_iteration_complete: Callable[[int, list[EpisodeRecord], int], None] | None = None,
) -> list[EpisodeRecord]:
    """Run one outer loop (one independent loop instance) for a run config.

    start_episode_id: global episode ID for this instance's first iteration when
    starting FRESH (see resumability note below -- ignored if resuming). Callers running
    multiple independent loop instances MUST reserve non-overlapping ID ranges per
    instance, sized to at least max_iterations, or concurrent/sequential instances will
    collide on the same results/skills/run_{label}/episode_XXXX/ directories.

    RESUMABILITY: a loop instance can run for several iterations at up to the episode
    time/token budget each -- multi-episode, single-process runtime with no prior
    checkpointing otherwise. Each completed iteration is persisted immediately to
    results/loop_state/run_{run_label}.jsonl (see _append_loop_state); on start, if that
    file exists, this function resumes from exactly where it left off (loaded
    episode_index, correct next episode_id derived from the last loaded record) instead
    of starting a fresh empty instance -- the passed-in start_episode_id is only used
    when NOT resuming.

    Does NOT write to results/episode_indexes/ itself -- returns the full in-memory
    iteration history (episode_index) and leaves the caller to decide what represents
    this instance's contribution. agent/run.py uses the LAST iteration only
    (episode_index[-1]), never max()/mean() -- both introduce selection bias or don't
    correspond to a single deployable policy.

    on_iteration_complete: optional callback invoked right after each iteration's record
    is appended to episode_index and persisted to loop_state, before checking for
    plateau. Called with (iteration_num [1-indexed], episode_index [the real,
    still-mutating list -- callers needing a frozen snapshot MUST copy it], start_episode_id).
    A pure extension point, doesn't change this function's own behavior when omitted.
    """
    cfg = _load_loop_config()
    max_iterations: int = cfg["max_iterations"]
    plateau_window: int = cfg["plateau_window"]
    plateau_threshold: float = cfg["plateau_threshold"]

    episode_index: list[EpisodeRecord] = _load_loop_state(run_config.run_label)
    if episode_index:
        start_episode_id = episode_index[0].episode_id
        print(f"[Loop] Resuming run '{run_config.run_label}' instance "
              f"(start_episode_id={start_episode_id}) from iteration {len(episode_index)+1} "
              f"-- {len(episode_index)} iteration(s) already completed", flush=True)

    # This instance's full envelope of episode ids -- used to scope read access
    # (agent/app/middleware/guardrails.py's _build_instance_read_rules) to "this
    # instance's own episodes" rather than every episode this run has ever run.
    # Computed as the full theoretical range up front (not just iterations completed so
    # far) since a not-yet-created directory can't be read regardless of whether it's
    # pre-allowed.
    instance_episode_id_range = (start_episode_id, start_episode_id + max_iterations - 1)

    for iteration in range(len(episode_index), max_iterations):
        print(f"\n[Loop] Run '{run_config.run_label}' | Iteration {iteration + 1}/{max_iterations}", flush=True)

        if run_config.loop_mode == LoopMode.SINGLE_PRIOR:
            injected = episode_index[-1:] if episode_index else []
        elif run_config.loop_mode == LoopMode.ALL_PRIOR:
            injected = episode_index
        else:
            # No loop — run exactly once
            injected = []

        record = run_episode(
            run_config, injected, start_episode_id + iteration,
            instance_episode_id_range=instance_episode_id_range,
        )

        if run_config.use_post_mortem:
            episode_id = start_episode_id + iteration
            skill_dir = get_skill_dir(run_config.run_label, episode_id)
            trace_dir = TRACES_DIR / f"run_{run_config.run_label}"
            # val-scored, NOT holdout -- same reasoning as EpisodeRecord.to_index_line():
            # the post-mortem's output feeds forward into future episodes' context, so its
            # own input can't carry holdout signal without the same adaptive-reuse risk.
            metrics = {"c_stat": record.c_stat_val, "eval_status": record.eval_status}
            print(f"[Loop] Running post-mortem for episode {episode_id:04d}...", flush=True)
            learnings = run_post_mortem(
                llm=get_primary_client(run_config, max_tokens_override=get_post_mortem_max_tokens(),
                                        reasoning_effort=run_config.critic_reasoning_effort),
                skill_dir=skill_dir,
                main_trace_path=trace_dir / f"episode_{episode_id:04d}.jsonl",
                subagent_trace_path=trace_dir / f"episode_{episode_id:04d}_subagents.jsonl",
                metrics=metrics,
                episode_index=episode_index,
            )
            record.learnings_positive = learnings["positive"]
            record.learnings_negative = learnings["negative"]
            record.learnings_negative_ids = [
                f"ep{episode_id:04d}-neg{i}" for i in range(len(learnings["negative"]))
            ]
            record.learnings_resolved_ids = learnings["resolved_ids"]
            record.duplicate_of_episode_id = learnings["duplicate_of_episode_id"]
            status_bits = [f"{len(learnings['positive'])} positive, {len(learnings['negative'])} negative learnings"]
            if learnings["resolved_ids"]:
                status_bits.append(f"resolved {len(learnings['resolved_ids'])} prior open item(s)")
            if learnings["duplicate_of_episode_id"] is not None:
                status_bits.append(f"DUPLICATE of ep{learnings['duplicate_of_episode_id']:04d}")
            print(f"[Loop] Post-mortem: {', '.join(status_bits)}", flush=True)

        episode_index.append(record)
        _append_loop_state(run_config.run_label, record)

        print(f"[Loop] C-stat: {record.c_stat:.3f} | C-stat (val): {record.c_stat_val:.3f} | "
              f"Tokens: {record.tokens_used:,}", flush=True)

        if on_iteration_complete is not None:
            on_iteration_complete(iteration + 1, episode_index, start_episode_id)

        # Single-episode runs always stop after one run
        if run_config.loop_mode == LoopMode.NONE:
            break

        if _check_plateau(episode_index, plateau_window, plateau_threshold):
            print(f"[Loop] Plateau detected after {len(episode_index)} episodes — stopping.", flush=True)
            break

    # Instance is done (max_iterations or plateau) -- clear resumability state so the
    # NEXT launch for this run starts a fresh instance, not resumes this one.
    _clear_loop_state(run_config.run_label)
    return episode_index
