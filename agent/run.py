#!/usr/bin/env python
"""
CLI entry point for running one of this repo's three illustrative run configs.

Examples:
    uv run python agent/run.py --run-label baseline --n-runs 1
    uv run python agent/run.py --run-label advisor --n-runs 1
    uv run python agent/run.py --run-label ltm_full          # loops until plateau
"""
from __future__ import annotations
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import click

from agent.app.models.episode import RUN_CONFIGS
from agent.loop import run_loop

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results" / "skills"


def _next_episode_id(run_label: str) -> int:
    """Return the next episode ID not yet present on disk."""
    run_dir = RESULTS_DIR / f"run_{run_label}"
    if not run_dir.exists():
        return 0
    existing = sorted(
        int(p.name.split("_")[1])
        for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    return existing[-1] + 1 if existing else 0


@click.command()
@click.option("--run-label", required=True,
              type=click.Choice(list(RUN_CONFIGS.keys())),
              help="Which run config to use -- see RUN_CONFIGS in agent/app/models/episode.py")
@click.option("--n-runs", default=None, type=int,
              help="Episodes to run. No-loop configs default to 1; "
                   "looped configs run until plateau.")
def main(run_label: str, n_runs: int | None) -> None:
    config = RUN_CONFIGS[run_label]

    if config.is_baseline:
        n = n_runs or 1
        start_id = _next_episode_id(run_label)
        print(f"Running '{run_label}' — {n} independent episode(s) (ep {start_id}–{start_id+n-1})")
        from agent.app.agent import run_episode
        index_path = RESULTS_DIR.parent / "episode_indexes" / f"run_{run_label}.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        all_records = []
        for i in range(n):
            episode_id = start_id + i
            print(f"\n=== Run {i + 1}/{n} (ep {episode_id:04d}) ===")
            record = run_episode(config, [], episode_id)
            all_records.append(record)
            with open(index_path, "a") as f:
                f.write(record.model_dump_json() + "\n")
            print(f"  C-stat: {record.c_stat:.3f} | Tokens: {record.tokens_used:,}")
    else:
        if n_runs:
            print(f"Note: --n-runs={n_runs} noted (loop uses plateau detection from configs/base.yaml)")
        start_id = _next_episode_id(run_label)
        loop_records = run_loop(config, start_episode_id=start_id)
        # Only the LAST iteration represents this loop instance's contribution to any
        # reported sample across instances -- max()/mean() both introduce selection bias
        # or don't correspond to a single deployable policy.
        all_records = loop_records[-1:]
        if all_records:
            index_path = RESULTS_DIR.parent / "episode_indexes" / f"run_{run_label}.jsonl"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "a") as f:
                f.write(all_records[0].model_dump_json() + "\n")

    c_stats = [r.c_stat for r in all_records]
    if c_stats:
        print(f"\n{'='*50}")
        print(f"Run '{run_label}' complete — {len(c_stats)} episode(s)")
        if len(c_stats) > 1:
            print(f"  C-stat  mean: {statistics.mean(c_stats):.3f}  "
                  f"stdev: {statistics.stdev(c_stats):.3f}")
        else:
            print(f"  C-stat: {c_stats[0]:.3f}")


if __name__ == "__main__":
    main()
