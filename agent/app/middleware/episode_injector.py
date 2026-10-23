from __future__ import annotations
from ..models.episode import EpisodeRecord, LoopMode, RunConfig


def build_initial_human_message(
    run_config: RunConfig,
    episode_index: list[EpisodeRecord],
    skill_dir: str,
    episode_id: int,
) -> str:
    """Build the opening human message injected at the start of each episode."""
    header = f"Working directory: {skill_dir}\nEpisode: {episode_id:03d}"

    if run_config.loop_mode == LoopMode.NONE or not episode_index:
        context = "No prior episodes. Start fresh."
    elif run_config.loop_mode == LoopMode.SINGLE_PRIOR:
        last = episode_index[-1]
        context = (
            "Prior episode (single-prior mode — only the most recent result):\n"
            + last.to_index_line()
            + "\n\nBuild on this. Identify one thing to change or improve."
        )
    else:  # ALL_PRIOR
        lines = "\n".join(ep.to_index_line() for ep in episode_index)
        context = (
            f"Prior episodes ({len(episode_index)} total — all-prior mode):\n"
            + lines
            + "\n\nReview the full history. Identify patterns and what to try next."
        )

    return f"{header}\n\n{context}\n\nBegin."
