from __future__ import annotations
import json
import logging
import pathlib
import time
from typing import IO

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from .callbacks.subagent_trace import attribute_to_delegations
from .middleware.episode_injector import build_initial_human_message
from .models.episode import EpisodeRecord, RunConfig
from ..graph import build_graph

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")
RESULTS_DIR = PROJECT_ROOT / "results"
TRACES_DIR = RESULTS_DIR / "traces"


def get_skill_dir(run_label: str, episode_id: int) -> pathlib.Path:
    d = RESULTS_DIR / "skills" / f"run_{run_label}" / f"episode_{episode_id:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _serialize_message(msg: AnyMessage, node: str, turn: int) -> dict:
    """Serialise a LangChain message to a trace-friendly dict."""
    entry: dict = {
        "ts": time.strftime("%H:%M:%S"),
        "turn": turn,
        "node": node,
        "type": type(msg).__name__,
    }

    # Content — can be str or list of blocks (thinking model returns list)
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        # Separate thinking blocks from text blocks for readability
        thinking = [b.get("thinking", "") for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
        text = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if thinking:
            entry["thinking"] = "\n".join(thinking)
        entry["content"] = "\n".join(text) if text else content
    else:
        entry["content"] = content

    # Tool calls on AI messages
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = tool_calls

    # Tool message metadata
    name = getattr(msg, "name", None)
    if name:
        entry["name"] = name
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        entry["tool_call_id"] = tool_call_id

    # Token usage
    usage = getattr(msg, "usage_metadata", None)
    if usage:
        entry["usage"] = usage

    # LangChain source tag (e.g. "summarization")
    lc_source = getattr(msg, "additional_kwargs", {}).get("lc_source")
    if lc_source:
        entry["source"] = lc_source

    return entry


def _write_trace(f: IO[str], msg: AnyMessage, node: str, turn: int) -> None:
    try:
        entry = _serialize_message(msg, node, turn)
        f.write(json.dumps(entry, default=str) + "\n")
        f.flush()
    except Exception as exc:
        log.warning("Trace write failed for message type %s: %s", type(msg).__name__, exc)


def _write_subagent_trace(path: pathlib.Path, events: list[dict], run_tree: dict) -> None:
    """Write SubagentTraceCallbackHandler's captured events, grouped by which task()
    delegation (if any) each belongs to. See callbacks/subagent_trace.py for how
    attribution is reconstructed from run_id/parent_run_id -- deepagents' own
    ls_agent_type tag never reaches a plain callback handler, so this doesn't rely on it.

    Always writes the file, even for runs with no subagents (use_deep_agent=False) or
    episodes that never delegated -- everything just lands in the "main" group, harmless
    and consistent with every other run's file layout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        grouped = attribute_to_delegations(events, run_tree)
    except Exception as exc:
        log.warning("Subagent trace attribution failed: %s", exc)
        grouped = {"main": events}

    delegation_ids = [rid for rid in grouped if rid != "main"]
    delegation_label = {rid: f"delegation_{i}" for i, rid in enumerate(delegation_ids)}

    with open(path, "w") as f:
        for rid, group_events in grouped.items():
            label = delegation_label.get(rid, "main")
            for e in group_events:
                f.write(json.dumps({"delegation": label, **e}, default=str) + "\n")


def run_episode(
    run_config: RunConfig,
    episode_index: list[EpisodeRecord],
    episode_id: int,
    system_prompt_override: dict[str, str] | None = None,
    eval_dataset: str | None = None,
    instance_episode_id_range: tuple[int, int] | None = None,
) -> EpisodeRecord:
    """Run one full agent episode and return the logged EpisodeRecord.

    instance_episode_id_range: (lo, hi) inclusive -- which of this run's episode ids the
    agent's read tools (ls/read_file/glob/grep) may access under results/skills and
    results/traces, beyond its own episode_id. Defaults to None, which build_graph
    resolves to (episode_id, episode_id) -- i.e. secure-by-default, solo-episode-only.
    Only agent/loop.py's run_loop() passes a wider range, sized to the whole loop
    instance's own span -- without this, an episode could read another, unrelated
    instance's saved files via an unrestricted recursive glob/ls, undermining the "N
    independent instances/episodes" statistical assumption any reported comparison
    across episodes relies on. See agent/graph.py and
    agent/app/middleware/guardrails.py::_build_instance_read_rules.
    """
    skill_dir = get_skill_dir(run_config.run_label, episode_id)

    initial_message = build_initial_human_message(
        run_config, episode_index, str(skill_dir), episode_id
    )

    graph, callbacks, token_counter, subagent_trace = build_graph(
        run_config, skill_dir, system_prompt_override, episode_index,
        instance_episode_id_range=instance_episode_id_range or (episode_id, episode_id),
    )

    config = {"callbacks": callbacks} if callbacks else {}
    t0 = time.time()
    log.info("[%s ep%04d] Starting episode", run_config.run_label, episode_id)

    trace_path = TRACES_DIR / f"run_{run_config.run_label}" / f"episode_{episode_id:04d}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    turn = 0

    with open(trace_path, "w") as trace_f:
        try:
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=initial_message)]},
                config=config,
                stream_mode="updates",
            ):
                for node, delta in chunk.items():
                    if not delta:
                        continue
                    for msg in delta.get("messages", []):
                        _write_trace(trace_f, msg, node, turn)
                        turn += 1

                        if isinstance(msg, AIMessage):
                            calls = getattr(msg, "tool_calls", [])
                            if calls:
                                for tc in calls:
                                    args_preview = str(tc.get("args", {}))[:120]
                                    log.info("  [model→tool] %s(%s)", tc["name"], args_preview)
                            elif msg.content:
                                preview = str(msg.content)[:200].replace("\n", " ")
                                log.info("  [model→done] %s", preview)
                        elif isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_source") == "summarization":
                            log.info("  [summarize ] context compressed — history trimmed")
                        elif isinstance(msg, ToolMessage):
                            result_preview = str(msg.content)[:120].replace("\n", " ")
                            log.info("  [tool→model] %s → %s", msg.name, result_preview)

        except Exception as e:
            log.error("[%s ep%04d] Episode crashed: %s", run_config.run_label, episode_id, e)

    log.info("[%s ep%04d] Trace written → %s (%d messages)", run_config.run_label, episode_id, trace_path, turn)

    subagent_trace_path = TRACES_DIR / f"run_{run_config.run_label}" / f"episode_{episode_id:04d}_subagents.jsonl"
    _write_subagent_trace(subagent_trace_path, subagent_trace.events, subagent_trace.run_tree)
    log.info(
        "[%s ep%04d] Subagent trace written → %s (%d events)",
        run_config.run_label, episode_id, subagent_trace_path, len(subagent_trace.events),
    )

    duration_s = time.time() - t0
    log.info("[%s ep%04d] Done in %.1fs", run_config.run_label, episode_id, duration_s)

    tokens_used = token_counter.total_tokens

    return _read_results(
        skill_dir, episode_id, run_config.run_label, tokens_used, duration_s,
        eval_dataset=eval_dataset,
        prompt_time_budget_minutes=run_config.max_episode_seconds // 60,
    )


def _read_results(
    skill_dir: pathlib.Path,
    episode_id: int,
    run_label: str,
    tokens_used: int,
    duration_s: float,
    eval_dataset: str | None = None,
    prompt_time_budget_minutes: int | None = None,
) -> EpisodeRecord:
    """
    c_stat is computed independently by the harness (agent/evaluator/evaluate_episode.py)
    from the agent's submitted risk_model.pkl -- never self-reported. Only
    approach/hypothesis are taken from the agent's own results.json, since those are
    free text with no metric to fabricate.
    """
    from ..evaluator.evaluate_episode import evaluate_episode

    metrics = evaluate_episode(skill_dir, dataset=eval_dataset)

    results_path = skill_dir / "results.json"
    if results_path.exists():
        try:
            narrative = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            narrative = {}
    else:
        narrative = {}

    if metrics["eval_status"] == "failed":
        approach = narrative.get(
            "approach", "Agent did not produce a valid risk_model.pkl."
        )
        hypothesis = narrative.get("hypothesis", "N/A")
    else:
        approach = str(narrative.get("approach", ""))
        hypothesis = str(narrative.get("hypothesis", ""))

    return EpisodeRecord(
        episode_id=episode_id,
        run_label=run_label,
        c_stat=float(metrics["c_stat"]),
        # In single-dataset (eval_dataset set) mode, evaluate_episode() doesn't compute
        # c_stat_val at all -- stays at the EpisodeRecord default (0.0).
        c_stat_val=float(metrics["c_stat_val"]) if "c_stat_val" in metrics else 0.0,
        eval_status=metrics["eval_status"],
        approach=approach,
        hypothesis=hypothesis,
        skill_path=str(skill_dir.relative_to(PROJECT_ROOT)),
        tokens_used=tokens_used,
        duration_s=round(duration_s, 1),
        prompt_time_budget_minutes=prompt_time_budget_minutes,
    )
