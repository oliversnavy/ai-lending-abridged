from __future__ import annotations
import pathlib

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit, ContextEditingMiddleware
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgent

from agent.app.backends.sandboxed_shell import SandboxedLocalShellBackend
from agent.app.callbacks.subagent_trace import SubagentTraceCallbackHandler
from agent.app.callbacks.token_counter import TokenCountingCallbackHandler
from agent.app.clients.llm import get_primary_client
from agent.app.models.episode import EpisodeRecord, RunConfig
from agent.app.prompts.system import get_system_prompt
from agent.app.middleware.guardrails import HarnessGuardrailsMiddleware
from agent.app.middleware.learnings_reinjection import LearningsReinjectionMiddleware, build_learnings_reminder
from agent.app.middleware.overflow_recovery import OverflowRecoveryMiddleware
from agent.app.middleware.periodic_critic import PeriodicCriticMiddleware
from agent.app.middleware.preloop_anchor import PreLoopAnchorMiddleware
from agent.app.middleware.results_guard import ResultsGuardMiddleware
from agent.app.middleware.subagent_time_awareness import SubagentTimeAwarenessMiddleware
from agent.app.middleware.time_awareness import TimeAwarenessMiddleware
from agent.app.middleware.token_awareness import TokenAwarenessMiddleware
from agent.app.middleware.truncation_recovery import TruncationRecoveryMiddleware
from agent.app.tools import get_t0_tools, get_supplementary_tools

# Context management constants for the vanilla (use_deep_agent=False) path.
# ContextEditingMiddleware's trigger assumes a conservative worst-case ~2.5x undercount
# by the approximate token counter against this model's real usable context -- see the
# deep-agent constants below for the full derivation; both paths share the same model
# and the same real ceiling, so there's nothing path-specific about the number itself.
_T0_SUMMARIZE_TRIGGER   = ("tokens", 45_000)
_T0_SUMMARIZE_KEEP      = ("tokens", 20_000)
_T0_CONTEXT_EDIT_TRIGGER = 40_000
_T0_CONTEXT_EDIT_KEEP    = 5

# Context management constants for the deep-agent (use_deep_agent=True) path.
# Trigger derivation: an approximate-count trigger needs real margin below the model's
# actual usable context to guarantee firing before a hard rejection, given how much a
# fast approximate counter can undercount dense numeric tool output relative to the
# real tokenizer (confirmed via live instrumentation in the source research project, at
# up to ~2.5x for this kind of workload) -- 40,000 is a conservative choice with that
# margin built in, not tuned to any specific model's exact context window.
_DEEP_AGENT_CONTEXT_EDIT_TRIGGER = 40_000
_DEEP_AGENT_CONTEXT_EDIT_KEEP    = 5

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")

# --- Subagent prompt structure -------------------------------------------------------


def _subagent_prompt_prefix(skill_dir: pathlib.Path) -> str:
    return (
        f"You are a Python/data-science assistant helping with a credit risk ranking task.\n\n"
        f"Key paths (always use absolute paths in code):\n"
        f"  Data (read-only):  {PROJECT_ROOT / 'data' / 'processed'}/\n"
        f"    train.parquet, val.parquet, holdout.parquet\n"
        f"    val_features_only.parquet, holdout_features_only.parquet (event/observed_time/\n"
        f"      is_censored removed -- this is what the harness re-scores your saved artifact\n"
        f"      against, not the files above)\n"
        f"  Working directory: {skill_dir}/\n\n"
        f"DELIVERABLE (if you're asked to build/finalize a model): the episode's output is ONE\n"
        f"file saved to the working directory, independently reloaded and re-scored by a separate\n"
        f"harness process after the episode ends -- nothing you compute/print yourself is used for\n"
        f"the final c_stat:\n"
        f"  risk_model.pkl -- pickled object exposing predict_default_proba(X: pd.DataFrame)\n"
        f"                    -> np.ndarray (called ONCE on the full dataset)\n"
        f"Pickling gotcha: if you define your model class in a script run directly, Python records its\n"
        f"module as __main__ and the harness's separate process can't unpickle it. Put the class in its\n"
        f"own file and import it before pickling."
    )


_SUBAGENT_PROMPT_GUIDANCE = (
    "Work efficiently: make sure you understand the full scope of what's being asked "
    "before writing code, and verify your output (load it back, sanity-check shapes/"
    "values) before reporting back to the main agent."
)

_SUBAGENT_PROMPT_SUFFIX = (
    "To run Python: write code with write_file, then:\n"
    "  execute(command='python3 /absolute/path/to/script.py')\n"
    "Never pass code= or filename= to execute — only command= is accepted."
)


def _default_subagent_prompt(skill_dir: pathlib.Path) -> str:
    return "\n\n".join([_subagent_prompt_prefix(skill_dir), _SUBAGENT_PROMPT_GUIDANCE, _SUBAGENT_PROMPT_SUFFIX])


def _resolve_system_prompt(rc: RunConfig, override: dict[str, str] | None) -> str:
    if override and "main" in override:
        return override["main"]
    return get_system_prompt(rc.use_advisor, rc.use_deep_agent, rc.use_periodic_critic,
                              episode_minutes=rc.max_episode_seconds // 60)


def _resolve_subagent_prompt(rc: RunConfig, skill_dir: pathlib.Path, override: dict[str, str] | None) -> str:
    if override and "subagent" in override:
        middle = override["subagent"]
    else:
        middle = _SUBAGENT_PROMPT_GUIDANCE
    return "\n\n".join([_subagent_prompt_prefix(skill_dir), middle, _SUBAGENT_PROMPT_SUFFIX])


def build_graph(
    run_config: RunConfig,
    skill_dir: pathlib.Path,
    system_prompt_override: dict[str, str] | None = None,
    episode_index: list[EpisodeRecord] | None = None,
    instance_episode_id_range: tuple[int, int] | None = None,
):
    """Return (compiled_graph, callbacks, token_counter, subagent_trace) for one episode.

    instance_episode_id_range: always resolved (non-None) by the time it reaches here --
    agent/app/agent.py::run_episode defaults it to (episode_id, episode_id) before
    calling build_graph, so this function never has to guess. Passed to
    HarnessGuardrailsMiddleware (deep-agent path only) to scope ls/read_file/glob/
    grep access -- see that module for why deepagents' own native permissions= mechanism
    can't be used here (it raises NotImplementedError on any execute-capable backend,
    which SandboxedLocalShellBackend is). A run with no loop never widens this -- it
    correctly gets the (episode_id, episode_id) solo default, meaning no OTHER episode is
    readable at all, not even one from the same run_label. Only agent/loop.py explicitly
    widens it to a real instance's full span. Not used by the vanilla path at all -- that
    path doesn't use deepagents' FilesystemMiddleware, it has its own, separate read
    restriction in agent/app/tools/filesystem.py::build_filesystem_tools.

    episode_index: this loop instance's prior EpisodeRecords, needed only when
    run_config.use_learnings_reinjection is set -- build_learnings_reminder() derives
    the reinjection middleware's fixed reminder text from it.

    token_counter is a TokenCountingCallbackHandler included in `callbacks` — return it
    separately too so callers can read its accumulated total after the episode finishes.
    LangChain callbacks propagate into deepagents subagent invocations automatically
    (langgraph config merging), so this sees subagent-internal token usage that the main
    graph's own final message list never does — see agent/app/callbacks/token_counter.py.

    subagent_trace is a SubagentTraceCallbackHandler, same propagation guarantee as
    token_counter but persisting actual call content instead of a running total — see
    agent/app/callbacks/subagent_trace.py for why attribution is done via run_id/
    parent_run_id reconstruction rather than deepagents' own ls_agent_type tag (that tag
    never reaches a plain callback handler -- it's LangSmith-tracing-only).
    """
    llm = get_primary_client(run_config, reasoning_effort=run_config.reasoning_effort)
    system_prompt = _resolve_system_prompt(run_config, system_prompt_override)
    token_counter = TokenCountingCallbackHandler()
    subagent_trace = SubagentTraceCallbackHandler()
    callbacks = [token_counter, subagent_trace]
    # LangSmith tracing (if LANGCHAIN_TRACING_V2/LANGSMITH_API_KEY are set in the
    # environment) is picked up automatically by LangChain's own callback system --
    # nothing to wire in here. See ARCHITECTURE.md.

    # "" (build_learnings_reminder's empty-input behavior) makes LearningsReinjectionMiddleware
    # a no-op, and PeriodicCriticMiddleware/PreLoopAnchorMiddleware just see "(none yet)" as
    # their past-learnings context, so this is safe to compute unconditionally.
    learnings_reminder = (
        build_learnings_reminder(episode_index or [])
        if run_config.use_learnings_reinjection or run_config.use_periodic_critic
        or run_config.use_preloop_anchor
        else ""
    )

    if not run_config.use_deep_agent:
        # Vanilla create_agent — our full custom tool set, no batteries.
        # SummarizationMiddleware is the one exception: without it the growing tool-call
        # history overflows the context window mid-episode. deepagents includes this by
        # default for the deep-agent path; we add it manually here so both paths get
        # context management and stay comparable.
        tools = get_t0_tools(run_config, skill_dir)
        summarizer = SummarizationMiddleware(
            model=llm,
            trigger=_T0_SUMMARIZE_TRIGGER,
            keep=_T0_SUMMARIZE_KEEP,
        )
        context_editor = ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(
                trigger=_T0_CONTEXT_EDIT_TRIGGER,
                keep=_T0_CONTEXT_EDIT_KEEP,
            )],
        )
        overflow_recovery = OverflowRecoveryMiddleware()
        time_aware = TimeAwarenessMiddleware(max_seconds=run_config.max_episode_seconds)
        token_aware = TokenAwarenessMiddleware(token_counter=token_counter, max_tokens=run_config.max_episode_tokens)
        # A safety net: one recovery-prompt retry if the model ends the episode without
        # saving its deliverable, even though it did real, working training earlier in
        # the same episode. Not a cognitive capability like a critic or delegation, just
        # one more chance to save work already done.
        results_guard = ResultsGuardMiddleware(skill_dir=skill_dir, max_retries=2)
        # Catches the more specific "last turn got cut off by the per-turn token cap"
        # failure before results_guard's broader "is the artifact missing/broken" check
        # runs -- must be ordered BEFORE results_guard: each middleware's after_agent is
        # its own chained graph node, and returning jump_to="model" short-circuits that
        # chain for this pass, so whichever of the two fires first is the one that acts.
        truncation_recovery = TruncationRecoveryMiddleware(max_retries=2)
        # Middleware ordering matters for wrap_model_call: last = innermost wrapper.
        # overflow_recovery is last so it sits closest to the actual API call and
        # catches real context-overflow errors before context_editor or summarizer see them.
        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[summarizer, context_editor, time_aware, token_aware, truncation_recovery, results_guard, overflow_recovery],
        )
    else:
        # Deep agent: create_deep_agent with batteries on + our supplementary tools
        tools = get_supplementary_tools(run_config)
        # create_deep_agent() unconditionally injects its own SummarizationMiddleware via
        # deepagents.middleware.summarization.compute_summarization_defaults(). That
        # function only picks a sane fraction-based trigger when the model has LangChain
        # profile metadata exposing max_input_tokens; without it, it falls back to a fixed
        # trigger that can sit above a model's real usable context, making that fallback
        # permanently inert. ContextEditingMiddleware is the PROACTIVE half of what
        # protects the deep-agent path (see overflow_recovery below for the reactive
        # half). There's no public create_deep_agent() hook to exclude the auto-injected
        # default, so this adds a stateless, request-only backstop on top of it.
        # Built once here and reused for BOTH the main agent (below) and the
        # general-purpose subagent (via its own "middleware" spec key) -- the
        # auto-generated default subagent gets the same inert fallback summarizer and
        # nothing else, so a delegated subagent would otherwise have zero effective
        # context protection despite sharing the same model and the same real ceiling.
        context_editor = ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(
                trigger=_DEEP_AGENT_CONTEXT_EDIT_TRIGGER,
                keep=_DEEP_AGENT_CONTEXT_EDIT_KEEP,
            )],
        )
        # Reactive backstop: fires only on the actual, authoritative API rejection (not
        # an estimate), so it's correct by construction regardless of how badly the
        # approximate counter above undercounts -- progressively clears the oldest tool
        # result and retries until the request fits. One shared, stateless instance for
        # both the main agent and subagent, same reuse pattern as context_editor above.
        # Must be the LAST wrap_model_call middleware in each list (innermost, closest
        # to the real API call).
        overflow_recovery = OverflowRecoveryMiddleware()
        # Gives the subagent a soft time signal of its own, parsed from the delegating
        # task() call's description (see the "state a time budget" instruction in
        # SUBAGENT_GUIDANCE), closing the gap where TimeAwarenessMiddleware never fires
        # while the main graph is blocked awaiting a subagent.
        subagent_time_aware = SubagentTimeAwarenessMiddleware()
        # Built once here and reused for BOTH the main agent (below) and the
        # general-purpose subagent -- stateless (just resolved skill_dir/data_dir path
        # checks), so one instance is safe to reuse. Added to the subagent because the
        # Docker sandbox (SandboxedLocalShellBackend, below) only wraps execute() --
        # write_file/edit_file are inherited unchanged from FilesystemBackend and run as
        # direct, unsandboxed host filesystem I/O. This middleware's _check_write_path is
        # the ONLY thing restricting those two tools to skill_dir.
        guardrails = HarnessGuardrailsMiddleware(
            skill_dir=skill_dir,
            data_dir=PROJECT_ROOT / "data" / "processed",
            run_label=run_config.run_label,
            instance_episode_id_range=instance_episode_id_range,
        )
        # Deliberately a SEPARATE instance from the main graph's: deepagents subagents
        # don't inherit the main agent's message history ("messages" is in deepagents'
        # _EXCLUDED_STATE_KEYS), so this needs its own before_model hook and its own
        # state to scan against the subagent's own message thread.
        subagent_middleware: list = [guardrails, context_editor, overflow_recovery, subagent_time_aware]
        if run_config.use_learnings_reinjection and learnings_reminder:
            subagent_middleware.append(LearningsReinjectionMiddleware(learnings_reminder))
        subagent: SubAgent = {
            "name": "general-purpose",
            "description": (
                "General-purpose Python/data-science assistant. "
                "Use for data exploration, model training, evaluation, or any "
                "task that benefits from running code in a subprocess."
            ),
            # No per-subagent model override -- inherits the main agent's model (the
            # default when "model" is omitted from a SubAgent spec).
            "middleware": subagent_middleware,
            "system_prompt": _resolve_subagent_prompt(run_config, skill_dir, system_prompt_override),
        }
        # SandboxedLocalShellBackend gives the agent real filesystem access and a working
        # execute() tool, but execute() runs inside an ephemeral Docker container:
        # PROJECT_ROOT is bind-mounted read-only, skill_dir read-write on top, --network
        # none. See sandboxed_shell.py for the full incident writeup and design rationale.
        # 180s execute timeout: forces the agent toward faster approaches (LR, sampling)
        # rather than burning time on a full-dataset GBM/grid-search that repeatedly times out.
        backend = SandboxedLocalShellBackend(
            root_dir=str(skill_dir),
            virtual_mode=False,
            inherit_env=True,
            timeout=180,
            project_root=PROJECT_ROOT,
            run_label=run_config.run_label,
            instance_episode_id_range=instance_episode_id_range,
        )
        results_guard = ResultsGuardMiddleware(skill_dir=skill_dir, max_retries=2)
        # See the vanilla path above (and truncation_recovery.py's module docstring) for
        # why this must be ordered BEFORE results_guard.
        truncation_recovery = TruncationRecoveryMiddleware(max_retries=2)
        time_aware = TimeAwarenessMiddleware(max_seconds=run_config.max_episode_seconds)
        token_aware = TokenAwarenessMiddleware(token_counter=token_counter, max_tokens=run_config.max_episode_tokens)
        # use_learnings_reinjection and use_periodic_critic are mutually exclusive by
        # RunConfig construction (see episode.py) -- before_model ordering relative to
        # the other middleware here doesn't matter (none of guardrails/results_guard/
        # time_aware/token_aware conflict with either), they just need to precede
        # context_editor so context_editor stays last/innermost.
        main_middleware: list = [guardrails, truncation_recovery, results_guard, time_aware, token_aware]
        if run_config.use_learnings_reinjection and learnings_reminder:
            main_middleware.append(LearningsReinjectionMiddleware(learnings_reminder))
        elif run_config.use_periodic_critic:
            critic_trace_path = (
                PROJECT_ROOT / "results" / "traces" / f"run_{run_config.run_label}"
                / f"{skill_dir.name}_critic.jsonl"
            )
            critic_llm = get_primary_client(run_config, reasoning_effort=run_config.critic_reasoning_effort)
            main_middleware.append(PeriodicCriticMiddleware(critic_llm, learnings_reminder, critic_trace_path))
        if run_config.use_preloop_anchor and learnings_reminder:
            # See preloop_anchor.py's module docstring for the adherence-gap finding
            # this addresses and why it's deliberately NOT paired with a fall-back-to-
            # known-good instruction. Independent of (not mutually exclusive with)
            # use_periodic_critic above -- both are True in the ltm_full config, this
            # just fires once, first.
            anchor_trace_path = (
                PROJECT_ROOT / "results" / "traces" / f"run_{run_config.run_label}"
                / f"{skill_dir.name}_anchor.jsonl"
            )
            # Reuses the critic's client (built above) when available, same effort-level
            # reasoning as periodic_critic's own choice -- this is the same kind of
            # bounded, infrequent, deliberation-worth-the-latency call.
            anchor_llm = critic_llm if run_config.use_periodic_critic else get_primary_client(
                run_config, reasoning_effort=run_config.critic_reasoning_effort,
            )
            main_middleware.append(PreLoopAnchorMiddleware(anchor_llm, learnings_reminder, anchor_trace_path))
        main_middleware.append(context_editor)
        main_middleware.append(overflow_recovery)
        # context_editor then overflow_recovery (both built above, before the subagent
        # spec) are positioned last here, in that order, so overflow_recovery ends up
        # innermost -- closest to the actual API call, catching the real exception after
        # context_editor's proactive (but approximate-counter-based, unreliable near the
        # real ceiling) clearing has already had its chance.
        graph = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            subagents=[subagent],
            backend=backend,
            middleware=main_middleware,
        )

    return graph, callbacks, token_counter, subagent_trace
