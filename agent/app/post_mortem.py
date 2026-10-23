"""Post-mortem step -- the "end-of-episode critic" in this harness's three-critic
lifecycle (see also its siblings: preloop_anchor.py the starting critic,
periodic_critic.py the mid-episode critic).

After an episode finishes, a separate, dedicated agent -- fresh context, not a
continuation of the episode's own conversation -- reviews what happened and produces
capped positive/negative learnings that get folded into that episode's EpisodeRecord, for
future episodes' ALL_PRIOR injection to see (agent/app/middleware/episode_injector.py).
This is what makes loop_mode=ALL_PRIOR's cross-episode memory useful rather than just a
raw metric history -- it's the harness's long-term-memory distillation step.

Deliberately built on create_agent (plain LangChain agent), NOT create_deep_agent -- this
agent must not get its own subagent-delegation capability, a second, unrelated axis of
scope creep on top of the one already carefully bounded: this agent's tool-based access
to the episode's own subagent trace stays post-hoc, single-episode, and produces a fixed
distillation -- it must not become a general query-anything agent.

Fresh context, not a continuation, because ContextEditingMiddleware clears old tool
outputs from the OUTGOING REQUEST only, not persisted state -- a continued conversation
risks the model working from an already-incomplete picture of its own recent turns.

Provenance tracking, additive on top of the above. Two mechanisms, both feeding the SAME
single post-mortem call rather than adding a new agent/phase:
  - NEGATIVE-learning provenance: build_open_negatives_digest() mechanically assembles
    every still-unresolved negative learning raised by earlier episodes in this
    instance, tagged with harness-assigned ids (never model-invented). Post-mortem is
    shown this digest and can report which ids this episode's own work directly
    attempted via submit_learnings' resolved_ids field -- the model only ever
    REFERENCES an id it was shown, never has to reconstruct or match free text across
    episodes, which is a much more reliable task than semantic matching would be.
  - POSITIVE-learning / duplication provenance: find_duplicate_episode() is a pure hash
    comparison (risk_model.pkl) against EVERY prior episode in the instance, not just
    the immediately preceding one -- a later episode could reach back and copy a
    non-adjacent prior episode's exact artifact, not just its immediate predecessor.
    This is ground truth independent of what the model says: even if post-mortem's
    prose doesn't mention it, EpisodeRecord.duplicate_of_episode_id records the fact
    mechanically.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool

from agent.app.agent import _serialize_message
from agent.app.middleware.time_awareness import TimeAwarenessMiddleware
from agent.evaluator.pathological_errors import _looks_like_error

log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")

_RETRY_NUDGE = (
    "You did not call submit_learnings. Call it now with your best assessment based on "
    "what you've already reviewed -- do not investigate further, do not explain why, "
    "just call the tool with whatever positive/negative items you can support from what "
    "you've seen so far, even if that's fewer than 3 per category."
)

POST_MORTEM_MAX_SECONDS = 600  # 10 minutes -- a review task, much lighter than the main
                                # episode's own run-configured build-and-train budget.
_MAX_LEARNINGS_PER_CATEGORY = 3  # mechanical backstop -- do not rely on the prompt alone

_POST_MORTEM_SUMMARIZE_TRIGGER = ("tokens", 80_000)
_POST_MORTEM_SUMMARIZE_KEEP = ("tokens", 20_000)
_POST_MORTEM_CONTEXT_EDIT_TRIGGER = 60_000
_POST_MORTEM_CONTEXT_EDIT_KEEP = 5

# Budget for the MAIN graph's trace specifically, leaving headroom for the system
# prompt, deliverable file contents, tool descriptions/results, and the model's own
# response. Nothing structurally caps how long the main graph's conversation can run,
# so this is a real, not theoretical, safety bound.
_MAIN_TRACE_TOKEN_BUDGET = 40_000


def _file_hash(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def find_duplicate_episode(skill_dir: pathlib.Path, episode_index: list) -> int | None:
    """Check whether skill_dir's risk_model.pkl is byte-identical to ANY prior episode in
    this instance -- not just the immediately preceding one, since a later episode could
    reach back and copy a non-adjacent prior episode's exact artifact (episode_injector.py's
    ALL_PRIOR context gives every episode visibility into every earlier one's approach,
    not just the last). Returns the earliest matching prior episode_id, or None if this
    episode's deliverable is novel. Pure hash comparison -- no model judgment involved,
    ground truth independent of what post-mortem says about it.
    """
    model_hash = _file_hash(skill_dir / "risk_model.pkl")
    if model_hash is None:
        return None
    for prior in episode_index:
        prior_dir = PROJECT_ROOT / prior.skill_path
        if _file_hash(prior_dir / "risk_model.pkl") == model_hash:
            return prior.episode_id
    return None


def build_open_negatives_digest(episode_index: list) -> tuple[str, dict[str, str]]:
    """Mechanically assemble every negative learning raised by an earlier episode in this
    instance that no LATER episode's post-mortem has reported as resolved. Returns
    (formatted digest text for the prompt, {id: text} lookup used to validate whatever
    resolved_ids the model reports back -- an id outside this set is dropped rather than
    trusted, so a hallucinated id can never silently mark something resolved).

    Ids are harness-assigned ("ep{episode_id:04d}-neg{index}"), never model-invented --
    the model only ever has to REFERENCE an id it was just shown, never reconstruct or
    semantically match free text across episodes, which would be a much less reliable
    task for it.
    """
    all_ids: dict[str, str] = {}
    raised_at: dict[str, int] = {}
    resolved: set[str] = set()
    for rec in episode_index:
        for item_id, text in zip(rec.learnings_negative_ids, rec.learnings_negative):
            all_ids[item_id] = text
            raised_at[item_id] = rec.episode_id
        resolved.update(rec.learnings_resolved_ids)

    open_ids = [i for i in all_ids if i not in resolved]
    if not open_ids:
        return "(none yet)", {}
    lines = [f"[{i}] (raised in episode {raised_at[i]:04d}, still open) {all_ids[i]}" for i in open_ids]
    return "\n".join(lines), {i: all_ids[i] for i in open_ids}


POST_MORTEM_SYSTEM_PROMPT = """You are reviewing one finished attempt at a credit risk ranking task, after the fact. You
did not do this work yourself — a different agent instance did, working under a time
limit, and this episode is now over. Your job is to extract concrete, reusable lessons
for a FUTURE attempt at the same task, who will see only what you write here (not the
original agent's reasoning, not this trace) before starting its own attempt from scratch.

You are given directly, in this message:
- The final deliverable: risk_model.pkl (reported as present/absent -- it's a binary
  pickle, not something to read as text) and any supporting .py files left in the
  working directory.
- Independently-computed metrics for this episode, scored against a validation sample
  the episode's own agent was already free to use during its work (c-stat) — not
  self-reported by the agent that produced them. These are diagnostic, not the harness's
  official reported figure for this episode (which is scored against a separate,
  held-out sample this agent never had access to, and is deliberately withheld from you
  too — your conclusions get reused across many future episodes, and repeatedly
  reasoning about the same held-out sample's score would let information about that
  specific sample leak into strategy over time, the same failure mode as overfitting to
  a fixed test set).
- The full turn-by-turn trace of the main agent's work (may be truncated if unusually
  long — a truncation notice will say so explicitly if it happened).
- OPEN PRIOR NEGATIVE LEARNINGS: negative items raised by EARLIER episodes in this same
  instance that no later episode has yet reported as addressed, each with an id like
  [ep0047-neg0]. May say "(none yet)" if this is early in the instance or nothing is
  currently open.
- Possibly, a note that this episode's saved risk_model.pkl is byte-identical to a
  specific earlier episode's — see below for what to do with that.

You also have tools for investigating any subagent delegations the main agent made during
the episode (the main agent commonly delegates data exploration, model training, or
debugging to a subagent — its full internal work is NOT included in the main trace above,
only a final summary the main agent itself saw). A single delegation can be very large, so
investigate in stages rather than trying to read everything at once:
- list_delegations(): see what delegations happened and their descriptions.
- list_delegation_events(delegation_id, offset, limit): abbreviated, indexed overview of
  one delegation's events — use this BEFORE reading anything in full, to find which part is
  actually worth reading.
- read_delegation_events(delegation_id, offset, limit): full content for a bounded range of
  a delegation's events (small number at a time, by index) — use only for the specific
  range that looked relevant from the abbreviated list.
- search_for_errors(delegation_id): check whether a delegation hit real errors (tracebacks,
  failed commands) as opposed to ordinary try-fail-fix iteration — often a faster first
  check than reading through events manually.
Use these only if the main trace leaves something unclear that a delegation's internal
work would explain — you do not need to investigate every delegation for every episode.

Produce exactly two lists:

POSITIVE (what worked, worth repeating) — up to 3 items, most important first.
NEGATIVE (what went wrong or cost time/performance, worth avoiding) — up to 3 items, most
important first.

Each item must be a single, concrete, actionable sentence — not a restatement of the
c-stat number, not a vague platitude ("the model could be better" is not useful on its
own; "the model's feature encoding silently zeroed out zip codes not seen in training,
collapsing a chunk of val/holdout rows to identical predictions — check for this with an
nunique count, not just accuracy" is). Prefer items that point at a specific, identifiable
CAUSE over items that just describe the outcome. If nothing genuinely rises to 3 items in
a category, give fewer — do not pad with filler.

Do not report or estimate c-stat yourself — it is given to you as ground truth above;
your job is to explain what produced it, not to recompute it.

If the given metrics look inconsistent with what the trace and files actually show (for
example: the metrics say the episode failed or scored zero, but the trace shows a complete
episode with risk_model.pkl saved and no errors) — say so explicitly as a negative item
instead of inventing a plausible-sounding explanation to reconcile the contradiction. An
unexplained mismatch is itself useful signal; a fabricated cause is not.

For each item in OPEN PRIOR NEGATIVE LEARNINGS: check whether THIS episode's own files or
trace show it was directly attempted (not just mentioned — actually implemented and
tested). If so, include its id in submit_learnings' resolved_ids, and write what actually
happened as one of your own positive or negative items (e.g. "attempted the
censoring-reweighting approach flagged in ep0047-neg0 — improved c-stat by 0.01" or "... —
made predictions worse, reverted"). If an item was NOT attempted this episode, do not
include its id — it stays open for a future episode. Only mark ids you can directly
support from what you observed; do not guess.

If you were told this episode's deliverable is byte-identical to an earlier episode's:
this means no new code was written this iteration — a pure resubmission, not an
independent test. Say so explicitly and plainly in your positive/negative items (e.g.
"episode resubmitted episode 0047's exact solution unchanged — not a new confirmation of
anything") rather than describing it as if it were a fresh, independent success. This
distinction matters more than the c-stat number for anyone reading your output later.

Call submit_learnings(positive=[...], negative=[...], resolved_ids=[...]) as soon as you
have enough information to write concrete, specific items — you do not need to
exhaustively investigate every delegation or explore every angle first. Once you call it,
you are done: do not re-analyze, second-guess, or reconsider your answer afterward, and do
not call it again. If you find yourself wanting to say "wait, let me reconsider" after
already submitting, stop — your first answer stands.

Do NOT draft your positive/negative items as visible text (e.g. a "POSITIVE:" /
"NEGATIVE:" list in your response) before calling the tool. Form your conclusions in your
own reasoning, then go STRAIGHT to calling submit_learnings with them as the tool's
arguments — writing the full answer out twice (once as prose, once as tool args) wastes
your response budget on duplicate work and risks running out of room before the tool
call ever happens. The tool call IS your answer; there is no separate written answer to
produce first.
"""


def _load_bounded_main_trace(trace_path: pathlib.Path, max_tokens: int = _MAIN_TRACE_TOKEN_BUDGET) -> str:
    """Format the main episode trace as readable text, dropping the OLDEST entries first
    if it exceeds max_tokens (same "keep most recent" philosophy already used by
    ContextEditingMiddleware/SummarizationMiddleware elsewhere in this harness) rather than
    assuming every trace fits -- nothing structurally bounds how long a trace can get, so
    this is a real safety check, not a formality.
    """
    if not trace_path.exists():
        return "(no main trace found)"

    entries = []
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def _to_message(entry: dict):
        t = entry.get("type")
        content = entry.get("content", "")
        if t == "HumanMessage":
            return HumanMessage(content=str(content))
        if t == "AIMessage":
            return AIMessage(content=str(content))
        if t == "ToolMessage":
            return ToolMessage(content=str(content), tool_call_id="x")
        return None

    messages = [m for m in (_to_message(e) for e in entries) if m is not None]
    if not messages:
        return "(main trace empty or unparseable)"

    total_tokens = count_tokens_approximately(messages)
    dropped = 0
    while total_tokens > max_tokens and len(messages) > 1:
        messages.pop(0)
        dropped += 1
        total_tokens = count_tokens_approximately(messages)

    lines = []
    if dropped:
        lines.append(f"[TRACE TRUNCATED: {dropped} earliest entries omitted to fit context budget]")
    for entry, msg in zip(entries[dropped:], messages):
        node = entry.get("node", "")
        lines.append(f"[{entry.get('type', '')} node={node}] {msg.content}")
    return "\n".join(lines)


def _build_tools(subagent_trace_path: pathlib.Path) -> list:
    """Tools scoped to ONE episode's subagent trace file -- deliberately bounded, on-demand
    access (list/read/search) rather than dumping all delegations' full content into
    context."""
    if subagent_trace_path.exists():
        raw_events = [json.loads(line) for line in subagent_trace_path.read_text().splitlines() if line.strip()]
    else:
        raw_events = []

    delegations: dict[str, list[dict]] = {}
    for e in raw_events:
        if e.get("delegation") and e["delegation"] != "main":
            delegations.setdefault(e["delegation"], []).append(e)

    @tool
    def list_delegations() -> str:
        """List every subagent delegation made during this episode, with a short
        description of what was delegated. Use this first to see what's available before
        deciding whether to read any of them in full."""
        if not delegations:
            return "No subagent delegations were made during this episode."
        lines = []
        for label, events in delegations.items():
            start = next((e for e in events if e.get("kind") == "tool_start"), None)
            desc = ""
            if start and isinstance(start.get("input"), dict):
                desc = start["input"].get("description", "")
            lines.append(f"{label}: {desc} ({len(events)} events)")
        return "\n".join(lines)

    def _preview(e: dict, max_chars: int = 100) -> str:
        kind = e.get("kind")
        if kind in ("llm_start", "llm_end"):
            body = e.get("content", "")
        elif kind == "tool_start":
            body = f"{e.get('name')}({e.get('input')})"
        else:
            body = e.get("output", "")
        body = str(body).replace("\n", " ")
        return body[:max_chars] + ("..." if len(body) > max_chars else "")

    def _full(e: dict) -> str:
        kind = e.get("kind")
        if kind in ("llm_start", "llm_end"):
            return f"[{kind}] {e.get('content', '')}"
        if kind == "tool_start":
            return f"[tool_start] {e.get('name')}({e.get('input')})"
        return f"[tool_end] {e.get('output')}"

    @tool
    def list_delegation_events(delegation_id: str, offset: int = 0, limit: int = 50) -> str:
        """List a delegation's events as an ABBREVIATED, indexed overview (one truncated
        line per event) -- use this before read_delegation_events to see the shape of a
        delegation's work and decide which range is actually worth reading in full. Bounded
        regardless of delegation size (a single delegation can be very large -- this never
        dumps full content). Paginate with offset/limit if the delegation has more events
        than one call returns."""
        events = delegations.get(delegation_id)
        if events is None:
            return f"No delegation found with id '{delegation_id}'. Call list_delegations() to see valid ids."
        page = events[offset:offset + limit]
        lines = [f"[{offset + i}] {e.get('kind')}: {_preview(e)}" for i, e in enumerate(page)]
        remaining = len(events) - (offset + len(page))
        if remaining > 0:
            lines.append(f"... {remaining} more event(s), call again with offset={offset + limit} to see more")
        return "\n".join(lines) if lines else "(no events)"

    @tool
    def read_delegation_events(delegation_id: str, offset: int = 0, limit: int = 10) -> str:
        """Read FULL content for a bounded range of a delegation's events (default 10 at a
        time), by index -- use list_delegation_events first to find which indices are worth
        reading. Deliberately bounded per call so a single large delegation can't dump
        unbounded content into your context; call again with a later offset for more."""
        events = delegations.get(delegation_id)
        if events is None:
            return f"No delegation found with id '{delegation_id}'. Call list_delegations() to see valid ids."
        page = events[offset:offset + limit]
        if not page:
            return "(no events in this range)"
        lines = [_full(e) for e in page]
        remaining = len(events) - (offset + len(page))
        if remaining > 0:
            lines.append(f"... {remaining} more event(s), call again with offset={offset + limit} to see more")
        return "\n".join(lines)

    @tool
    def search_for_errors(delegation_id: str, max_hits: int = 5) -> str:
        """Check whether a delegation hit genuine errors (tracebacks, failed commands) as
        opposed to ordinary try-fail-fix iteration, using the same error-detection logic
        the harness uses elsewhere (agent/evaluator/pathological_errors.py). Capped at
        max_hits, each truncated -- a delegation with many errors won't dump them all."""
        events = delegations.get(delegation_id)
        if events is None:
            return f"No delegation found with id '{delegation_id}'. Call list_delegations() to see valid ids."
        hits = [
            e.get("output", "") for e in events
            if e.get("kind") == "tool_end" and _looks_like_error(e.get("output", ""))
        ]
        if not hits:
            return "No errors detected in this delegation."
        shown = hits[:max_hits]
        result = f"{len(hits)} error(s) detected (showing {len(shown)}):\n" + "\n---\n".join(str(h)[:500] for h in shown)
        if len(hits) > max_hits:
            result += f"\n... {len(hits) - max_hits} more error(s) not shown"
        return result

    return [list_delegations, list_delegation_events, read_delegation_events, search_for_errors]


@tool
def submit_learnings(positive: list[str], negative: list[str], resolved_ids: list[str]) -> str:
    """Call this exactly once, when you are done reviewing, to submit your final
    positive/negative learnings. resolved_ids: ids from OPEN PRIOR NEGATIVE LEARNINGS
    (e.g. "ep0047-neg0") that THIS episode's own work directly attempted -- pass an
    empty list if none apply or none were shown to you. This is how your conclusions are
    recorded -- do not write them as a plain-text response instead."""
    return "Learnings recorded."


def _read_deliverable_files(skill_dir: pathlib.Path) -> str:
    parts = []
    model_path = skill_dir / "risk_model.pkl"
    parts.append(
        f"=== risk_model.pkl ===\n"
        + (f"(present, {model_path.stat().st_size:,} bytes -- binary, not shown as text)"
           if model_path.exists() else "(missing)")
    )
    supporting = sorted(p.name for p in skill_dir.glob("*.py"))
    if supporting:
        for name in supporting:
            parts.append(f"=== {name} ===\n{(skill_dir / name).read_text()}")
    else:
        parts.append("=== supporting .py files ===\n(none)")
    return "\n\n".join(parts)


def _find_submit_learnings_call(messages: list) -> dict | None:
    """First (not last) SUCCESSFUL submit_learnings tool-call args found, or None.

    First, not last, because the model sometimes calls it correctly then keeps going
    into an unprompted second pass -- a later re-submission must not silently override
    a good first answer.

    "Successful" matters: a tool call can fail structured-arg validation (the model
    passes positive/negative as malformed strings instead of real lists; langgraph's
    ToolNode catches this and sets the resulting ToolMessage's status="error", feeding
    the error back so the model can retry). That first, invalid attempt must not be
    preferred just because it's first. Only a call whose paired ToolMessage is NOT
    status="error" counts as a real submission.
    """
    tool_results_by_id = {
        getattr(msg, "tool_call_id", None): msg
        for msg in messages
        if msg.__class__.__name__ == "ToolMessage"
    }
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") != "submit_learnings":
                continue
            result = tool_results_by_id.get(tc.get("id"))
            if result is not None and getattr(result, "status", "success") == "error":
                continue
            return tc.get("args", {})
    return None


def _write_post_mortem_trace(path: pathlib.Path, messages: list) -> None:
    """Persist the post-mortem agent's own conversation for diagnosability. Reuses
    agent.py's own message serializer for a consistent format with the main/subagent
    traces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            for i, msg in enumerate(messages):
                entry = _serialize_message(msg, "post_mortem", i)
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        log.warning("Failed to write post-mortem trace to %s: %s", path, exc)


def run_post_mortem(
    llm,
    skill_dir: pathlib.Path,
    main_trace_path: pathlib.Path,
    subagent_trace_path: pathlib.Path,
    metrics: dict,
    episode_index: list | None = None,
) -> dict:
    """Run the post-mortem agent for one finished episode. Returns
    {"positive": [...], "negative": [...], "resolved_ids": [...], "duplicate_of_episode_id": int|None},
    positive/negative each capped at _MAX_LEARNINGS_PER_CATEGORY (mechanical backstop, not
    just a prompt instruction).

    episode_index: this instance's prior EpisodeRecords (same list agent/loop.py already
    maintains), used for provenance tracking -- open_negatives_digest/duplicate_of_
    episode_id are both computed from it BEFORE the model call, so they're available even
    if the model call itself fails below (duplicate_of_episode_id in particular is pure
    hash comparison, ground truth independent of the model entirely). Pass None/omit for
    a run with no loop -- degrades to "(none yet)"/None, no behavior change.

    Retries once, in the SAME conversation with an explicit nudge appended, if the model
    doesn't call submit_learnings on the first attempt -- mirrors ResultsGuardMiddleware's
    exact one-retry pattern for the main episode. The failure mode this guards against:
    the model getting stuck in open-ended analysis and never reaching a decision within
    its turn/time budget, producing ZERO submit_learnings calls anywhere in the
    conversation.

    Always persists the full conversation (whichever attempt) to a companion trace file
    next to main_trace_path, regardless of outcome, so any future failure is diagnosable
    without needing to reproduce it separately.

    Never raises on a malformed/missing model response -- a failed post-mortem should
    degrade to empty learnings, not crash the episode that already succeeded/failed on its
    own merits. Logged, not swallowed silently.
    """
    tools = _build_tools(subagent_trace_path) + [submit_learnings]
    time_aware = TimeAwarenessMiddleware(max_seconds=POST_MORTEM_MAX_SECONDS)
    summarizer = SummarizationMiddleware(
        model=llm,
        trigger=_POST_MORTEM_SUMMARIZE_TRIGGER,
        keep=_POST_MORTEM_SUMMARIZE_KEEP,
    )
    context_editor = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(
            trigger=_POST_MORTEM_CONTEXT_EDIT_TRIGGER,
            keep=_POST_MORTEM_CONTEXT_EDIT_KEEP,
        )],
    )
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=POST_MORTEM_SYSTEM_PROMPT,
        # Layered context management: summarizer catches sustained runaway state,
        # context_editor is the closer-to-the-API-call backstop. Neither should fire in
        # the normal case (bounded tool results + a 10-minute cap keep this agent's own
        # tool-calling loop short) -- this exists for the pathological case where it
        # doesn't. time_aware last so it's innermost/closest to the actual call.
        middleware=[summarizer, context_editor, time_aware],
    )

    files_text = _read_deliverable_files(skill_dir)
    main_trace_text = _load_bounded_main_trace(main_trace_path)
    metrics_text = "\n".join(f"{k}: {v}" for k, v in metrics.items())

    # Provenance tracking -- computed eagerly, before the model call, so
    # duplicate_of_episode_id (pure hash comparison) is available in the return value
    # even if the model call below fails entirely.
    episode_index = episode_index or []
    open_negatives_digest, valid_open_ids = build_open_negatives_digest(episode_index)
    duplicate_of_episode_id = find_duplicate_episode(skill_dir, episode_index)

    duplicate_note = ""
    if duplicate_of_episode_id is not None:
        duplicate_note = (
            f"\n\n=== NOTE ===\nThis episode's saved risk_model.pkl "
            f"is BYTE-IDENTICAL to episode {duplicate_of_episode_id:04d}'s -- no new code "
            f"was written this iteration."
        )

    initial_message = (
        f"=== METRICS (ground truth, independently computed) ===\n{metrics_text}\n\n"
        f"{files_text}\n\n"
        f"=== MAIN EPISODE TRACE ===\n{main_trace_text}\n\n"
        f"=== OPEN PRIOR NEGATIVE LEARNINGS (this instance, not yet resolved) ===\n"
        f"{open_negatives_digest}"
        f"{duplicate_note}\n\n"
        "Review the above and produce your positive/negative learnings now."
    )

    # Extract from the submit_learnings TOOL CALL's structured args, not from the model's
    # final text content -- a model will sometimes ignore a "respond in this exact format"
    # text instruction and write plain prose instead, which then fails to parse. Tool-call
    # args are already framework-parsed into real Python types, so this sidesteps that
    # failure mode entirely rather than trying to prompt-engineer around it.
    postmortem_trace_path = main_trace_path.parent / f"{main_trace_path.stem}_postmortem.jsonl"
    messages: list = [HumanMessage(content=initial_message)]
    try:
        result = graph.invoke({"messages": messages})
        messages = result["messages"]
        call_args = _find_submit_learnings_call(messages)

        if call_args is None:
            log.warning("Post-mortem did not call submit_learnings on first attempt, retrying once with an explicit nudge...")
            retry_messages = messages + [HumanMessage(content=_RETRY_NUDGE)]
            result = graph.invoke({"messages": retry_messages})
            messages = result["messages"]
            call_args = _find_submit_learnings_call(messages)

        _write_post_mortem_trace(postmortem_trace_path, messages)

        if call_args is None:
            raise ValueError("model never called submit_learnings, even after retry")
        positive = [str(x) for x in call_args.get("positive", [])][:_MAX_LEARNINGS_PER_CATEGORY]
        negative = [str(x) for x in call_args.get("negative", [])][:_MAX_LEARNINGS_PER_CATEGORY]
        # Only trust ids the model was actually shown -- a hallucinated/unknown id must
        # never silently "resolve" an open item (see build_open_negatives_digest docstring).
        resolved_ids = [str(x) for x in call_args.get("resolved_ids", []) if str(x) in valid_open_ids]
        return {
            "positive": positive, "negative": negative,
            "resolved_ids": resolved_ids, "duplicate_of_episode_id": duplicate_of_episode_id,
        }
    except Exception as exc:
        log.warning("Post-mortem failed, degrading to empty learnings: %s", exc)
        _write_post_mortem_trace(postmortem_trace_path, messages)
        return {
            "positive": [], "negative": [],
            "resolved_ids": [], "duplicate_of_episode_id": duplicate_of_episode_id,
        }
