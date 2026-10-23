# Architecture

This document explains the design decisions behind the harness, not just what the code
does. See `README.md` for quick start and the task definition.

## Episode lifecycle

```
                      ┌─────────────────────────────────────────┐
                      │  PreLoopAnchorMiddleware (starting critic)│
                      │  fires once, before turn 1, if prior      │
                      │  learnings exist                           │
                      └───────────────────┬─────────────────────┘
                                           ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  Main agent loop (LangGraph, deepagents)                        │
   │  - delegates to a sandboxed "general-purpose" subagent for      │
   │    data exploration / model training / debugging                │
   │  - PeriodicCriticMiddleware fires every N turns (mid-episode     │
   │    critic), advisory, trace-aware                                │
   │  - guardrails / time-and-token-awareness / overflow & truncation │
   │    recovery run throughout                                       │
   └───────────────────────────┬───────────────────────────────────┘
                                ▼
                      risk_model.pkl saved to skill_dir
                                ▼
          ┌──────────────────────────────────────────┐
          │  ResultsGuardMiddleware: verify artifact   │
          │  exists, smoke-test it, check for a         │
          │  degenerate prediction distribution — retry │
          │  (jump back into the loop) if any fail       │
          └──────────────────┬───────────────────────┘
                              ▼
          ┌──────────────────────────────────────────┐
          │  Independent evaluator (separate process):│
          │  score risk_model.pkl against val, then     │
          │  holdout — Harrell's C, never self-reported │
          └──────────────────┬───────────────────────┘
                              ▼
          ┌──────────────────────────────────────────┐
          │  post_mortem.py (end-of-episode critic):   │
          │  fresh-context review of the finished       │
          │  episode → capped positive/negative          │
          │  learnings, folded into this episode's       │
          │  record for the NEXT episode to see           │
          └──────────────────────────────────────────┘
```

A loop instance (`agent/loop.py`) repeats this with `loop_mode=ALL_PRIOR`: every
episode's initial context includes every prior episode's distilled record
(`EpisodeRecord.to_index_line()`), and the loop checks for a plateau in the diagnostic
(val) c-stat before stopping.

## Multi-agent patterns and state management

**Main agent / subagent split.** The main agent delegates data exploration, model
training, and debugging to a "general-purpose" subagent running in a sandboxed shell
(`agent/app/backends/sandboxed_shell.py`) — an ephemeral Docker container per `execute()`
call, project data mounted read-only, the episode's own working directory mounted
read-write, `--network none`. This replaced a plain, unsandboxed shell backend after a
real incident in the source research project: an agent hit an import error on a shared
project module and "fixed" it by overwriting the real file via a shell redirect — a
guardrail middleware that only pattern-matches dangerous commands (`rm -rf`, `curl`,
`sudo`, ...) never caught an absolute-path `cat > ... << EOF`, because output-redirection
targets aren't a command-blocklist problem, they're a filesystem-boundary problem. Docker
closes that structurally instead of trying to out-regex it.

**deepagents subagents don't inherit the main agent's message history** — `"messages"`
is excluded from the state deepagents passes into a delegation. Any middleware that
needs to track conversation-scoped state (the learnings-reinjection pattern below,
in-episode time awareness) needs its own instance for the subagent, with its own
tracking, not a shared one.

**Short-term memory** (within one episode): LangGraph's own message state, managed by
`ContextEditingMiddleware` (proactively clears old tool results once a token threshold
is crossed) and `OverflowRecoveryMiddleware` (reactively retries on an actual context-
length rejection — the authoritative backstop, since an approximate token counter can
meaningfully undercount dense numeric tool output relative to the real tokenizer).

**Long-term memory** (across episodes): not a vector store or RAG — distilled,
capped text. `post_mortem.py` reviews a finished episode and produces up to 3 positive
and 3 negative learnings, each tagged with a mechanical, harness-assigned id. A later
episode's post-mortem can reference a specific still-open negative learning's id to
mark it resolved — the model only ever *references* an id it was shown, never
reconstructs or semantically matches free text across episodes, which would be a much
less reliable task. A pure hash comparison (independent of what the model says)
flags byte-identical resubmissions across non-adjacent episodes, so an agent silently
re-submitting a known-good prior artifact can't be mistaken for a fresh, independent
result.

## The three-critic lifecycle

Three critics intervene at three different points in an episode's life, each solving a
different problem:

1. **Starting critic** (`preloop_anchor.py`, `PreLoopAnchorMiddleware`) — fires once,
   before the agent's first turn. Surfaces the single strongest prior approach from past
   learnings as a *non-binding suggested starting point*. Built to address a specific,
   measured adherence gap: agents given concrete, actionable prior-approach text
   (already present in their own injected context) consistently re-derived everything
   from scratch instead of building on it. Deliberately **not** paired with an
   instruction to fall back to the anchored approach if the agent's own attempt doesn't
   beat it — that would reintroduce a ratchet/selection-bias confound (submit a
   known-good prior result whenever your own attempt looks worse), which would make any
   measured improvement ambiguous between "the agent engaged with the memory" and "the
   harness quietly swapped in a safety net."

2. **Mid-episode critic** (`periodic_critic.py`, `PeriodicCriticMiddleware`) — fires
   every N turns during the episode. A bounded, single LLM call reviews a trace excerpt
   of the *live* episode plus past learnings, and injects one observation + one
   advisory (never imperative) suggestion. This replaced an earlier design,
   `learnings_reinjection.py` (`LearningsReinjectionMiddleware`), kept in this repo
   specifically as a documented, tried-and-rejected alternative: re-emphasizing the same
   static learnings text on *every* turn, imperatively ("Keep doing: X — APPLY this
   now"), caused several consecutive episodes to produce byte-identical artifacts — the
   agent stopped exploring and just copied the recipe. Lower frequency + dynamic,
   trace-aware, advisory phrasing was built to get adherence without collapsing
   exploration into copying.

3. **End-of-episode critic** (`post_mortem.py`) — runs once, after the episode ends, in
   a *fresh* context (not a continuation of the episode's own conversation — context-
   editing clears old tool outputs from the outgoing request only, not persisted state,
   so a continued conversation risks working from an already-incomplete picture of its
   own recent turns). Reviews the finished trace, the saved artifact, and independently-
   computed diagnostic metrics, and produces the capped learnings that feed the next
   episode's starting and mid-episode critics. This is what makes the long-term-memory
   loop a *distillation* loop rather than just a growing pile of raw history.

## Eval methodology for a non-deterministic system

- **Harrell's C, not AUC.** The val/holdout sets contain censored loans (still paying at
  the data snapshot — not yet confirmed non-default). `sklearn.roc_auc_score` treats
  "hasn't defaulted yet" as "confirmed good," which is wrong; `lifelines`'
  `concordance_index` only compares pairs where the earlier-ending observation is a real
  default, correctly handling censoring by construction.
- **Independent, subprocess-isolated re-scoring.** The agent's own self-reported numbers
  are never authoritative. `agent/evaluator/score_runner.py` runs in its own subprocess,
  sees only a features-only file (outcome columns physically absent), and is the *only*
  place the agent's unpickled model ever executes — it never shares a process with
  ground truth or with the trusted harness code that computes the final metric.
- **val vs. holdout, not just train vs. test.** val is the sample the agent's own local
  testing is tuned against *during* the episode; holdout is touched by nothing the agent
  does. Reporting the val number as the headline result would be its own, non-leakage
  source of overfitting (repeatedly testing strategies against the same finite sample) —
  distinct from, and in addition to, the outcome-leakage problem the train/val/holdout
  column exclusions address.
- **Degenerate-model detection as a first-class check**, not just a low score.
  `risk_model_check.py` + `ResultsGuardMiddleware` check a model's prediction
  distribution's *shape* (how many distinct values it produces on a real population),
  not just its accuracy — a model that collapsed to a near-constant output is a bug, and
  is caught and surfaced to the agent *during* the episode, not discovered only after
  the fact.
- **A `__main__` pickling gotcha, caught with a specific fix, not a generic retry.** A
  model class defined in a script the agent ran directly gets recorded with module
  `__main__`; the harness's separate scoring process has its own distinct `__main__`,
  so it can't unpickle the object. `ResultsGuardMiddleware` detects this exact traceback
  signature and injects the specific, proven fix rather than a generic "something's
  wrong" nudge.

## Orchestration patterns

Middleware ordering matters and is deliberate: `agent/graph.py`'s `wrap_model_call`
chain is built so request-shrinking (`ContextEditingMiddleware`) runs before the
reactive overflow backstop (`OverflowRecoveryMiddleware`), which sits innermost, closest
to the actual API call. `results_guard`/`truncation_recovery`/`time_aware`/`token_aware`
are all `before_model`/`after_agent` hooks that don't conflict with each other, but
`truncation_recovery` is deliberately checked before `results_guard` — both return
`jump_to="model"` to restart the loop, so whichever fires first is the one that acts,
and a token-cutoff is a more specific, more likely diagnosis than a generic "artifact
missing" check.

`RUN_CONFIGS`' three entries (`baseline`/`advisor`/`ltm_full`) are themselves an
orchestration-pattern demonstration: the same `build_graph()` function composes a
materially different agent depending on a handful of boolean/enum fields on `RunConfig`,
without any of the underlying middleware needing to know which combination is active.

## Model backends

`agent/app/clients/llm.py` supports two hosted backends:

- **Anthropic** (default) — so anyone can run this with just an API key.
- **Baseten** (`backend="baseten"`) — an OpenAI-compatible endpoint serving an
  open-weight model (GLM-5.3). Kept as a second path deliberately: real deployment work
  often has to accommodate a customer's model choice, including self-hostable/open-weight
  models for cost or compliance reasons, not just whatever's most convenient to build
  against first.

The source research project also runs this harness against a self-hosted vLLM server on
local GPU hardware — that path isn't included here (no GPU box for a reviewer to run
it against), but the backend abstraction is built to make adding a third option
mechanical: implement `get_primary_client`'s branch, point it at a base URL.

## Observability

LangSmith tracing is wired in by *not writing any code for it* — if
`LANGCHAIN_TRACING_V2` and `LANGSMITH_API_KEY` are set in the environment, LangChain's
own callback system picks them up automatically. The source research project instead
had a dead Langfuse integration (configured, never actually emitting usable traces) —
swapped for the thing LangChain's own tooling does natively.

## What was deliberately cut, and why

This repo is a fork-and-strip of a larger private research project. Cut, specifically:

- **The sequential, capital-constrained pricing mechanic.** The real project's task
  isn't just risk ranking — it's pricing a loan under a monthly capital constraint,
  deciding in real time whether to fund an applicant now or hold capital back for a
  possibly-better one later this month. That mechanic, and the empirical findings about
  it, are the actual subject of a forthcoming paper and aren't included here.
- **GEPA-style automated prompt evolution.** The real project has a separate
  trace-mining-driven prompt-optimization loop that evolves the agent's *own system
  prompt* across many episodes. It's not included here because the three-critic
  lifecycle above already demonstrates trace-mining-driven continuous improvement
  (mining a finished episode's trace to produce learnings that improve the *next*
  episode) without needing a second, heavier mechanism that targets prompt text itself
  to tell that story.
- **Survival-model-based censoring correction.** The real project corrects the
  probability-of-default estimate for still-open loans using a fitted Cox proportional-
  hazards model. Harrell's C doesn't need that correction — it works directly off raw
  event/observed_time — so this repo keeps the censoring-*aware* data (the `is_censored`
  flag, the temporal split) without the extra survival-model machinery, which is most of
  what makes this "a much simpler lending test" than the real project's.
- **Explore/exploit iteration phasing.** The real project alternates "blind" iterations
  (no injected memory, to measure a from-scratch baseline) against "informed" ones on a
  fixed schedule. That's a research-design feature of the real ablation study, not
  something that adds to this repo's own story.

## Verification

This repo was validated in layers before any real model call, then exercised for real:

1. **Static**: every `.py` file parses; a repo-wide sweep confirmed no leftover
   pricing/P&L language or research-ladder codenames from the source project.
2. **Import/construction**: `agent.graph.build_graph()` called for all three
   `RUN_CONFIGS` entries (`baseline`/`advisor`/`ltm_full`) and confirmed to produce a
   real, correctly-wired `CompiledStateGraph` for each — this is what caught a real bug
   (a stale `treatment_{...}` directory-naming pattern in
   `agent/app/backends/sandboxed_shell.py`'s Docker mount logic, left over from the
   source project's terminology, that would have silently broken cross-episode file
   visibility for the sandboxed subagent).
3. **Evaluator, against a synthetic fixture** (no real data or API calls needed): a
   hand-built stub model was scored through the real `score_runner.py` →
   `evaluate_episode.py` path, confirming the subprocess-isolated scoring, Harrell's C
   computation, and `ResultsGuardMiddleware`'s missing-artifact/smoke-test/degenerate-
   model checks all work end-to-end.
4. **Real run, against real data and the real API**: the full LendingClub pipeline was
   run end-to-end (catching two more real bugs along the way — `add_censoring_flag.py`'s
   row-alignment check assumed a column present in every split that `build_splits.py`
   deliberately drops from `train.parquet`, and its documented run order had the
   zip-code fix in the wrong position relative to the censoring-flag step), then all
   three `RUN_CONFIGS` were run for real against Anthropic's API — see README.md's
   "Real smoke-test results" for the numbers. This is also where the one honest gap
   noted there was found: `PeriodicCriticMiddleware`'s turn-count-based trigger never
   fired, because this task turned out simple enough that the agent solved it in very
   few main-agent turns.

## Known limitations

- The data-pipeline's feature-leakage guards (dropping outcome columns and LendingClub's
  own retrospective risk signal from the features the agent sees) don't close a
  *deliberate* bypass — an agent could still hardcode the real file's absolute path and
  read it directly. Closing that fully needs OS-level sandboxing (a different container/
  mount namespace/user) that this environment doesn't provide for the main agent process
  (only the subagent's `execute()` calls go through Docker).
- `max_episode_seconds`/`max_episode_tokens` defaults in `RunConfig` are starting
  placeholders, not calibrated values — see the comments on those fields.
- `PeriodicCriticMiddleware` fires on a turn count, not wall-clock or task-progress —
  a heavily-delegated episode (most real work happening inside one subagent call) may
  never reach its trigger threshold. Confirmed live in the real smoke test (see
  "Verification" above); not a bug, but worth knowing before assuming it fired.
