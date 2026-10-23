# ai-lending-abridged

**[Read the landing page →](https://oliversnavy.github.io/ai-lending-abridged/)**

A deep-agent harness built on **LangGraph + deepagents**: multi-agent orchestration,
short- and long-term memory, a three-critic feedback lifecycle, and censoring-aware
evaluation methodology for a non-deterministic system — demonstrated on a public
credit-risk-ranking task.

## What this demonstrates

| Capability | Where |
|---|---|
| Multi-agent patterns, state management (short + long-term memory) | `agent/graph.py` (main agent + sandboxed subagent), `agent/loop.py` (cross-episode memory loop) |
| Agent architecture design, orchestration patterns | `agent/graph.py`'s middleware stack and ordering, `ARCHITECTURE.md` |
| Eval methodology for non-deterministic systems | `agent/evaluator/` — independent re-scoring, censoring-aware metric, degenerate-model detection |
| Advisory-to-embedded delivery range | `RUN_CONFIGS["advisor"]` vs `RUN_CONFIGS["ltm_full"]` — see below |
| Trace mining / continuous improvement loops | the three-critic lifecycle (`preloop_anchor.py` → `periodic_critic.py` → `post_mortem.py`) |
| Harness engineering under real failure modes | `ARCHITECTURE.md`'s "what broke and how it was fixed" section |

## Quick start

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY
source .env

# One-time data setup — see data_pipeline/README.md
uv run python data_pipeline/build_splits.py
uv run python data_pipeline/add_censoring_flag.py
uv run python data_pipeline/fix_holdout_zip_code_row.py
uv run python data_pipeline/build_features_only.py

# Run any of the three illustrative configs
uv run python agent/run.py --run-label baseline --n-runs 1
uv run python agent/run.py --run-label advisor --n-runs 1
uv run python agent/run.py --run-label ltm_full          # loops with memory until plateau
```

## The task

Build the best **credit-risk ranking model** you can on public LendingClub loan data,
scored by Harrell's C (concordance index) on data the agent never sees. No pricing, no
portfolio allocation — just an honest, well-ranked default-risk score per applicant.
See `agent/app/prompts/system.py` for the full task given to the agent, and
`data_pipeline/README.md` for the data split and why certain columns are excluded.

## Three run configs, one spectrum

`agent/app/models/episode.py`'s `RUN_CONFIGS` ships three configs that double as a demo
of this harness's own range, from plain deep agent through advisory escalation to a
fully embedded, memory-carrying loop:

- **`baseline`** — a single deep-agent episode, no critics, no memory.
- **`advisor`** — adds `advisor_consult`: the agent escalates to a senior-reviewer LLM
  at defined checkpoints (start / decision points / stuck / end). The "advisory" point
  on the delivery-range spectrum.
- **`ltm_full`** — a looped instance where every episode sees the full history of prior
  episodes (`loop_mode=ALL_PRIOR`), each one reviewed by a post-episode critic whose
  distilled learnings carry forward, plus a one-time "starting critic" that surfaces the
  single best prior approach before the next attempt begins, plus a periodic in-episode
  critic. The "embedded" point.

These aren't research findings about which approach is "best" — they're a demonstration
of the harness's ability to compose (or not compose) these capabilities cleanly.

## Real smoke-test results

All three configs have been run for real against the Anthropic API on the actual
LendingClub data (not simulated) — the output is committed under `results/` as a small,
real worked example, not a placeholder.

| Config | Episodes | Holdout c-stat | Notes |
|---|---|---|---|
| `baseline` | 1 | 0.662 | XGBoost Cox survival model, chosen over a logistic-regression baseline |
| `advisor` | 1 | 0.682 | Consulted the advisor twice (methodology at the start, sanity-check at the end) |
| `ltm_full` | 3 (loop) | 0.667 → 0.682 → 0.696 | Monotonically improving holdout c-stat across iterations; post-mortem resolved 2 prior open learnings by iteration 2 |

One honest caveat from this run: the mid-episode critic (`PeriodicCriticMiddleware`)
fires on a turn count, not a wall-clock or task-progress basis, and this task turned out
simple enough that the agent solved it in very few main-agent turns (most of the real
work happened inside a single subagent delegation) — so it never actually fired in this
particular smoke test, even though the starting critic and end-of-episode critic both
did. Its construction/wiring is confirmed correct (see "Verification" in
`ARCHITECTURE.md`) and it would fire on a longer-running or less delegation-heavy
episode; this run's task just didn't happen to produce one.

## Repo layout

```
agent/
  graph.py              deepagents wiring: main agent + sandboxed subagent
  loop.py                the cross-episode memory loop (resumable, plateau detection)
  run.py                  CLI entry point
  app/
    middleware/           guardrails, context management, the three critics
    post_mortem.py        end-of-episode critic / long-term-memory distillation
    clients/llm.py        Anthropic / Baseten backend switch
    tools/                sandboxed filesystem + code-execution tools
  evaluator/              independent, subprocess-isolated scoring
data_pipeline/            LendingClub data prep (temporal split, censoring, leakage guards)
configs/base.yaml         loop + model settings
ARCHITECTURE.md           the deeper write-up — read this for the "why"
```

## Relationship to the real research project

This is a deliberate fork-and-strip of a private, ongoing multi-month research project
that studies long-term memory in deep agents for sequential financial decision-making.
That project's actual novel contribution — a sequential, capital-constrained pricing
mechanic, and the empirical findings about what helps agents use memory well — is **not**
included here; it's reserved for a forthcoming paper. What's shown here is the generic
harness engineering underneath it: the orchestration, sandboxing, memory architecture,
and eval discipline, re-platformed onto a simpler, standard, publicly-available task so
it can be shared without exposing unpublished research or anything proprietary. The
underlying data is the public LendingClub dataset — never any employer's or proprietary
data.

## License

No license is granted. This code is shared publicly so it can be read and evaluated —
for a job application, for example — but all rights are reserved; reuse, modification,
or redistribution requires permission.
