"""
ResultsGuardMiddleware — re-triggers the agent if its deliverable artifact is missing or
broken.

When the agent finishes an episode without saving risk_model.pkl (context exhaustion,
getting stuck in analysis, etc.), after_agent() fires, injects a short recovery
HumanMessage, and jumps back to the model node to restart the agentic loop. The agent
sees its full message history + pipeline files in skill_dir and should be able to save
the artifact in one more pass.

Existence alone isn't sufficient: a saved risk_model.pkl can still be broken (e.g. fit
on raw categorical string columns it was never encoded for) and crash the independent
scorer. To catch this class of bug before the episode ends, if the file exists we
additionally run a real smoke test: a tiny real sample is scored through the exact same
score_runner.py subprocess the independent evaluator uses. A thrown exception there is
treated the same as "missing" (same recovery-retry mechanism), but the recovery prompt
includes the actual traceback so the agent has concrete signal instead of just
re-guessing.

Deliberately does NOT ask the agent to compute or report final metrics itself -- c_stat
is computed independently by agent/evaluator/evaluate_episode.py from the submitted
artifact. The source research project hit a real incident where asking the agent to
"compute the metrics and write results.json right now" nudged an episode toward
reporting fabricated numbers under time pressure; the fix is structural (the agent's
self-report is never authoritative for anything but free-text approach/hypothesis), not
just a wording change, but the wording here still avoids inviting that failure mode.

Max retries per episode bounded by max_retries=2 to avoid infinite loops while still
giving the agent real runway to fix a diagnosed bug rather than giving up after one
attempt.

__main__ pickling signature detection: the system prompt's "Avoiding a pickling gotcha"
section already has the proven fix, so when the smoke-test traceback matches that exact
signature, the recovery prompt injects the specific known fix directly instead of a
generic hint.

Degenerate-risk-model check: a collapsed risk model (near-constant predictions) is a
genuine modeling failure worth surfacing to the agent DURING the episode, not just after
scoring. Uses a much larger sample than the artifact smoke test above -- nunique is a
whole-distribution diagnostic, meaningless on a handful of rows -- see
risk_model_check.py for the real threshold calibration.
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

from agent.evaluator.evaluate_episode import DATA_DIR, PROJECT_ROOT, SCORE_RUNNER
from agent.evaluator.risk_model_check import DEGENERACY_NUNIQUE_FLOOR

log = logging.getLogger(__name__)

_SMOKE_TEST_SAMPLE_ROWS = 5
_SMOKE_TEST_TIMEOUT_SECONDS = 60
_SMOKE_TEST_FEATURES_PATH = DATA_DIR / "val_features_only.parquet"
_TRACEBACK_CHARS_KEPT = 4000

# Degenerate-risk-model check: deliberately reuses the full val_features_only.parquet
# (855K rows), not the 5-row artifact-smoke-test sample above -- nunique needs real
# population scale to mean anything (see module docstring).
_DEGENERACY_CHECK_SCRIPT = PROJECT_ROOT / "agent" / "evaluator" / "risk_model_check.py"
_DEGENERACY_CHECK_FEATURES_PATH = DATA_DIR / "val_features_only.parquet"
_DEGENERACY_CHECK_TIMEOUT_SECONDS = 60

# Matches e.g. "AttributeError: Can't get attribute 'RiskModel' on <module '__main__' ...>"
_MAIN_MODULE_PICKLE_RE = re.compile(r"module '__main__'")

_MAIN_MODULE_PICKLE_FIX = (
    "This is the exact pickling gotcha described in your system prompt: the model's "
    "class was defined in a script that YOU ran directly (e.g. via execute() or "
    "code_executor), so Python recorded its module as `__main__` -- but the harness's "
    "scoring process is a separate Python process with its OWN `__main__`, which does "
    "not contain your class, so it can't unpickle the object.\n\n"
    "Fix: put the class definition in its own file (e.g. model_def.py), written via "
    "write_file (not executed directly as a script), then in a SEPARATE script that "
    "IMPORTS that file, instantiate the model and pickle it there:\n"
    "```python\n"
    "# model_def.py\n"
    "class MyRiskModel:\n"
    "    def __init__(self, ...): ...\n"
    "    def predict_default_proba(self, X): ...\n\n"
    "# save_model.py -- a DIFFERENT file that imports model_def, never run as the\n"
    "# script that DEFINES the class\n"
    "import sys; sys.path.insert(0, '.')\n"
    "import model_def\n"
    "wrapped = model_def.MyRiskModel(...)\n"
    "import pickle\n"
    "with open('risk_model.pkl', 'wb') as f:\n"
    "    pickle.dump(wrapped, f)\n"
    "```\n"
    "Then verify by loading it in a FRESH process that only imports model_def and "
    "unpickles -- do not just re-run your training script and check the in-memory "
    "object, that will not catch this bug."
)


class ResultsGuardMiddleware(AgentMiddleware):
    """Re-triggers the agent with a recovery prompt if its deliverable artifact is
    absent or broken."""

    def __init__(self, skill_dir: pathlib.Path, max_retries: int = 2) -> None:
        self.skill_dir = skill_dir
        self.max_retries = max_retries
        self._attempts = 0

    def _missing_artifacts(self) -> list[str]:
        if not (self.skill_dir / "risk_model.pkl").exists():
            return ["risk_model.pkl"]
        return []

    def _smoke_test_artifacts(self) -> str | None:
        """Score a tiny real sample through score_runner.py, the same subprocess path
        evaluate_episode.py uses at episode-end scoring. Returns None if it passes, else
        the captured error output. Also returns None (does not penalize the agent) if the
        smoke test itself can't run for a harness-side reason (e.g. sample data missing).
        """
        try:
            sample = pd.read_parquet(_SMOKE_TEST_FEATURES_PATH).head(_SMOKE_TEST_SAMPLE_ROWS)
        except Exception as exc:
            log.warning("[ResultsGuard] smoke test could not load sample data: %s", exc)
            return None

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            features_sample_path = tmp_dir / "smoke_features.parquet"
            output_path = tmp_dir / "smoke_scored.parquet"
            sample.to_parquet(features_sample_path, index=False)

            env = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "PYTHONPATH": f"{self.skill_dir}:{PROJECT_ROOT}",
            }
            try:
                result = subprocess.run(
                    [sys.executable, str(SCORE_RUNNER), str(self.skill_dir),
                     str(features_sample_path), str(output_path)],
                    cwd=str(self.skill_dir),
                    capture_output=True,
                    text=True,
                    timeout=_SMOKE_TEST_TIMEOUT_SECONDS,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return f"Smoke test timed out after {_SMOKE_TEST_TIMEOUT_SECONDS}s scoring a {_SMOKE_TEST_SAMPLE_ROWS}-row sample."

            if result.returncode != 0 or not output_path.exists():
                return (result.stderr or result.stdout or "score_runner.py exited non-zero with no output").strip()

        return None

    def _degenerate_risk_model(self) -> dict | None:
        """Runs risk_model.pkl against the full val feature set in an isolated
        subprocess (same isolation pattern as _smoke_test_artifacts) and checks whether
        its prediction distribution has collapsed to a handful of repeated values.
        Returns None if it looks healthy, or if the check itself can't run for a
        harness-side reason (never penalizes the agent for that) -- else a dict with the
        diagnostic numbers for the recovery prompt."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = pathlib.Path(tmp) / "degeneracy_check.json"
            env = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "PYTHONPATH": f"{self.skill_dir}:{PROJECT_ROOT}",
            }
            try:
                result = subprocess.run(
                    [sys.executable, str(_DEGENERACY_CHECK_SCRIPT), str(self.skill_dir),
                     str(_DEGENERACY_CHECK_FEATURES_PATH), str(output_path)],
                    cwd=str(self.skill_dir),
                    capture_output=True,
                    text=True,
                    timeout=_DEGENERACY_CHECK_TIMEOUT_SECONDS,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                log.warning("[ResultsGuard] degeneracy check timed out — skipping (not penalizing the agent)")
                return None

            if result.returncode != 0 or not output_path.exists():
                log.warning(
                    "[ResultsGuard] degeneracy check couldn't run, skipping: %s",
                    (result.stderr or result.stdout or "no output")[-500:].strip(),
                )
                return None

            stats = json.loads(output_path.read_text())

        if stats["nunique"] <= DEGENERACY_NUNIQUE_FLOOR:
            return stats
        return None

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        missing = self._missing_artifacts()
        broken_traceback = None if missing else self._smoke_test_artifacts()
        degeneracy_stats = None if (missing or broken_traceback) else self._degenerate_risk_model()

        if not missing and broken_traceback is None and degeneracy_stats is None:
            return None

        if self._attempts >= self.max_retries:
            if missing:
                log.warning(
                    "[ResultsGuard] artifact still missing (%s) after %d retries — giving up",
                    ", ".join(missing), self._attempts,
                )
            elif broken_traceback is not None:
                log.warning(
                    "[ResultsGuard] artifact still broken after %d retries — giving up",
                    self._attempts,
                )
            else:
                log.warning(
                    "[ResultsGuard] risk model still degenerate (nunique=%d) after %d retries — giving up",
                    degeneracy_stats["nunique"], self._attempts,
                )
            return None

        self._attempts += 1

        if missing:
            log.warning(
                "[ResultsGuard] missing %s — injecting recovery prompt (attempt %d/%d)",
                ", ".join(missing), self._attempts, self.max_retries,
            )
            content = (
                f"IMPORTANT: your episode is ending but you have not saved: {', '.join(missing)}.\n\n"
                "This is your deliverable — the harness independently re-applies it to score "
                "your episode; it does not use any c_stat number you report yourself. "
                "Do NOT re-train from scratch. Using the model you already have "
                "in this session:\n\n"
                "Save your trained risk model as risk_model.pkl in your working directory. "
                "The pickled object must expose predict_default_proba(X: pd.DataFrame) -> np.ndarray "
                "returning P(default) per row.\n\n"
                "A simple pipeline that actually saves this file beats a sophisticated one "
                "that doesn't. You may optionally also write results.json with \"approach\" and "
                "\"hypothesis\" text fields — those are for readability only, not scored."
            )
        elif broken_traceback is not None:
            log.warning(
                "[ResultsGuard] artifact present but broken — injecting recovery prompt with traceback (attempt %d/%d)",
                self._attempts, self.max_retries,
            )
            truncated = broken_traceback[-_TRACEBACK_CHARS_KEPT:]
            if _MAIN_MODULE_PICKLE_RE.search(broken_traceback):
                specific_guidance = _MAIN_MODULE_PICKLE_FIX
            else:
                specific_guidance = (
                    "Do NOT re-train from scratch. A common cause is a saved model that implicitly "
                    "assumes training-time preprocessing (categorical encoding, column selection, etc.) "
                    "was already applied — but the harness calls predict_default_proba(X) directly on "
                    "raw feature columns. Fix the bug in your existing risk_model.pkl "
                    "(re-pickling/re-saving as needed) in your working directory."
                )
            content = (
                "IMPORTANT: your episode is ending but your saved artifact fails when the harness's "
                "independent scorer tries to use it. This is the ACTUAL error from re-applying your "
                "risk_model.pkl to a small held-out sample, exactly the way "
                "scoring will happen for real:\n\n"
                f"```\n{truncated}\n```\n\n"
                f"{specific_guidance}"
            )
        else:
            log.warning(
                "[ResultsGuard] risk model looks degenerate (nunique=%d on %d real rows) — "
                "injecting recovery prompt (attempt %d/%d)",
                degeneracy_stats["nunique"], degeneracy_stats["n"], self._attempts, self.max_retries,
            )
            content = (
                "IMPORTANT: your risk_model.pkl's predictions look degenerate when checked against a "
                f"large real sample ({degeneracy_stats['n']:,} rows): only "
                f"{degeneracy_stats['nunique']} distinct predicted values across the whole sample "
                f"(mean={degeneracy_stats['mean']:.4f}, std={degeneracy_stats['std']:.4f}). A model doing "
                "real per-applicant risk reasoning essentially never collapses to a handful of repeated "
                "values like this, even a weak one -- this is a strong signal of a bug, not just "
                "low accuracy.\n\n"
                "Two common real causes of exactly this failure:\n"
                "1. A feature-mapping/encoding step (e.g. a category or zip-code map) that was fit only "
                "on train and silently zeroes out any value not seen there -- if val/holdout have "
                "categories train didn't, many rows can end up with identical (all-zero) feature vectors "
                "and therefore identical predictions.\n"
                "2. Accidentally using a hard classifier's discrete output (predict(), 0/1 or a small "
                "number of classes) instead of a continuous probability (predict_proba()[:, 1] or "
                "equivalent) inside predict_default_proba.\n\n"
                "Before finishing: check your own predict_default_proba(X) output's number of distinct "
                "values on a large real sample (not just a few rows) and confirm it varies smoothly "
                "with applicant risk, not just technically runs without error."
            )

        return {
            "messages": [HumanMessage(content=content)],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
