"""
Independent evaluator — the sole source of truth for episode metrics.

The agent does not self-report c_stat. It submits one artifact (risk_model.pkl); this
module reloads it, applies it to a fresh read of the data, and computes Harrell's C
itself. Nothing the agent writes to results.json (if anything) is authoritative for
this field — only approach/hypothesis free text is taken from the agent, for
readability in the episode index.

Two datasets are scored against the SAME saved artifact, no retraining:
  - holdout.parquet  -> the PRIMARY, reported metric (c_stat). The agent never touches
    this during the episode, so it's a genuinely unbiased read on how the model
    generalizes.
  - val.parquet      -> DIAGNOSTIC metric only (c_stat_val), kept for within-loop
    analysis. The agent's own local testing is tuned against this exact sample during
    the episode, so it's expected to look somewhat better than holdout — that's exactly
    why holdout, not val, is the reported number: repeatedly testing against the same
    finite val sample is its own (non-leakage) source of overfitting, distinct from the
    ground-truth-leakage problem the features-only split addresses.

The scoring step (agent-authored model code) runs in a subprocess via score_runner.py,
which only ever sees a features-only file (event/observed_time/is_censored physically
absent). This process never unpickles the agent's model directly and never holds ground
truth in the same process as agent code.
"""
from __future__ import annotations
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending-abridged")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCORE_RUNNER = pathlib.Path(__file__).parent / "score_runner.py"
SCORING_TIMEOUT_SECONDS = 600

DATASETS = {
    "holdout": (DATA_DIR / "holdout_features_only.parquet", DATA_DIR / "holdout.parquet"),
    "val": (DATA_DIR / "val_features_only.parquet", DATA_DIR / "val.parquet"),
}

_ZERO_METRICS = {"c_stat": 0.0}


def _run_score_subprocess(skill_dir: pathlib.Path, features_path: pathlib.Path) -> pd.DataFrame | None:
    with tempfile.TemporaryDirectory() as tmp:
        output_path = pathlib.Path(tmp) / "scored.parquet"
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            # skill_dir on PYTHONPATH so custom classes the agent's risk_model.pkl
            # depends on (defined in the agent's own scripts) can be found again for
            # unpickling. PROJECT_ROOT doesn't need to be here for the model to work,
            # but is included for consistency with the rest of this harness's
            # subprocess calls.
            "PYTHONPATH": f"{skill_dir}:{PROJECT_ROOT}",
        }
        try:
            result = subprocess.run(
                [sys.executable, str(SCORE_RUNNER), str(skill_dir), str(features_path), str(output_path)],
                cwd=str(skill_dir),
                capture_output=True,
                text=True,
                timeout=SCORING_TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"[evaluator] scoring timed out after {SCORING_TIMEOUT_SECONDS}s ({features_path.name})")
            return None

        if result.returncode != 0:
            print(f"[evaluator] scoring failed ({features_path.name}):\n{result.stdout}\n{result.stderr}")
            return None

        if not output_path.exists():
            print(f"[evaluator] scoring subprocess reported success but wrote no output ({features_path.name})")
            return None

        return pd.read_parquet(output_path)


def _score_dataset(skill_dir: pathlib.Path, features_path: pathlib.Path, ground_truth_path: pathlib.Path) -> dict | None:
    scored = _run_score_subprocess(skill_dir, features_path)
    if scored is None:
        return None

    truth = pd.read_parquet(ground_truth_path, columns=["observed_time", "event"])
    if len(truth) != len(scored):
        print(f"[evaluator] row count mismatch: {ground_truth_path.name}={len(truth)} scored={len(scored)}")
        return None

    p_default_hat = scored["p_default_hat"].values

    # A model can produce a NaN p_default_hat for some rows even when it looked fine
    # during the agent's own val-based iteration (a real failure mode observed on a
    # larger population than val alone covers). concordance_index() has no NaN
    # tolerance and raises uncaught. Treated the same as the row-count mismatch above
    # (a structural scoring failure, not a partial result to patch around): dropping
    # only the NaN rows before scoring would silently change which population of loans
    # this episode's c_stat is computed over relative to every other episode's,
    # breaking cross-episode comparability.
    nan_mask = np.isnan(p_default_hat)
    if nan_mask.any():
        print(f"[evaluator] {int(nan_mask.sum())}/{len(p_default_hat)} NaN p_default_hat "
              f"values ({ground_truth_path.name}) -- failing this dataset's scoring rather "
              f"than computing metrics over a silently reduced population")
        return None

    c_stat = float(concordance_index(truth["observed_time"], -p_default_hat, truth["event"]))
    return {"c_stat": c_stat}


def evaluate_episode(skill_dir: pathlib.Path, dataset: str | None = None) -> dict:
    """
    Default (dataset=None): returns the primary (holdout) metric as c_stat, plus the
    val-diagnostic metric as c_stat_val. eval_status is "ok" (both scored), "partial"
    (only one scored), or "failed" (neither).

    dataset="val"|"holdout": score against exactly one dataset -- that dataset's number
    populates c_stat regardless of which one was requested; eval_status is "ok"/"failed"
    only (no "partial", since there's only one dataset to score).
    """
    if dataset is not None:
        metrics = _score_dataset(skill_dir, *DATASETS[dataset])
        result = dict(metrics or _ZERO_METRICS)
        result["eval_status"] = "ok" if metrics is not None else "failed"
        return result

    holdout_metrics = _score_dataset(skill_dir, *DATASETS["holdout"])
    val_metrics = _score_dataset(skill_dir, *DATASETS["val"])

    if holdout_metrics is None and val_metrics is None:
        status = "failed"
    elif holdout_metrics is None or val_metrics is None:
        status = "partial"
    else:
        status = "ok"

    result = dict(holdout_metrics or _ZERO_METRICS)
    result.update({f"{k}_val": v for k, v in (val_metrics or _ZERO_METRICS).items()})
    result["eval_status"] = status
    return result
