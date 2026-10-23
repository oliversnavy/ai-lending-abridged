"""
Runs INSIDE a subprocess, isolated from the harness's main process. This is the only
place agent-authored code (risk_model.pkl's unpickled class) ever executes. It is only
ever given a features-only file — event/observed_time/is_censored are physically
absent, so there is nothing to accidentally leak. (A model that deliberately hardcodes
the real val.parquet/holdout.parquet's absolute path and reads it directly is a known,
unclosed residual risk; closing that needs real OS-level sandboxing this environment
doesn't have yet.)

Called once per dataset (val, then holdout) against the SAME saved artifact — no
retraining, just two forward passes, so the extra holdout pass is cheap.

Usage (invoked by evaluate_episode.py, not run directly):
    python score_runner.py <skill_dir> <features_path> <output_path>

Writes a parquet with a single column: p_default_hat
"""
from __future__ import annotations
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd


def main() -> None:
    skill_dir = pathlib.Path(sys.argv[1])
    features_path = pathlib.Path(sys.argv[2])
    output_path = pathlib.Path(sys.argv[3])

    model_path = skill_dir / "risk_model.pkl"
    if not model_path.exists():
        print(f"ERROR: risk_model.pkl not found in {skill_dir}", file=sys.stderr)
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict_default_proba"):
        print("ERROR: risk_model.pkl must expose a predict_default_proba(X) method", file=sys.stderr)
        sys.exit(1)

    # features only — no event/observed_time/is_censored/grade/sub_grade/int_rate/
    # installment present in this file (see build_features_only.py); agent code never
    # sees any of them.
    X = pd.read_parquet(features_path)

    p_default_hat = np.asarray(model.predict_default_proba(X), dtype=float)
    if p_default_hat.shape[0] != len(X):
        print(f"ERROR: predict_default_proba returned {p_default_hat.shape[0]} rows, expected {len(X)}", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame({"p_default_hat": p_default_hat})
    out.to_parquet(output_path, index=False)

    print(f"Scored {len(out):,} rows -> {output_path}")


if __name__ == "__main__":
    main()
