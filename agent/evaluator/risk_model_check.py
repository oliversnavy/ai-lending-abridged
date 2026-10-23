"""
Runs INSIDE a subprocess, isolated from the harness's main process -- same isolation
rationale as score_runner.py (this also unpickles agent-authored risk_model.pkl, so it
must never execute in the same process as the trusted agent-loop/middleware code).

Only loads risk_model.pkl and runs predict_default_proba over a real feature sample.
Used by ResultsGuardMiddleware's degenerate-model check, which needs the prediction
distribution's shape (specifically, how many distinct values it takes), not a full score.

Usage: python risk_model_check.py <skill_dir> <features_path> <output_path>
Writes JSON to output_path: {"n": ..., "nunique": ..., "mean": ..., "std": ...}
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

# A prediction distribution collapsed to only a handful of distinct values is a strong
# signal of a degenerate model (e.g. it fell back to a constant or near-constant output)
# rather than genuine per-applicant risk ranking. Calibrated against real pilot data in
# the source research project: degenerate episodes showed nunique=2; healthy episodes
# showed nunique between ~1,450 and ~940,000 -- this floor sits with wide margin below
# that real gap, not tuned to the boundary.
DEGENERACY_NUNIQUE_FLOOR = 50


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

    X = pd.read_parquet(features_path)
    p = np.asarray(model.predict_default_proba(X), dtype=float)
    if p.shape[0] != len(X):
        print(f"ERROR: predict_default_proba returned {p.shape[0]} rows, expected {len(X)}", file=sys.stderr)
        sys.exit(1)

    with open(output_path, "w") as f:
        json.dump({
            "n": int(len(p)),
            "nunique": int(len(np.unique(p))),
            "mean": float(np.mean(p)),
            "std": float(np.std(p)),
        }, f)


if __name__ == "__main__":
    main()
