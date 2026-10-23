"""
Build features-only copies of val.parquet and holdout.parquet for the independent
evaluator harness. Two DIFFERENT categories of column get excluded here, for two
different reasons:

  1. Outcome-derived leakage (`event`, `observed_time`, `is_censored`) — the harness's
     re-application of an agent-submitted risk_model.pkl must never have access to
     realized outcomes, or a model could (accidentally or deliberately) leak ground
     truth into a decision that's supposed to be ex-ante. Passing a DataFrame that
     structurally lacks those columns closes the accidental case completely.
  2. Rival-signal leakage (`grade`, `sub_grade`, `int_rate`, `installment`). `grade`/
     `sub_grade` are LendingClub's own internal risk categorization for this specific
     applicant — information a genuinely new applicant's lender wouldn't realistically
     have obtained yet, not classic outcome leakage (it's pre-origination). `int_rate`
     is the rate LendingClub itself historically *charged* on the loan — i.e. the record
     of a retrospective pricing decision, not an input that should exist yet at
     prediction time for a new applicant. `installment` is a near-deterministic function
     of `int_rate` + `loan_amnt` + `term` and leaks it back in if left alone. Both
     categories exist so a risk model is forced to actually rank risk from the
     applicant's own pre-origination characteristics, not look up an answer a rival
     signal already encodes.

Neither category closes a *deliberate* bypass (agent code hardcoding the real file's
absolute path and reading it directly) — that requires real OS-level sandboxing
(container / mount namespace / different user), which this repo's sandboxing (see
agent/app/backends/sandboxed_shell.py) only partially provides.

HOLDOUT-ONLY column nulling: 16 secondary-applicant/joint-application columns
(HOLDOUT_NULL_COLS below) are ~100% null in train.parquet (2007-2014) and val.parquet
(2015-2016) but ~11-12% populated in holdout.parquet (2017-2018) — a genuine
LendingClub data-collection completeness change between periods, not corrupted data or
a pipeline bug. Left alone, this causes structurally-uncatchable holdout-only crashes
whenever a model's feature-engineering code touches these columns — the agent has no
way to observe or validate against a value distribution it never once sees during
training or across its entire val-based feedback loop. Nulling these columns in
holdout_features_only.parquet ONLY (never holdout.parquet itself, which keeps the real
data for the harness's own ground-truth scoring, and never train/val, so nothing a
model has already seen changes) removes this specific confound going forward without
touching anything upstream of scoring.

Usage:
    uv run python data_pipeline/build_features_only.py
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path("data/processed")

# event/observed_time: the original outcome columns. is_censored (added by
# add_censoring_flag.py) is derived from them and equally outcome-adjacent -- same
# leakage concern, same exclusion.
OUTCOME_LEAKAGE_COLS = ["event", "observed_time", "is_censored"]

# Rival-signal leakage -- see module docstring.
RIVAL_SIGNAL_COLS = ["grade", "sub_grade", "int_rate", "installment"]

DROP_COLS = OUTCOME_LEAKAGE_COLS + RIVAL_SIGNAL_COLS

# All confirmed ~100% null in both train and val, ~11-12% populated in holdout only.
HOLDOUT_NULL_COLS = [
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog", "revol_bal_joint", "annual_inc_joint",
    "dti_joint", "verification_status_joint",
]


def build(name: str) -> None:
    src = DATA_DIR / f"{name}.parquet"
    dst = DATA_DIR / f"{name}_features_only.parquet"
    df = pd.read_parquet(src)
    features_only = df.drop(columns=DROP_COLS, errors="ignore")

    if name == "holdout":
        present = [c for c in HOLDOUT_NULL_COLS if c in features_only.columns]
        all_true = pd.Series(True, index=features_only.index)
        for col in present:
            # .mask(all_true) replaces every value with the dtype-appropriate null
            # (NaN for float64, pd.NA/None for string/extension dtypes) while
            # preserving the column's original dtype -- matters for round-tripping
            # through parquet consistently with how train/val already represent an
            # all-null column of this same dtype. A blanket `= None` assignment can
            # silently coerce dtype instead.
            features_only[col] = features_only[col].mask(all_true)
        missing = set(HOLDOUT_NULL_COLS) - set(present)
        if missing:
            print(f"  WARNING: expected columns not found in {name}: {sorted(missing)}")

    # Atomic write: the evaluator reads this file fresh on every scoring call, so a
    # concurrent reader must never observe a partially-written file.
    tmp = dst.with_suffix(".parquet.tmp")
    features_only.to_parquet(tmp, index=False)
    tmp.replace(dst)
    print(f"{dst}  ({len(features_only):,} rows, {len(features_only.columns)} cols)")


if __name__ == "__main__":
    build("val")
    build("holdout")
    print("Done.")
