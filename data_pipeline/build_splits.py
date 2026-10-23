"""
Build the three-way temporal split from the raw LendingClub dataset.

Outputs (to data/processed/):
    train.parquet    — 2007-2014, agent model fitting
    val.parquet      — 2015-2016, within-loop validation feedback
    holdout.parquet  — 2017-2018, terminal evaluation only

Usage:
    uv run python data_pipeline/build_splits.py
"""

import pathlib
import pandas as pd
from survival_targets import add_survival_targets


RAW_PATH = pathlib.Path("data/accepted_2007_to_2018Q4.csv.gz")
OUT_DIR = pathlib.Path("data/processed")

# Post-origination leakage — reflect outcomes AFTER the loan was issued.
# Using these would let the model "predict" default with information that only
# exists because the loan defaulted. Hard exclude.
LEAKAGE_COLS = {
    # Payment outcomes
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "out_prncp", "out_prncp_inv",
    "last_pymnt_amnt", "next_pymnt_d",
    # Post-origination credit pulls
    "last_fico_range_high", "last_fico_range_low", "last_credit_pull_d",
    # Hardship program (post-origination event)
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    # Debt settlement (post-origination event)
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
    # Payment plan flag — set post-origination
    "pymnt_plan",
}

# PII and non-informative identifiers — exclude for privacy and signal reasons.
# Note: zip_code is intentionally NOT excluded. It is predictive (local economic
# conditions correlate with default risk) but carries disparate impact risk due
# to historical redlining. Leaving it in; flag if agent relies heavily on it.
PII_COLS = {
    "id", "member_id",
    "url",
    "emp_title",   # free-text employer name, too noisy
    "title",       # borrower-entered loan title, free text
    "desc",        # free-text loan description
    "policy_code", # internal LC code, no signal
}

# Target construction columns — used to build event/observed_time, then dropped
TARGET_COLS = {"loan_status", "issue_d", "last_pymnt_d"}

# Rival-signal / pricing-circularity columns -- see data_pipeline/build_features_only.py's
# module docstring for the full reasoning, and docs/GRADE_FEATURE_REMOVAL_PLAN_2026_09_14.md.
# Dropped from train.parquet ONLY (added 2026-09-15) -- val.parquet/holdout.parquet keep
# them, since they're needed as real ground truth by LGD_BY_GRADE and the sensitivity
# models, and build_features_only.py separately strips them from the agent-facing
# *_features_only.parquet copies of those files. train.parquet has no such ground-truth
# use case (its only consumer is the agent's own model training), so dropping it here
# directly avoids ever training a model that depends on a column it won't have at
# scoring time.
TRAIN_ONLY_DROP_COLS = ["grade", "sub_grade", "int_rate", "installment"]


def load_and_prepare() -> pd.DataFrame:
    all_exclude = LEAKAGE_COLS | PII_COLS | TARGET_COLS

    print(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"  Raw rows:    {len(df):,}")
    print(f"  Raw cols:    {len(df.columns)}")

    # Drop duplicate columns (raw CSV has a handful of repeated names)
    df = df.loc[:, ~df.columns.duplicated()]

    df = add_survival_targets(df)
    df["issue_year"] = df["issue_d"].dt.year

    # Reset index to eliminate any fragmentation from filtering in survival_targets
    df = df.reset_index(drop=True)

    derived_cols = {"event", "observed_time", "issue_year"}
    feature_cols = [c for c in df.columns if c not in all_exclude | derived_cols]
    keep_cols = feature_cols + ["event", "observed_time", "issue_year"]
    df = df[keep_cols]

    print(f"  Rows after status filtering: {len(df):,}")
    print(f"  Feature cols kept: {len(feature_cols)}")
    return df


def split_and_save(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train":   df.loc[df["issue_year"] <= 2014].reset_index(drop=True),
        "val":     df.loc[(df["issue_year"] >= 2015) & (df["issue_year"] <= 2016)].reset_index(drop=True),
        "holdout": df.loc[df["issue_year"] >= 2017].reset_index(drop=True),
    }

    for name, split in splits.items():
        path = OUT_DIR / f"{name}.parquet"
        # Diagnostics always read from the full split, regardless of what gets written --
        # grade_counts is print-only, never touches disk.
        event_rate = split["event"].mean()
        grade_counts = split["grade"].value_counts().sort_index().to_dict()

        to_write = split.drop(columns=TRAIN_ONLY_DROP_COLS) if name == "train" else split
        to_write.to_parquet(path, index=False)

        print(f"\n{name}.parquet")
        print(f"  Rows:       {len(split):,}")
        print(f"  Cols:       {len(to_write.columns)}" + (" (rival-signal cols dropped)" if name == "train" else ""))
        print(f"  Event rate: {event_rate:.2%}")
        print(f"  Grades:     {grade_counts}")
        print(f"  Saved →     {path}")


if __name__ == "__main__":
    df = load_and_prepare()
    split_and_save(df)
    print("\nDone.")
