"""
One-time fix (2026-08-25): add an `is_censored` boolean column to train/val/holdout.

Motivation: `loan_status` was used to derive `event`/`observed_time` in
survival_targets.py, then dropped from the processed files entirely (see
build_splits.py's TARGET_COLS). This means "Fully Paid" (fully resolved, confirmed
non-default) and "Current"/"Late"/"In Grace Period" (still open, outcome genuinely
unknown) are currently indistinguishable -- both just show event=0. Any downstream
work that needs to know which rows are censored (the right-censoring correction --
see docs/RESTART_PLAN_2026_08_24.md Part 1) needs this distinction restored.

There's no stable join key to recover this by (id/member_id were dropped as PII), so
this re-runs the exact same loading/filtering/ordering logic as build_splits.py --
same raw file, same unchanged survival_targets.add_survival_targets(), same
reset_index() timing -- to reproduce the identical row order, then verifies that
reproduction against the real processed files on several already-shared columns
(grade, loan_amnt, event, observed_time) before trusting the alignment enough to
write anything. Verified 2026-08-25: 100% exact match on all three splits once the
holdout zip_code-row drop (fix_holdout_zip_code_row.py) is replicated too.

is_censored = True for loan_status in {Current, Late (16-30 days),
Late (31-120 days), In Grace Period} -- i.e. event=0 rows whose true eventual
outcome isn't actually known yet. False for everything else (Fully Paid / Charged
Off / their "does not meet credit policy" variants) -- already fully resolved,
nothing to correct.

Deliberately NOT applied to *_features_only.parquet -- those files are the ONLY
thing agent-authored code ever sees during scoring (score_runner.py), specifically
built to contain nothing derived from event/observed_time. Adding is_censored there
would leak outcome-adjacent information across that boundary.

Usage:
    uv run python data_pipeline/add_censoring_flag.py
"""
import pathlib

import pandas as pd
from survival_targets import add_survival_targets

RAW_PATH = pathlib.Path("data/accepted_2007_to_2018Q4.csv.gz")
DATA_DIR = pathlib.Path("data/processed")

CENSORED_STATUSES = {"Current", "Late (16-30 days)", "Late (31-120 days)", "In Grace Period"}

CHECK_COLS = ["grade", "loan_amnt", "event", "observed_time"]


def reconstruct_with_status() -> pd.DataFrame:
    print(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()]

    df = add_survival_targets(df)
    df["issue_year"] = df["issue_d"].dt.year
    df = df.reset_index(drop=True)  # identical reset point to build_splits.py

    df["is_censored"] = df["loan_status"].isin(CENSORED_STATUSES)
    # sanity: censored rows must always read event=0 (a charged-off loan is resolved,
    # not "still open" by definition)
    assert not (df["is_censored"] & (df["event"] == 1)).any(), \
        "is_censored and event=1 co-occur -- logic error, do not proceed"

    return df[["loan_status", "is_censored", "grade", "loan_amnt", "event",
               "observed_time", "issue_year", "zip_code"]]


def split_recon(recon: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {
        "train": recon.loc[recon["issue_year"] <= 2014].reset_index(drop=True),
        "val": recon.loc[recon["issue_year"].between(2015, 2016)].reset_index(drop=True),
        "holdout": recon.loc[recon["issue_year"] >= 2017].reset_index(drop=True),
    }
    # replicate fix_holdout_zip_code_row.py's row drop, applied after the original build
    null_mask = splits["holdout"]["zip_code"].isna()
    n_null = int(null_mask.sum())
    if n_null != 1:
        raise RuntimeError(
            f"Expected exactly 1 null zip_code row in reconstructed holdout, found {n_null}. "
            "The zip_code fix's own precondition may no longer hold -- stop and investigate."
        )
    splits["holdout"] = splits["holdout"][~null_mask].reset_index(drop=True)
    return splits


def verify_and_apply(name: str, recon_split: pd.DataFrame) -> None:
    path = DATA_DIR / f"{name}.parquet"
    real = pd.read_parquet(path)

    if len(real) != len(recon_split):
        raise RuntimeError(
            f"{name}: row count mismatch (real={len(real):,}, reconstructed={len(recon_split):,}) "
            "-- refusing to write is_censored under an unverified alignment."
        )

    # `grade` is deliberately absent from train.parquet (build_splits.py's
    # TRAIN_ONLY_DROP_COLS -- see that module's docstring for why), so it can't be part
    # of the alignment check there. loan_amnt/event/observed_time are present in every
    # split and are enough on their own to verify row alignment.
    check_cols = [c for c in CHECK_COLS if c in real.columns]
    if len(check_cols) < 3:
        raise RuntimeError(
            f"{name}: only {len(check_cols)} of {CHECK_COLS} present in {path} -- too few "
            "columns to trust an alignment check, refusing to write is_censored."
        )

    for col in check_cols:
        if col == "observed_time":
            match = (real[col].round(6).values == recon_split[col].round(6).values).mean()
        else:
            match = (real[col].values == recon_split[col].values).mean()
        if match != 1.0:
            raise RuntimeError(
                f"{name}: column '{col}' only matches {match:.4%} of rows -- alignment is not "
                "trustworthy, refusing to write is_censored."
            )
    print(f"{name}: alignment verified 100% on {check_cols} ({len(real):,} rows)")

    real["is_censored"] = recon_split["is_censored"].values
    censored_rate = real["is_censored"].mean()
    print(f"  is_censored rate: {censored_rate:.2%}")

    tmp = path.with_suffix(".parquet.tmp")
    real.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"  wrote is_censored column -> {path}")


def main() -> None:
    recon = reconstruct_with_status()
    splits = split_recon(recon)
    for name, recon_split in splits.items():
        verify_and_apply(name, recon_split)
    print("\nDone. train/val/holdout.parquet now have an is_censored column.")
    print("NOTE: *_features_only.parquet deliberately untouched -- agent-facing, must stay "
          "free of anything derived from event/observed_time/loan_status.")


if __name__ == "__main__":
    main()
