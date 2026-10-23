"""
One-time fix (2026-08-14): drop the single holdout.parquet row with a null zip_code.

Audited (null_pattern_audit2.py, broad sweep over all 105 columns): zip_code is NEVER
null in train.parquet (2007-2014) or val.parquet (2015-2016) -- 0 exceptions in either --
but holdout.parquet (2017-2018) has exactly 1 null row (index 805006, an otherwise
ordinary AL/grade-C/2017 loan record with no other anomalies). This is the mirror image
of the sec_app_*/joint-column issue fixed earlier the same day (see
build_features_only.py's HOLDOUT_NULL_COLS): train+val give an agent zero opportunity to
ever observe a null zip_code, so code that assumes zip_code is always present (e.g.
`df['zip_code'].astype(str).str[:2]` to derive a state code) crashes on this one holdout
row in a way no amount of train/val-based learning could have anticipated.

Unlike the sec_app_ fix (nulling columns, still ~939k rows), this is a single missing
INPUT value with no defensible imputation -- dropping the row is simpler and more
defensible than inventing a zip code. Per-episode effect of removing 1 row out of 938,793
from the P&L/c-stat aggregate is well below any measurable threshold, so -- unlike the
sec_app_ fix -- this does NOT warrant a full retroactive re-scoring sweep of every
already-"ok" episode; only episodes that actually crashed on this specific row need
re-verification.

This script modifies holdout.parquet itself (the ground-truth source both
holdout_features_only.parquet and the harness's own scoring are derived from/matched
against). Row-positional alignment between holdout.parquet and holdout_features_only.parquet
means build_features_only.py MUST be re-run immediately after this script, so both files
drop the exact same row and stay aligned.

Usage:
    uv run python data_pipeline/fix_holdout_zip_code_row.py
    uv run python data_pipeline/build_features_only.py   # regenerate features-only files
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path("data/processed")
HOLDOUT_PATH = DATA_DIR / "holdout.parquet"


def main() -> None:
    df = pd.read_parquet(HOLDOUT_PATH)
    before = len(df)

    null_mask = df["zip_code"].isna()
    n_null = int(null_mask.sum())
    if n_null != 1:
        raise RuntimeError(
            f"Expected exactly 1 null zip_code row, found {n_null}. "
            "Re-run null_pattern_audit2.py-style checks before proceeding -- "
            "the data may have changed since this script was written."
        )

    dropped = df[null_mask]
    print("Dropping row:")
    print(dropped[["loan_amnt", "grade", "addr_state", "issue_year", "zip_code"]].to_string())

    fixed = df[~null_mask].reset_index(drop=True)

    tmp = HOLDOUT_PATH.with_suffix(".parquet.tmp")
    fixed.to_parquet(tmp, index=False)
    tmp.replace(HOLDOUT_PATH)

    print(f"\n{HOLDOUT_PATH}: {before:,} -> {len(fixed):,} rows")
    print("Now re-run: uv run python data_pipeline/build_features_only.py")


if __name__ == "__main__":
    main()
