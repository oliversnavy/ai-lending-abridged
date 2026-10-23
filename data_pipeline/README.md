# Data setup

This repo trains on the public LendingClub "accepted loans" dataset (2007–2018Q4).
Nothing proprietary — the same dataset many public credit-scoring notebooks use.

## 1. Download

Get `accepted_2007_to_2018Q4.csv.gz` from Kaggle (search "LendingClub Loan Data" / "all
lending club loan data") or another public mirror, and place it at:

```
data/accepted_2007_to_2018Q4.csv.gz
```

## 2. Build the splits

Run these from the repo root, in order:

```bash
uv run python data_pipeline/build_splits.py              # raw CSV -> train/val/holdout.parquet
uv run python data_pipeline/fix_holdout_zip_code_row.py   # drops 1 bad row (see script docstring) --
                                                            # MUST run before add_censoring_flag.py:
                                                            # that script's own row-alignment check
                                                            # assumes this fix already applied to holdout
uv run python data_pipeline/add_censoring_flag.py         # adds is_censored to all three
uv run python data_pipeline/build_features_only.py        # regenerate *_features_only.parquet
```

This produces, under `data/processed/`:

| File | Rows | Purpose |
|---|---|---|
| `train.parquet` | ~466k | loans issued ≤2014 — model training |
| `val.parquet` | ~855k | loans 2015–2016 — your own local testing/iteration |
| `holdout.parquet` | ~938k | loans ≥2017 — **never touched during an episode** |
| `val_features_only.parquet` | ~855k | what the harness actually feeds a submitted `risk_model.pkl` |
| `holdout_features_only.parquet` | ~938k | same, for the final held-out score |

## Why this split design

- **Temporal split** (train ≤2014, val 2015–16, holdout ≥2017), not a random split — a
  real lender never gets to train on the future, and a model that looks good under a
  random split can still fail under genuine temporal generalization.
- **`grade`/`sub_grade`/`int_rate`/`installment` excluded from training** — these are
  LendingClub's own retrospective risk categorization and pricing decision for the
  loan, not information that exists yet for a new applicant. See
  `build_features_only.py`'s module docstring for the full reasoning.
- **Censoring-aware targets** (`event`, `observed_time`, `is_censored`) instead of a
  naive binary "did this loan ever default" label — a loan still open at the data
  snapshot hasn't been *confirmed* non-default, it just hasn't defaulted *yet*. Scoring
  with Harrell's C (not `sklearn.roc_auc_score`) against these targets handles that
  correctly; see `agent/evaluator/evaluate_episode.py`.
