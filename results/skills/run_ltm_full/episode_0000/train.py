import sys
sys.path.insert(0, "/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0000")

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression

from model_def import RiskModel, DROP_COLS

DATA = "/home/oliversnavy/repos/ai-lending-abridged/data/processed"
WORK = "/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0000"

train = pd.read_parquet(f"{DATA}/train.parquet")
val = pd.read_parquet(f"{DATA}/val.parquet")

feature_cols_all = [c for c in train.columns if c not in DROP_COLS]

cat_cols = [c for c in feature_cols_all if train[c].dtype == object or str(train[c].dtype) == "str"]
num_cols = [c for c in feature_cols_all if c not in cat_cols]

print("n cat:", len(cat_cols), "n num:", len(num_cols))

# build cat maps from train
cat_maps = {}
for c in cat_cols:
    vals = train[c].astype(str).fillna("__missing__").unique()
    cat_maps[c] = {v: i for i, v in enumerate(vals)}

num_medians = {c: train[c].median() for c in num_cols}

feature_cols = num_cols + cat_cols

def prep(df):
    out = pd.DataFrame(index=df.index)
    for c in num_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(num_medians[c])
    for c in cat_cols:
        out[c] = df[c].astype(str).map(cat_maps[c]).fillna(-1).astype(int)
    return out[feature_cols]

Xtr = prep(train)
ytr = train["event"].astype(int).values

Xval = prep(val)
yval_event = val["event"].astype(int).values
yval_time = val["observed_time"].values

# XGBoost classifier (event as target, censored as class 0)
dtrain = xgb.DMatrix(Xtr, label=ytr)
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "seed": 42,
}
booster = xgb.train(params, dtrain, num_boost_round=300)

dval = xgb.DMatrix(Xval)
xgb_preds = booster.predict(dval)

c_xgb = concordance_index(yval_time, -xgb_preds, yval_event)
print("XGB c-index:", c_xgb)

# Logistic regression baseline
Xtr_lr = Xtr.fillna(0.0).replace([np.inf, -np.inf], 0.0)
Xval_lr = Xval.fillna(0.0).replace([np.inf, -np.inf], 0.0)
lr = LogisticRegression(max_iter=200)
lr.fit(Xtr_lr, ytr)
lr_preds = lr.predict_proba(Xval_lr)[:, 1]
c_lr = concordance_index(yval_time, -lr_preds, yval_event)
print("LR c-index:", c_lr)

print("Chosen model:", "XGB" if c_xgb >= c_lr else "LR")

# Save chosen model (expect XGB better) as RiskModel
model = RiskModel(booster=booster, feature_cols=feature_cols, cat_cols=cat_cols,
                   num_cols=num_cols, cat_maps=cat_maps, num_medians=num_medians)

with open(f"{WORK}/risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved risk_model.pkl")

# sanity check using the RiskModel wrapper directly
preds_check = model.predict_default_proba(val.drop(columns=["event", "observed_time", "is_censored"]))
c_check = concordance_index(yval_time, -preds_check, yval_event)
print("RiskModel wrapper c-index check:", c_check)
