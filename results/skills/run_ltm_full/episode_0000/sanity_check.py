import pickle
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

WORK = "/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0000"
DATA = "/home/oliversnavy/repos/ai-lending-abridged/data/processed"

with open(f"{WORK}/risk_model.pkl", "rb") as f:
    model = pickle.load(f)

Xfo = pd.read_parquet(f"{DATA}/val_features_only.parquet")
preds = model.predict_default_proba(Xfo)
print("shape:", preds.shape, "dtype:", preds.dtype)
print("len matches:", len(preds) == len(Xfo))
print("any nan:", np.isnan(preds).any(), "any inf:", np.isinf(preds).any())
print("min/max:", preds.min(), preds.max())

val = pd.read_parquet(f"{DATA}/val.parquet")
c = concordance_index(val["observed_time"].values, -preds, val["event"].astype(int).values)
print("c-index on val:", c)

# also check holdout_features_only runs without error
Hfo = pd.read_parquet(f"{DATA}/holdout_features_only.parquet")
preds_h = model.predict_default_proba(Hfo)
print("holdout shape:", preds_h.shape, "nan:", np.isnan(preds_h).any())
