import time
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from model_def import RiskModel, LEAK_COLS, NON_FEATURE_COLS

DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'
train = pd.read_parquet(DATA + 'train.parquet')

feature_cols = [c for c in train.columns if c not in LEAK_COLS + NON_FEATURE_COLS]
cat_cols = [c for c in feature_cols if train[c].dtype == 'str' or train[c].dtype == object]
num_cols = [c for c in feature_cols if c not in cat_cols]

cat_maps = {}
for c in cat_cols:
    uniques = train[c].astype(str).unique()
    cat_maps[c] = {v: i for i, v in enumerate(uniques)}


def encode(df):
    parts = []
    for c in cat_cols:
        mapped = df[c].astype(str).map(cat_maps[c]).fillna(-1).astype(np.float64)
        mapped.name = c
        parts.append(mapped)
    num_df = df[num_cols].apply(pd.to_numeric, errors='coerce')
    full = pd.concat([num_df] + parts, axis=1)
    return full[feature_cols]


t0 = time.time()
Xtr = encode(train)
obs = train['observed_time'].values.astype(np.float64)
ev = train['event'].values.astype(np.float64)
lbl = np.where(ev == 1, obs, -obs)

dtrain = xgb.DMatrix(Xtr)
dtrain.set_label(lbl)

params = dict(objective='survival:cox', max_depth=4, eta=0.05, subsample=0.8)
bst = xgb.train(params, dtrain, num_boost_round=400)
print('train time', time.time() - t0)

model = RiskModel(bst, feature_cols, cat_cols, cat_maps, 'survival:cox')
with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('saved risk_model.pkl')
