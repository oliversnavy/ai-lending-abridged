import time
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index

from model_def import LEAK_COLS, NON_FEATURE_COLS

DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'

train = pd.read_parquet(DATA + 'train.parquet')
val = pd.read_parquet(DATA + 'val.parquet')

SUBSAMPLE = 200000
if len(train) > SUBSAMPLE:
    train_s = train.sample(SUBSAMPLE, random_state=42)
else:
    train_s = train

feature_cols = [c for c in train.columns if c not in LEAK_COLS + NON_FEATURE_COLS]
cat_cols = [c for c in feature_cols if train[c].dtype == 'str' or train[c].dtype == object]
num_cols = [c for c in feature_cols if c not in cat_cols]

print('n features', len(feature_cols), 'n cats', len(cat_cols))

# build cat maps from train uniques
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


Xtr = encode(train_s)
Xval = encode(val)

results = []

def run(objective, params, label):
    t0 = time.time()
    dtrain = xgb.DMatrix(Xtr)
    if objective == 'binary:logistic':
        dtrain.set_label(train_s['event'].values.astype(np.float64))
    else:
        obs = train_s['observed_time'].values.astype(np.float64)
        ev = train_s['event'].values.astype(np.float64)
        lbl = np.where(ev == 1, obs, -obs)
        dtrain.set_label(lbl)

    dval = xgb.DMatrix(Xval)

    full_params = dict(params)
    full_params['objective'] = objective
    bst = xgb.train(full_params, dtrain, num_boost_round=params.get('n_estimators', 200))
    preds = bst.predict(dval)
    c = concordance_index(val['observed_time'], -preds, val['event'])
    dt = time.time() - t0
    results.append((label, objective, params, c, dt))
    print(label, objective, params, 'C=', c, 'time=', dt)
    return bst, c


# Baseline binary:logistic
run('binary:logistic', dict(max_depth=5, eta=0.1, subsample=0.8, n_estimators=200), 'baseline_logistic')

# Cox baseline
run('survival:cox', dict(max_depth=5, eta=0.1, subsample=0.8, n_estimators=200), 'cox_base')

# small grid on cox (assume cox wins per task hint, but we'll check both)
grid = [
    dict(max_depth=3, eta=0.1, subsample=0.8, n_estimators=300),
    dict(max_depth=4, eta=0.05, subsample=0.8, n_estimators=400),
    dict(max_depth=6, eta=0.1, subsample=0.7, n_estimators=200),
    dict(max_depth=5, eta=0.1, subsample=1.0, n_estimators=250),
]
for i, g in enumerate(grid):
    run('survival:cox', g, f'cox_grid_{i}')

for i, g in enumerate(grid):
    run('binary:logistic', g, f'logistic_grid_{i}')

print('\n=== SUMMARY ===')
for label, obj, params, c, dt in sorted(results, key=lambda r: -r[3]):
    print(f'{label:20s} {obj:18s} C={c:.4f} time={dt:.1f}s params={params}')

import json
with open('compare_results.json', 'w') as f:
    json.dump([{'label': l, 'objective': o, 'params': p, 'c_index': c, 'time': t} for l, o, p, c, t in results], f, indent=2)
