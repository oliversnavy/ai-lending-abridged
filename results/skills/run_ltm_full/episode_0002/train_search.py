import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0002')
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from model_def import LEAKAGE_COLS, CAT_COLS, build_cat_maps, encode_features

DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'
train = pd.read_parquet(DATA + 'train.parquet')
val = pd.read_parquet(DATA + 'val.parquet')

cat_maps = build_cat_maps(train, CAT_COLS)
feature_cols = [c for c in train.columns if c not in LEAKAGE_COLS]

Xtr_full = encode_features(train, cat_maps, feature_cols)
Xval = encode_features(val, cat_maps, feature_cols)

# subsample train for search
rng = np.random.RandomState(0)
n = len(train)
idx = rng.choice(n, size=min(200000, n), replace=False)
Xtr = Xtr_full.iloc[idx]
train_s = train.iloc[idx]

label_cox = np.where(train_s['event'] == 1, train_s['observed_time'], -train_s['observed_time'])
dtrain_cox = xgb.DMatrix(Xtr, label=label_cox)
dval = xgb.DMatrix(Xval)

val_event = val['event'].values
val_time = val['observed_time'].values

results = []

configs = []
for max_depth in [3,4,5,6]:
    for eta in [0.01,0.03,0.05,0.1]:
        configs.append((max_depth, eta))

for max_depth, eta in configs:
    params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': max_depth,
        'eta': eta,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
    }
    bst = xgb.train(params, dtrain_cox, num_boost_round=600,
                     evals=[(dtrain_cox, 'train')],
                     early_stopping_rounds=30, verbose_eval=False)
    preds = bst.predict(dval, iteration_range=(0, bst.best_iteration+1))
    c = concordance_index(val_time, -preds, val_event)
    results.append({'type':'cox','max_depth':max_depth,'eta':eta,
                     'best_iter':bst.best_iteration,'c_index':c})
    print(results[-1], flush=True)

import json
with open('/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0002/search_results.json','w') as f:
    json.dump(results, f, indent=2)
