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

rng = np.random.RandomState(0)
n = len(train)
idx = rng.choice(n, size=min(200000, n), replace=False)
Xtr = Xtr_full.iloc[idx]
train_s = train.iloc[idx]

dtrain = xgb.DMatrix(Xtr, label=train_s['event'].values)
dval = xgb.DMatrix(Xval)

val_event = val['event'].values
val_time = val['observed_time'].values

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
}
bst = xgb.train(params, dtrain, num_boost_round=600,
                 evals=[(dtrain, 'train')],
                 early_stopping_rounds=30, verbose_eval=False)
preds = bst.predict(dval, iteration_range=(0, bst.best_iteration+1))
c = concordance_index(val_time, -preds, val_event)
print({'type':'binary','c_index':c, 'best_iter': bst.best_iteration})
