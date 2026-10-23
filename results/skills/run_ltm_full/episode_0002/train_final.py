import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0002')
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from lifelines.utils import concordance_index
from model_def import LEAKAGE_COLS, CAT_COLS, build_cat_maps, encode_features, RiskModel

DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'
train = pd.read_parquet(DATA + 'train.parquet')
val = pd.read_parquet(DATA + 'val.parquet')

combined = pd.concat([train, val], axis=0, ignore_index=True)

cat_maps = build_cat_maps(combined, CAT_COLS)
feature_cols = [c for c in combined.columns if c not in LEAKAGE_COLS]

Xcomb = encode_features(combined, cat_maps, feature_cols)
label = np.where(combined['event'] == 1, combined['observed_time'], -combined['observed_time'])
dtrain = xgb.DMatrix(Xcomb, label=label)

params = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'max_depth': 5,
    'eta': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
}
bst = xgb.train(params, dtrain, num_boost_round=600)

model = RiskModel(bst, cat_maps, feature_cols)

# sanity check on val (in-sample now, but just for orientation check)
Xval_check = encode_features(val, cat_maps, feature_cols)
preds_val = bst.predict(xgb.DMatrix(Xval_check))
c = concordance_index(val['observed_time'].values, -preds_val, val['event'].values)
print('in-sample-ish val c-index (val included in training):', c)

with open('/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0002/risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('saved risk_model.pkl')
