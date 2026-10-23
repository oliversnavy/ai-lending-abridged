import pickle
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

WD = '/home/oliversnavy/repos/ai-lending-abridged/results/skills/run_ltm_full/episode_0002/'
DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'

with open(WD + 'risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

val_feat = pd.read_parquet(DATA + 'val_features_only.parquet')
val_full = pd.read_parquet(DATA + 'val.parquet')

preds = model.predict_default_proba(val_feat)
preds = np.asarray(preds)
print('shape', preds.shape, 'nrows', len(val_feat))
assert preds.shape[0] == len(val_feat)
assert np.isfinite(preds).all()
print('range', preds.min(), preds.max())

c = concordance_index(val_full['observed_time'].values, -preds, val_full['event'].values)
print('final c-index on val:', c)

# orientation check: higher preds -> higher empirical default rate
q = pd.qcut(preds, 5, duplicates='drop')
print(val_full.groupby(q)['event'].mean())
