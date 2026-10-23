import pickle
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

import model_def  # noqa: F401 needed for unpickling

DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'

with open('risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

val_feat = pd.read_parquet(DATA + 'val_features_only.parquet')
val_full = pd.read_parquet(DATA + 'val.parquet')

preds = model.predict_default_proba(val_feat)
preds = np.asarray(preds)

print('shape', preds.shape, 'rows', len(val_feat))
assert preds.shape[0] == len(val_feat)
assert not np.isnan(preds).any()
assert not np.isinf(preds).any()

c = concordance_index(val_full['observed_time'], -preds, val_full['event'])
print('Harrell C on val:', c)
