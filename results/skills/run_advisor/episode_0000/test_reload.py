import pickle
import pandas as pd
import numpy as np

val_feat_only = pd.read_parquet('/home/oliversnavy/repos/ai-lending-abridged/data/processed/val_features_only.parquet')
with open('risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

preds = model.predict_default_proba(val_feat_only)
print(type(preds), preds.shape, np.isnan(preds).sum(), preds[:5])
assert preds.ndim == 1
assert len(preds) == len(val_feat_only)
assert not np.isnan(preds).any()
print("OK")
