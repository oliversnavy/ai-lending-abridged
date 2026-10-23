import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression

from model_def import RiskModel

DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'

train = pd.read_parquet(DATA + 'train.parquet')
val = pd.read_parquet(DATA + 'val.parquet')
val_feat = pd.read_parquet(DATA + 'val_features_only.parquet')

exclude = {'event', 'observed_time', 'is_censored', 'issue_year',
           'grade', 'sub_grade', 'int_rate', 'installment'}
feature_cols = [c for c in val_feat.columns if c not in exclude]
print('n features', len(feature_cols))

cat_cols = [c for c in feature_cols if train[c].dtype == object or str(train[c].dtype) == 'str' or train[c].dtype.name == 'category']
# detect via dtype
cat_cols = [c for c in feature_cols if train[c].dtype == 'object' or pd.api.types.is_string_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in cat_cols]
print('cat_cols', cat_cols)

cat_categories = {}
for c in cat_cols:
    cats = sorted(train[c].astype(str).replace('nan', np.nan).dropna().unique().tolist())
    if len(cats) == 0:
        cats = ['__MISSING__']
    cat_categories[c] = cats

def prep(df, feature_cols, cat_cols, num_cols, cat_categories):
    X = df.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c].astype(str).replace('nan', np.nan), categories=cat_categories[c])
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X

X_train = prep(train, feature_cols, cat_cols, num_cols, cat_categories)

# Cox label convention: positive = observed_time for event, negative = -observed_time for censored
labels = np.where(train['event'] == 1, train['observed_time'], -train['observed_time'])

dtrain = xgb.DMatrix(X_train, label=labels, enable_categorical=True)

params = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
}

booster = xgb.train(params, dtrain, num_boost_round=400)

X_val = prep(val_feat, feature_cols, cat_cols, num_cols, cat_categories)
dval = xgb.DMatrix(X_val, enable_categorical=True)
risk_cox = booster.predict(dval)

c_cox = concordance_index(val['observed_time'], -risk_cox, val['event'])
print('Cox c-index:', c_cox)

# Baseline: logistic regression on event, simple encoding (frequency encode cats, fillna median)
X_train_lr = X_train.copy()
freqs = {}
for c in cat_cols:
    X_train_lr[c] = X_train_lr[c].astype(object)
    freq = X_train_lr[c].value_counts(normalize=True)
    freqs[c] = freq
    X_train_lr[c] = X_train_lr[c].map(freq).fillna(0).astype(float)
medians = {}
for c in num_cols:
    med = X_train_lr[c].astype(float).median()
    if pd.isna(med):
        med = 0.0
    medians[c] = med
    X_train_lr[c] = X_train_lr[c].fillna(med)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_lr, train['event'])

X_val_lr = X_val.copy()
for c in cat_cols:
    X_val_lr[c] = X_val_lr[c].astype(object)
    X_val_lr[c] = X_val_lr[c].map(freqs[c]).fillna(0).astype(float)
for c in num_cols:
    X_val_lr[c] = pd.to_numeric(X_val_lr[c], errors='coerce').fillna(medians[c])

risk_lr = lr.predict_proba(X_val_lr)[:, 1]
c_lr = concordance_index(val['observed_time'], -risk_lr, val['event'])
print('LR c-index:', c_lr)

winner = 'cox' if c_cox >= c_lr else 'lr'
print('Winner:', winner)

if winner == 'cox':
    model = RiskModel(booster, feature_cols, cat_cols, num_cols, cat_categories)
else:
    # wrap LR similarly - but RiskModel is for xgb; build quick wrapper inline not needed since cox expected to win
    raise SystemExit('LR won unexpectedly - need separate wrapper')

with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)

import json
results = {
    'approach': 'XGBoost survival:cox model trained on observed_time/event with categorical native handling; '
                'compared against logistic regression baseline on event with frequency-encoded categoricals.',
    'hypothesis': f'Cox survival model directly optimizes concordance-like objective and handles censoring; expected to outperform naive logistic regression on event label. '
                  f'Local c-index: cox={c_cox:.4f}, logistic_regression={c_lr:.4f}. Saved winner={winner}.',
    'c_stat_cox': c_cox,
    'c_stat_logistic': c_lr,
    'winner': winner,
    'feature_cols': feature_cols,
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('done')
