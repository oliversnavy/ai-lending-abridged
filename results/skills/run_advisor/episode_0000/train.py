import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from lifelines.utils import concordance_index
from model_def import FeatureEngineer, XGBRiskModel

DATA = '/home/oliversnavy/repos/ai-lending-abridged/data/processed/'

train = pd.read_parquet(DATA + 'train.parquet')
val = pd.read_parquet(DATA + 'val.parquet')
val_feat_only = pd.read_parquet(DATA + 'val_features_only.parquet')

fe = FeatureEngineer().fit(train)
X_train = fe.transform(train)
X_val = fe.transform(val_feat_only)

# align columns
all_cols = sorted(set(X_train.columns) | set(X_val.columns))
for c in all_cols:
    if c not in X_train.columns:
        X_train[c] = -999.0
    if c not in X_val.columns:
        X_val[c] = -999.0
X_train = X_train[all_cols]
X_val = X_val[all_cols]

feature_names = list(X_train.columns)

# ---- Cox model ----
label = np.where(train['event'] == 1, train['observed_time'], -train['observed_time'])
dtrain = xgb.DMatrix(X_train, label=label)
dval = xgb.DMatrix(X_val)

mono = []
for c in feature_names:
    if c == 'fico_midpoint':
        mono.append(-1)
    elif c in ('dti', 'revol_util'):
        mono.append(1)
    else:
        mono.append(0)

params_cox = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'max_depth': 5,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'monotone_constraints': '(' + ','.join(str(m) for m in mono) + ')',
}

cox_booster = xgb.train(params_cox, dtrain, num_boost_round=300)

cox_score = cox_booster.predict(dval)  # higher = higher hazard typically = riskier
c_cox = concordance_index(val['observed_time'], -cox_score, val['event'])
print('Cox C-stat (assuming higher=riskier):', c_cox)
c_cox_flipped = concordance_index(val['observed_time'], cox_score, val['event'])
print('Cox C-stat flipped:', c_cox_flipped)

flip_cox = c_cox_flipped > c_cox
cox_c_final = max(c_cox, c_cox_flipped)

# ---- Baseline binary classifier on resolved loans ----
resolved_mask = (train['event'] == 1) | (train['is_censored'] == False)
y_bin = train.loc[resolved_mask, 'event'].astype(int)
X_bin = X_train.loc[resolved_mask]

dtrain_bin = xgb.DMatrix(X_bin, label=y_bin)
params_bin = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
bin_booster = xgb.train(params_bin, dtrain_bin, num_boost_round=300)
bin_score = bin_booster.predict(dval)  # prob of default, higher = riskier
c_bin = concordance_index(val['observed_time'], -bin_score, val['event'])
print('Binary baseline C-stat:', c_bin)

print('cox_c_final', cox_c_final, 'flip_cox', flip_cox)

# pick winner
if cox_c_final >= c_bin:
    print('Winner: COX')
    winner_booster = cox_booster
    winner_flip = flip_cox
    winner_type = 'cox'
else:
    print('Winner: BINARY')
    winner_booster = bin_booster
    winner_flip = False
    winner_type = 'binary'

model = XGBRiskModel(winner_booster, feature_names, fe, flip_sign=winner_flip)

with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Saved risk_model.pkl, winner_type=', winner_type)

# quick self-test
preds = model.predict_default_proba(val_feat_only)
print('preds shape', preds.shape, 'nan count', np.isnan(preds).sum())
c_final = concordance_index(val['observed_time'], -preds, val['event'])
print('Final saved model C-stat:', c_final)

with open('report.txt', 'w') as f:
    f.write(f'cox_c_final={cox_c_final}\n')
    f.write(f'c_bin={c_bin}\n')
    f.write(f'winner={winner_type}\n')
    f.write(f'final_saved_c_stat={c_final}\n')
