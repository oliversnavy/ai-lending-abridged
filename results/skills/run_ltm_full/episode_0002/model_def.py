import numpy as np
import pandas as pd
import xgboost as xgb

LEAKAGE_COLS = ['grade', 'sub_grade', 'int_rate', 'installment',
                'event', 'observed_time', 'is_censored']

CAT_COLS = ['term', 'emp_length', 'home_ownership', 'verification_status',
            'purpose', 'zip_code', 'addr_state', 'earliest_cr_line',
            'initial_list_status', 'application_type',
            'verification_status_joint', 'sec_app_earliest_cr_line',
            'disbursement_method']


def build_cat_maps(df, cat_cols):
    maps = {}
    for c in cat_cols:
        if c in df.columns:
            uniques = df[c].astype('category').cat.categories
            maps[c] = {v: i for i, v in enumerate(uniques)}
    return maps


def encode_features(df, cat_maps, feature_cols):
    df = df.copy()
    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    base = df.drop(columns=drop_cols, errors='ignore')

    parts = []
    for c in feature_cols:
        if c in cat_maps:
            m = cat_maps[c]
            if c in base.columns:
                col = base[c].map(m).fillna(-1).astype('float64')
            else:
                col = pd.Series(-1.0, index=base.index)
            parts.append(col.rename(c))
        else:
            if c in base.columns:
                col = pd.to_numeric(base[c], errors='coerce')
            else:
                col = pd.Series(np.nan, index=base.index)
            parts.append(col.rename(c))
    X = pd.concat(parts, axis=1)
    return X


class RiskModel:
    def __init__(self, booster, cat_maps, feature_cols):
        self.booster = booster
        self.cat_maps = cat_maps
        self.feature_cols = feature_cols

    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xenc = encode_features(X, self.cat_maps, self.feature_cols)
        dmat = xgb.DMatrix(Xenc)
        preds = self.booster.predict(dmat)
        return np.asarray(preds, dtype='float64')
