import numpy as np
import pandas as pd

LEAK_COLS = ['grade', 'sub_grade', 'int_rate', 'installment']
NON_FEATURE_COLS = ['event', 'observed_time', 'is_censored']


class RiskModel:
    def __init__(self, booster, feature_cols, cat_cols, cat_maps, objective):
        self.booster = booster
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.cat_maps = cat_maps
        self.objective = objective

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        drop = [c for c in LEAK_COLS + NON_FEATURE_COLS if c in X.columns]
        X = X.drop(columns=drop, errors='ignore')
        # ensure all expected feature cols present
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan

        encoded_parts = []
        for c in self.cat_cols:
            mapping = self.cat_maps[c]
            col = X[c].astype(str) if c in X.columns else pd.Series(['nan'] * len(X), index=X.index)
            mapped = col.map(mapping).fillna(-1).astype(np.float64)
            mapped.name = c
            encoded_parts.append(mapped)

        num_cols = [c for c in self.feature_cols if c not in self.cat_cols]
        num_df = X[num_cols].apply(pd.to_numeric, errors='coerce')

        full = pd.concat([num_df] + encoded_parts, axis=1)
        full = full[self.feature_cols]
        return full

    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        Xp = self._prep(X)
        dmat = xgb.DMatrix(Xp)
        preds = self.booster.predict(dmat)
        preds = np.asarray(preds, dtype=np.float64)
        if self.objective == 'survival:cox':
            # higher hazard ratio = higher risk already
            score = preds
        else:
            score = preds
        # normalize to [0,1]-ish via rank-safe sigmoid-free scaling not needed; just return score
        return score
