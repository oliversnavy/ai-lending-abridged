import numpy as np
import pandas as pd

DROP_COLS = {"event", "observed_time", "is_censored", "grade", "sub_grade",
             "int_rate", "installment"}


class RiskModel:
    def __init__(self, booster, feature_cols, cat_cols, num_cols, cat_maps, num_medians):
        self.booster = booster
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_maps = cat_maps
        self.num_medians = num_medians

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in DROP_COLS:
            if c in X.columns:
                X = X.drop(columns=[c])

        out = pd.DataFrame(index=X.index)
        for c in self.num_cols:
            if c in X.columns:
                col = pd.to_numeric(X[c], errors="coerce")
            else:
                col = pd.Series(np.nan, index=X.index)
            out[c] = col.fillna(self.num_medians.get(c, 0.0))

        for c in self.cat_cols:
            mapping = self.cat_maps.get(c, {})
            if c in X.columns:
                col = X[c].astype(str)
            else:
                col = pd.Series("__missing__", index=X.index)
            out[c] = col.map(mapping).fillna(-1).astype(int)

        out = out[self.feature_cols]
        return out

    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        Xp = self._prep(X)
        dmat = xgb.DMatrix(Xp)
        preds = self.booster.predict(dmat)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        return preds.astype(float)
