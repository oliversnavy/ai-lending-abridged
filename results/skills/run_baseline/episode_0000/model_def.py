import numpy as np
import pandas as pd


class RiskModel:
    """Wraps an XGBoost Cox-survival booster with feature preprocessing."""

    def __init__(self, booster, feature_cols, cat_cols, num_cols, cat_categories):
        self.booster = booster
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_categories = cat_categories  # dict col -> list of categories seen in training

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # ensure all expected columns exist
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.feature_cols]

        for c in self.cat_cols:
            cats = self.cat_categories[c]
            X[c] = pd.Categorical(X[c].astype(str).replace('nan', np.nan), categories=cats)

        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors='coerce')

        return X

    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        Xp = self._prepare(X)
        dmat = xgb.DMatrix(Xp, enable_categorical=True)
        risk = self.booster.predict(dmat)
        return np.asarray(risk, dtype=float)
