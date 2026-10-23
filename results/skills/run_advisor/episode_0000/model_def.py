import numpy as np
import pandas as pd

DROP_COLS = ['event', 'observed_time', 'is_censored']

CAT_COLS = ['term', 'home_ownership', 'verification_status', 'purpose', 'zip_code',
            'addr_state', 'initial_list_status', 'application_type',
            'verification_status_joint', 'disbursement_method']

EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10,
}


def emp_length_to_num(x):
    if pd.isna(x):
        return np.nan
    return EMP_LENGTH_MAP.get(str(x).strip(), np.nan)


def earliest_cr_line_to_years(series, ref_year=2018):
    # format like 'Jan-2001' or similar; try parse year
    def parse(v):
        if pd.isna(v):
            return np.nan
        try:
            dt = pd.to_datetime(v, errors='coerce')
            if pd.isna(dt):
                return np.nan
            return ref_year - dt.year
        except Exception:
            return np.nan
    return series.apply(parse)


class FeatureEngineer:
    def __init__(self):
        self.cat_categories = {}
        self.fitted = False

    def fit(self, df):
        d = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore').copy()
        for c in CAT_COLS:
            if c in d.columns:
                self.cat_categories[c] = pd.Categorical(d[c].astype(str)).categories
        self.fitted = True
        return self

    def transform(self, df):
        d = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore').copy()

        # emp_length
        if 'emp_length' in d.columns:
            d['emp_length_num'] = d['emp_length'].apply(emp_length_to_num)
            d['emp_length_missing'] = d['emp_length_num'].isna().astype(int)
            d['emp_length_num'] = d['emp_length_num'].fillna(-1)
            d = d.drop(columns=['emp_length'])

        # fico midpoint
        if 'fico_range_low' in d.columns and 'fico_range_high' in d.columns:
            d['fico_midpoint'] = (d['fico_range_low'] + d['fico_range_high']) / 2.0

        if 'sec_app_fico_range_low' in d.columns and 'sec_app_fico_range_high' in d.columns:
            d['sec_app_fico_midpoint'] = (d['sec_app_fico_range_low'] + d['sec_app_fico_range_high']) / 2.0

        # dti winsorize
        if 'dti' in d.columns:
            d['dti'] = d['dti'].clip(lower=-5, upper=100)
        if 'dti_joint' in d.columns:
            d['dti_joint'] = d['dti_joint'].clip(lower=-5, upper=150)

        # earliest_cr_line -> credit history years
        for col in ['earliest_cr_line', 'sec_app_earliest_cr_line']:
            if col in d.columns:
                d[col + '_years'] = earliest_cr_line_to_years(d[col])
                d = d.drop(columns=[col])

        # revol_util cap
        if 'revol_util' in d.columns:
            d['revol_util'] = d['revol_util'].clip(lower=0, upper=200)

        # categorical encoding
        for c in CAT_COLS:
            if c in d.columns:
                cats = self.cat_categories.get(c)
                d[c] = pd.Categorical(d[c].astype(str), categories=cats).codes
                d[c] = d[c].astype(float)
                d.loc[d[c] < 0, c] = -1

        # remaining object/str columns -> try numeric else drop
        for c in d.columns:
            if d[c].dtype == object or str(d[c].dtype) == 'str':
                d[c] = pd.to_numeric(d[c], errors='coerce')

        # missing indicators for numeric columns with many NaNs, then fillna
        for c in d.columns:
            if d[c].isna().any():
                d[c + '_isna'] = d[c].isna().astype(int)
                d[c] = d[c].fillna(-999)

        d = d.astype(float)
        return d


class XGBRiskModel:
    def __init__(self, booster, feature_names, fe: FeatureEngineer, flip_sign=False):
        self.booster = booster
        self.feature_names = feature_names
        self.fe = fe
        self.flip_sign = flip_sign

    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        feats = self.fe.transform(X)
        # align columns
        for c in self.feature_names:
            if c not in feats.columns:
                feats[c] = -999.0
        feats = feats[self.feature_names]
        dmat = xgb.DMatrix(feats)
        score = self.booster.predict(dmat)
        if self.flip_sign:
            score = -score
        return np.asarray(score, dtype=float)
