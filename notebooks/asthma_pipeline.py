"""
Shared pipeline components for the R3 revision (single source of truth).

Both 04_model.ipynb and (after the production run) 05_top10_sensitivity.ipynb
import from this module, so the cleaner / feature engineer / exclusion lists
can no longer drift between notebooks.

Decisions implemented here, agreed with K. Micheals 2026-08-14:
  - Primary-model exclusions: prior-diagnosis/treatment proxies,
    healthcare-utilization proxies (HUQ050), NHANES routing/eligibility
    variables (ENQ020, SPDBRONC).
  - PFQ020 RETAINED in the primary model (Khamron's position; Yaseen has
    not weighed in - recorded in the exclusion log).
  - CDC BMI-for-age z-score is the sole continuous BMI predictor.
    Raw BMXBMI, bmi_log, and weight-status categories leave the predictor
    set (categories remain descriptive, in Table 1 via build_table1.py).
  - The fixed 0.80 FEV1/FVC obstruction indicator is removed.
  - Spirometry-availability indicators are protected through feature
    selection (ProtectedSelectKBest) so the 718 children missing baseline
    spirometry remain visible to the model.
  - Correlation pruning (|r| > 0.90) is executed on training data only.
  - Imputation and scaling live INSIDE the model pipelines so they are
    re-fit within each CV fold (no fold leakage).
  - mutual_info selection is seeded for reproducibility.

Model fitting remains UNWEIGHTED pending a coauthor decision on survey
weights; weighted evaluation is reported alongside (see notebook 04).
"""
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from pediatric_corrections import (
    safe_indicator,          # noqa: F401  (re-exported for notebooks)
    safe_interaction,
    cdc_bmi_z,
    prune_correlated,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Exclusion lists (audit trail: email of 2026-08-12, reply of 2026-08-14)
# ---------------------------------------------------------------------------

# Asked because of a prior asthma diagnosis or its treatment.
LEAKY_PROXIES = ["MCQ025", "MCQ035", "MCQ040", "MCQ050", "MCQ051"]

AGE_RESTRICTED_VARS = [
    "ECQ020", "ECQ080", "ECQ090", "WHQ030E", "MCQ080E",
    "ECQ150", "ECD010", "ECD070A", "ECD070B",
    "FSD670ZC", "FSQ690", "FSD680", "FSD675",
]

IDENTIFIERS = ["SEQN"]

# NHANES protocol routing / eligibility, not physiology of the child.
# [2026-08-24 codebook adjudication] SPXNSTAT (spirometry status/quality
# flag), SPQ060 and SPQ100 (spirometry-session questionnaire) added under
# the same rationale as SPDBRONC/ENQ020.
# [2026-08-25, pending KM ratification] SPDNACC (Baseline Number of
# Acceptable Curves, ATS criteria) — same quality-metric class as SPXNSTAT.
PROTOCOL_ROUTING_VARS = ["ENQ020", "SPDBRONC", "SPXNSTAT", "SPQ060", "SPQ100",
                         "SPDNACC"]

# Diagnostic-opportunity proxies: excluded from the primary model, added
# back ONLY in the declared exploratory sensitivity analysis.
# [2026-08-24 codebook adjudication] HUQ071 (overnight hospital stay),
# HUQ090 (saw mental-health professional), HUQ030 (usual source of care)
# added under the same rationale as HUQ050.
UTILIZATION_PROXIES = ["HUQ050", "HUQ071", "HUQ090", "HUQ030"]

# PFQ020 deliberately NOT listed: retained per Khamron (2026-08-14).

PRIMARY_MODEL_EXCLUSIONS = PROTOCOL_ROUTING_VARS + UTILIZATION_PROXIES

# Kept regardless of univariate selection score.
PROTECTED_FEATURES = ["SPXNFEV1_missing", "SPXNFVC_missing"]

# Never dropped by correlation pruning.
PRUNE_PROTECT = (
    "bmi_z_cdc",
    "fev1_fvc_ratio",
    "family_spirometry_interaction",
    "SPXNFEV1_missing",
    "SPXNFVC_missing",
    "LBXCOT_missing",
    "BMXBMI_missing",
)

PRUNE_THRESHOLD = 0.90


def mutual_info_seeded(X, y):
    """Deterministic mutual_info_classif (module-level, so it pickles)."""
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Cleaning — codebook-verified missing-value handling [rewritten 2026-08-25]
#
# The previous version blanket-replaced {99,999,777,7777,9999} in every
# continuous variable and {7,9,77}+big in every categorical. Verified against
# the NHANES codebooks, both rules were wrong:
#   - Continuous exam/lab/derived variables code missing as blank and carry
#     NO numeric sentinels (e.g. RIDAGEEX_H=99 months, FEV=999 mL, weight
#     99 kg, urinary creatinine 99 are measurements). 99 real values were
#     being erased.
#   - Categorical variables with wide code ranges have 7/9 as VALID codes
#     (DMDEDUC3 7th/9th grade; INDHHIN2/INDFMIN2 income brackets;
#     DMDHHSIZ/DMDFMSIZ "7 or more"; SPDNACC curve counts). ~3,900 real
#     values were being erased. Refused/Don't-know codes match the code
#     width: {7,9} for 1-6-coded items, {77,99} for wider, etc.
# ---------------------------------------------------------------------------

# Per-variable sentinel sets verified against NHANES codebooks (DEMO_F,
# SPX_F, ...). Variables listed here bypass the width rule entirely.
CATEGORICAL_SENTINELS = {
    "DMDEDUC3": {77, 99},   # 0-15 grades (+55/66 special levels) valid
    "INDHHIN2": {77, 99},   # income brackets 1-15 valid
    "INDFMIN2": {77, 99},
    "DMDHHSIZ": set(),      # 1-7 valid, 7 = "7 or more"; no refused codes
    "DMDFMSIZ": set(),
    "SPDNACC":  set(),      # 0-9 curve counts; no refused codes
}

_SENTINEL_PAIRS = [(7, 9), (77, 99), (777, 999), (7777, 9999)]
_ALL_SENTINEL_CODES = {c for p in _SENTINEL_PAIRS for c in p}


class NHANESCleaner(BaseEstimator, TransformerMixin):
    """Codebook-verified NHANES missing-value handling.

    Continuous variables: no value replacement (NHANES codes missing as
    blank). Categorical variables: refused/don't-know codes only, taken
    from CATEGORICAL_SENTINELS when listed, otherwise from the width rule
    (the sentinel pair strictly above the training data's maximum
    non-sentinel code). If an unlisted variable shows a sentinel-looking
    code at implausibly high frequency (>2%, refused/DK are rare), fit()
    raises so the variable is adjudicated against its codebook instead of
    silently mangled. Binary (1/2) variables: 7/9 scrubbed defensively,
    then recoded to 1/0 with missingness preserved.
    """

    AMBIGUITY_FREQ = 0.02

    def __init__(self):
        self.continuous_cols_ = None
        self.categorical_cols_ = None
        self.binary_cols_ = None
        self.sentinel_map_ = None

    def fit(self, X, y=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        continuous_whitelist = {
            'BMXBMI', 'BMXHT', 'BMXWT', 'BMXWAIST', 'BMXLEG', 'BMXARML', 'BMXARMC',
            'SPXNFEV1', 'SPXNFVC', 'SPXNFEV3', 'SPXNFEV6', 'SPXNPEF', 'SPXBFEV1', 'SPXBFVC',
            'LBXCOT', 'LBXWBCSI', 'LBXEOPCT', 'INDFMPIR', 'RIDAGEYR'
        }

        self.continuous_cols_ = []
        self.categorical_cols_ = []
        self.binary_cols_ = []
        self.sentinel_map_ = {}

        for col in X_df.columns:
            s = X_df[col].dropna()
            if col in continuous_whitelist:
                self.continuous_cols_.append(col)
            elif len(s) > 0 and pd.api.types.is_numeric_dtype(X_df[col]):
                unique_vals = set(s.unique())
                if unique_vals == {1, 2} or unique_vals == {1.0, 2.0}:
                    self.binary_cols_.append(col)
                elif np.all(np.abs(s - np.round(s)) < 1e-6) and len(s.unique()) <= 20:
                    self.categorical_cols_.append(col)
                else:
                    self.continuous_cols_.append(col)
            else:
                self.continuous_cols_.append(col)

        for col in self.categorical_cols_:
            if col in CATEGORICAL_SENTINELS:
                self.sentinel_map_[col] = set(CATEGORICAL_SENTINELS[col])
                continue
            obs = set(int(v) for v in X_df[col].dropna().unique())
            nonsent_max = max((v for v in obs if v not in _ALL_SENTINEL_CODES),
                              default=0)
            pair = next((set(p) for p in _SENTINEL_PAIRS if p[0] > nonsent_max),
                        set())
            n = len(X_df[col].dropna())
            for code in sorted(pair & obs):
                freq = (X_df[col] == code).sum() / max(n, 1)
                if freq > self.AMBIGUITY_FREQ:
                    raise ValueError(
                        f"NHANESCleaner: '{col}' has sentinel-looking code "
                        f"{code} at {freq:.1%} frequency - refused/don't-know "
                        f"codes are rare, so this may be a valid value. Check "
                        f"the NHANES codebook and add '{col}' to "
                        f"CATEGORICAL_SENTINELS explicitly.")
            self.sentinel_map_[col] = pair

        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # continuous: no replacement — NHANES codes missing as blank
        for col in self.categorical_cols_:
            if col in X_df.columns and self.sentinel_map_.get(col):
                X_df[col] = X_df[col].replace(self.sentinel_map_[col], np.nan)

        for col in self.binary_cols_:
            if col in X_df.columns:
                X_df[col] = X_df[col].replace({7, 9}, np.nan)
                original_na = X_df[col].isna()
                X_df[col] = (X_df[col] == 1).astype(float)
                X_df.loc[original_na, col] = np.nan

        return X_df


# ---------------------------------------------------------------------------
# Feature engineering (R3-corrected)
# ---------------------------------------------------------------------------

class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create clinical features - R3 revision.

    Changes vs. the submitted version:
      - obstruction_indicator (fixed 0.80 threshold) REMOVED
      - bmi_log REMOVED; raw BMXBMI dropped after computing bmi_z_cdc,
        so the z-score is the sole continuous BMI predictor
      - 'obese' category removed from the predictor set (descriptive only)
      - interactions preserve missingness (safe_interaction)
    """

    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Spirometry features (continuous ratio retained; the fixed-threshold
        # obstruction indicator was removed for R3 - not defensible in children)
        if 'SPXNFEV1' in X_df.columns and 'SPXNFVC' in X_df.columns:
            mask = (X_df['SPXNFEV1'].notna() & X_df['SPXNFVC'].notna() & (X_df['SPXNFVC'] > 0))
            X_df['fev1_fvc_ratio'] = np.nan
            X_df.loc[mask, 'fev1_fvc_ratio'] = X_df.loc[mask, 'SPXNFEV1'] / X_df.loc[mask, 'SPXNFVC']
            X_df['fev1_log'] = np.log1p(X_df['SPXNFEV1'])

        # Cotinine features
        if 'LBXCOT' in X_df.columns:
            X_df['cotinine_log'] = np.log1p(X_df['LBXCOT'])
            X_df['smoke_exposure_heavy'] = (X_df['LBXCOT'] > 10.0).astype(float)
            X_df['likely_active_smoking'] = (X_df['LBXCOT'] > 50).astype(float)

        # BMI: CDC z-score is the sole continuous BMI predictor
        if 'BMXBMI' in X_df.columns:
            _z_cdc, _pct_cdc = cdc_bmi_z(
                X_df['BMXBMI'], X_df.get('RIDAGEEX_H'), X_df.get('RIAGENDR')
            )
            X_df['bmi_z_cdc'] = _z_cdc

        # Age features
        if 'RIDAGEYR' in X_df.columns:
            X_df['age_squared'] = X_df['RIDAGEYR'] ** 2
            X_df['age_log'] = np.log1p(X_df['RIDAGEYR'])

        # Family history interaction (missingness-preserving)
        if 'MCQ300B' in X_df.columns and 'fev1_fvc_ratio' in X_df.columns:
            X_df['family_spirometry_interaction'] = safe_interaction(
                X_df['MCQ300B'], X_df['fev1_fvc_ratio']
            )

        # Missing indicators (BEFORE dropping raw BMXBMI)
        for var in ['SPXNFEV1', 'SPXNFVC', 'LBXCOT', 'BMXBMI']:
            if var in X_df.columns:
                X_df[f'{var}_missing'] = X_df[var].isna().astype(int)

        # Drop redundant BMI forms from the predictor set
        X_df = X_df.drop(columns=[c for c in ('BMXBMI',) if c in X_df.columns])

        self.feature_names_ = X_df.columns.tolist()
        return X_df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_ if self.feature_names_ is not None else []


# ---------------------------------------------------------------------------
# Feature selection that cannot drop the availability indicators
# ---------------------------------------------------------------------------

class ProtectedSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest that always retains `protect` features.

    Selects the top-k features by `score_func`, then adds any protected
    features not already selected (so the output has k to k+len(protect)
    columns). `feature_names` gives the column order of the array this
    step receives.
    """

    def __init__(self, score_func=f_classif, k=20, feature_names=None, protect=()):
        self.score_func = score_func
        self.k = k
        self.feature_names = feature_names
        self.protect = protect

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        n_feat = X_arr.shape[1]
        scores = self.score_func(X_arr, y)
        if isinstance(scores, tuple):        # f_classif returns (F, p)
            scores = scores[0]
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)

        k = min(self.k, n_feat)
        top_idx = set(np.argsort(scores)[::-1][:k].tolist())

        if self.feature_names is not None:
            name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
            for name in self.protect:
                if name in name_to_idx and name_to_idx[name] < n_feat:
                    top_idx.add(name_to_idx[name])

        self.selected_idx_ = np.array(sorted(top_idx))
        self.scores_ = scores
        self.n_features_in_ = n_feat
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_idx_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_idx_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_idx_] = True
        return mask

    def get_selected_names(self):
        if self.feature_names is None:
            return [str(i) for i in self.selected_idx_]
        return [self.feature_names[i] for i in self.selected_idx_]


# ---------------------------------------------------------------------------
# Resampling — SMOTENC-based, per KM sign-off 2026-08-24
# ---------------------------------------------------------------------------

class AutoSMOTENCENN(BaseEstimator):
    """SMOTE-ENN with the SMOTE step replaced by categorical-aware SMOTENC.

    Agreed with K. Micheals 2026-08-24: integer-coded categorical predictors
    (race/ethnicity, language, citizenship, ...) must not receive synthetic
    between-category values, so SMOTENC replaces plain SMOTE; the ENN
    cleaning step is retained for comparability with the prior pipeline.

    Because resampling runs AFTER feature selection and fold membership
    varies, categorical columns cannot be hardcoded. They are inferred at
    resample time: columns taking <= max_categorical_unique distinct values
    (the same discreteness logic NHANESCleaner uses). Binary indicators and
    small integer codes qualify even after robust scaling, since scaling
    preserves the number of distinct values. Falls back to plain SMOTE in
    the degenerate cases (no categorical columns, or no continuous ones).
    """

    def __init__(self, random_state=RANDOM_STATE, max_categorical_unique=20):
        self.random_state = random_state
        self.max_categorical_unique = max_categorical_unique

    def fit_resample(self, X, y):
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE, SMOTENC

        X_arr = np.asarray(X)
        cat_idx = [j for j in range(X_arr.shape[1])
                   if len(np.unique(X_arr[:, j])) <= self.max_categorical_unique]
        if cat_idx and len(cat_idx) < X_arr.shape[1]:
            smote = SMOTENC(categorical_features=cat_idx,
                            random_state=self.random_state)
        else:
            smote = SMOTE(random_state=self.random_state)
        self.categorical_indices_ = cat_idx
        self.used_smotenc_ = bool(cat_idx) and len(cat_idx) < X_arr.shape[1]
        return SMOTEENN(smote=smote,
                        random_state=self.random_state).fit_resample(X_arr, y)


# ---------------------------------------------------------------------------
# Pipeline building blocks
# ---------------------------------------------------------------------------

def preprocessing_steps():
    """Imputer + scaler steps to place at the FRONT of every model pipeline,
    so they are re-fit inside each CV fold (no fold leakage)."""
    return [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ]


def apply_correlation_pruning(X_train_feat, X_val_feat=None, X_test_feat=None):
    """Fit pruning on TRAINING data only; apply the same drops elsewhere.

    Returns (X_train_pruned, X_val_pruned, X_test_pruned, dropped_list).
    """
    X_train_pruned, dropped = prune_correlated(
        X_train_feat, threshold=PRUNE_THRESHOLD, protect=PRUNE_PROTECT
    )
    out = [X_train_pruned]
    for X_other in (X_val_feat, X_test_feat):
        if X_other is None:
            out.append(None)
        else:
            out.append(X_other.drop(columns=[c for c in dropped if c in X_other.columns]))
    out.append(dropped)
    return tuple(out)


def weighted_binary_metrics(y_true, y_prob, sample_weight=None, threshold=0.5):
    """AUC / sensitivity / specificity, optionally survey-weighted."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    y_hat = (np.asarray(y_prob) >= threshold).astype(float)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    def _wmean(mask_num, mask_den):
        if w is None:
            den = mask_den.sum()
            return float(mask_num.sum() / den) if den else float('nan')
        den = w[mask_den].sum()
        return float(w[mask_num].sum() / den) if den else float('nan')

    pos, neg = y_true == 1, y_true == 0
    return {
        'auc': float(roc_auc_score(y_true, y_prob, sample_weight=w)),
        'sensitivity': _wmean((y_hat == 1) & pos, pos),
        'specificity': _wmean((y_hat == 0) & neg, neg),
        'threshold': threshold,
        'weighted': w is not None,
        'n': int(len(y_true)),
    }
