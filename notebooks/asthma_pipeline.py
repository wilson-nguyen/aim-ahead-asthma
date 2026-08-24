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
PROTOCOL_ROUTING_VARS = ["ENQ020", "SPDBRONC", "SPXNSTAT", "SPQ060", "SPQ100"]

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
# Cleaning (unchanged behavior from notebook 04 cell 6)
# ---------------------------------------------------------------------------

class NHANESCleaner(BaseEstimator, TransformerMixin):
    """Clean NHANES sentinel codes."""

    def __init__(self):
        self.continuous_cols_ = None
        self.categorical_cols_ = None
        self.binary_cols_ = None

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

        for col in X_df.columns:
            if col in continuous_whitelist:
                self.continuous_cols_.append(col)
            else:
                s = X_df[col].dropna()
                if len(s) > 0 and pd.api.types.is_numeric_dtype(X_df[col]):
                    unique_vals = set(s.unique())
                    if unique_vals == {1, 2} or unique_vals == {1.0, 2.0}:
                        self.binary_cols_.append(col)
                    elif np.all(np.abs(s - np.round(s)) < 1e-6) and len(s.unique()) <= 20:
                        self.categorical_cols_.append(col)
                    else:
                        self.continuous_cols_.append(col)

        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        big_sentinels = {99, 999, 777, 7777, 9999}
        small_sentinels = {7, 9, 77}

        for col in self.continuous_cols_:
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(big_sentinels, np.nan)

        for col in self.categorical_cols_:
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(big_sentinels | small_sentinels, np.nan)

        for col in self.binary_cols_:
            if col in X_df.columns:
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
