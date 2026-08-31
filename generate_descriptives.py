"""
generate_descriptives.py — committed generator for descriptive_statistics.json.

[2026-08-27 KM ruling, reporting item] The small descriptive analyses quoted
in the response letter previously had no committed generating code. This
script produces every number in descriptive_statistics.json from the locked
artifacts, so each has an artifact behind it. Descriptive only: fixed test
predictions are stratified; no modeling or threshold decisions are made.

Contents:
  - age SMD, cases vs controls, unweighted AND survey-weighted (WTMEC2YR)
  - test-set AUC by fixed age group (6-9 / 10-13 / 14-17; these are age
    BANDS, not empirical tertiles), raw scores, with stratified-bootstrap
    95% CIs (descriptive; the predictions are fixed)
  - test-set AUC by spirometry availability in three groups (both usable /
    one usable / neither usable), raw scores, unweighted and weighted
  - CDC BMI-for-age coverage, and weighted obesity prevalence (>=95th
    percentile) by asthma status, nonmissing denominators; ages enter the
    CDC reference as completed months + 0.5 per the CDC instruction
    [28 Aug 2026 correction]

Run from the repo root, AFTER run_final_analyses.py:
    python generate_descriptives.py

Writes: outputs/final_analyses_<runid>/descriptive_statistics.json
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

# [2026-08-31] pinned to the analysis of record
PINNED_RUN = "tuning_results_20260831_103201"
LOCKED_RUN = PINNED_RUN

from sklearn.metrics import roc_auc_score                            # noqa: E402
from pediatric_corrections import cdc_bmi_z                          # noqa: E402


def wmean(x, w):
    m = np.isfinite(x) & np.isfinite(w)
    return float(np.average(x[m], weights=w[m]))


def wvar(x, w):
    m = np.isfinite(x) & np.isfinite(w)
    mu = np.average(x[m], weights=w[m])
    return float(np.average((x[m] - mu) ** 2, weights=w[m]))


def smd(x, g, w=None):
    """Standardized mean difference, cases (g==1) vs controls (g==0)."""
    if w is None:
        w = np.ones_like(x)
    x1, w1 = x[g == 1], w[g == 1]
    x0, w0 = x[g == 0], w[g == 0]
    return (wmean(x1, w1) - wmean(x0, w0)) / np.sqrt(
        (wvar(x1, w1) + wvar(x0, w0)) / 2)


def main():
    run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)
    fin_dir = os.path.join(HERE, "outputs",
                           f"final_analyses_{LOCKED_RUN.split('_', 2)[2]}")

    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    primary = joblib.load(os.path.join(run_dir, "catboost_best_model.pkl"))

    Xte = art["X_test_feat"]
    yte = np.asarray(art["y_test"], float)
    raw_te = primary.predict_proba(Xte)[:, 1]      # raw scores (2026-08-27 rule)

    out = {"run": LOCKED_RUN,
           "generated_by": "generate_descriptives.py",
           "basis": ("descriptive stratification of the locked test "
                     "predictions (raw scores); no modeling decisions")}

    # ---- cohort-level: age SMD, obesity, bmi_z coverage --------------------
    df = pd.read_parquet(os.path.join(HERE, "data", "processed",
                                      "03_cleaned.parquet"))
    an = df[df["WTMEC2YR"] > 0].copy()
    g = (an["MCQ010"] == 1).astype(int).to_numpy()
    w = an["WTMEC2YR"].to_numpy(float)
    age = an["RIDAGEYR"].to_numpy(float)
    out["age_smd_cases_vs_controls"] = round(smd(age, g), 3)
    out["age_smd_cases_vs_controls_weighted"] = round(smd(age, g, w), 3)

    z, pct = cdc_bmi_z(an["BMXBMI"], an["RIDAGEEX_H"], an["RIAGENDR"])
    pct = np.asarray(pct, float)
    known = np.isfinite(pct)
    out["bmi_z_computable"] = {"n": int(known.sum()), "of": int(len(an)),
                               "pct": round(100 * known.mean(), 1)}
    obese = known & (pct >= 95)
    ob = {}
    for name, sel in (("asthma", g == 1), ("no_asthma", g == 0)):
        denom = w[sel & known].sum()
        ob[name] = round(100 * w[sel & obese].sum() / denom, 1)
    out["weighted_obesity_pct_ge95th_nonmissing_denominator"] = ob

    # ---- test-set stratifications (fixed predictions) ----------------------
    swte = np.asarray(art["sw_test"], float)
    rng = np.random.default_rng(42)

    def auc_ci(m, n_boot=2000):
        """Stratified-bootstrap 95% CI of the raw-score AUC in subgroup m."""
        y, r = yte[m], raw_te[m]
        ip, iq = np.where(y == 1)[0], np.where(y == 0)[0]
        stats = []
        for _ in range(n_boot):
            bi = np.concatenate([rng.choice(ip, len(ip), replace=True),
                                 rng.choice(iq, len(iq), replace=True)])
            stats.append(roc_auc_score(y[bi], r[bi]))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return [round(float(lo), 3), round(float(hi), 3)]

    age_te = Xte["RIDAGEYR"].to_numpy(float)
    groups = {}
    for label, lo, hi in (("6-9", 6, 9), ("10-13", 10, 13), ("14-17", 14, 17)):
        m = (age_te >= lo) & (age_te <= hi)
        groups[label] = {"n": int(m.sum()),
                         "auc": round(float(roc_auc_score(yte[m], raw_te[m])), 3),
                         "auc_ci95": auc_ci(m)}
    out["age_group_test_auc"] = {
        "note": ("fixed age BANDS (not empirical tertiles); exploratory "
                 "description of the fixed test predictions; raw-score AUC "
                 "with stratified-bootstrap CIs"),
        **groups}

    f_ok = (Xte["SPXNFEV1_missing"] == 0).to_numpy()
    v_ok = (Xte["SPXNFVC_missing"] == 0).to_numpy()
    avail = {"both_usable": f_ok & v_ok,
             "one_usable": f_ok ^ v_ok,
             "neither_usable": ~f_ok & ~v_ok}
    def safe_auc(m, w=None):
        """AUC or None when the subgroup is empty or single-class."""
        if m.sum() == 0 or len(np.unique(yte[m])) < 2:
            return None
        return round(float(roc_auc_score(
            yte[m], raw_te[m], sample_weight=None if w is None else w[m])), 3)

    block = {"note": ("usable = measured AND quality grade A/B (2026-08-27 "
                      "gating); AUC from raw scores, unweighted and "
                      "survey-weighted (WTMEC2YR); null = subgroup empty or "
                      "single-class")}
    for name, m in avail.items():
        block[name] = {"n": int(m.sum()),
                       "auc_unweighted": safe_auc(m),
                       "auc_weighted": safe_auc(m, swte)}
    out["spirometry_availability_test_auc"] = block

    os.makedirs(fin_dir, exist_ok=True)
    path = os.path.join(fin_dir, "descriptive_statistics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
