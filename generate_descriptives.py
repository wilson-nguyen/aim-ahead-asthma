"""
generate_descriptives.py — committed generator for descriptive_statistics.json.

[2026-08-27 KM ruling, reporting item] The small descriptive analyses quoted
in the response letter previously had no committed generating code. This
script produces every number in descriptive_statistics.json from the locked
artifacts, so each has an artifact behind it. Descriptive only: fixed test
predictions are stratified; no modeling or threshold decisions are made.

Contents:
  - age SMD, cases vs controls, unweighted AND survey-weighted (WTMEC2YR)
  - test-set AUC by age tertile (6-9 / 10-13 / 14-17), raw scores
  - test-set AUC by spirometry availability (both indicators), raw scores
  - CDC BMI-for-age coverage, and weighted obesity prevalence (>=95th
    percentile) by asthma status, nonmissing denominators

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

import glob as _glob
_runs = sorted(_glob.glob(os.path.join(HERE, "notebooks", "tuning_results_*")))
LOCKED_RUN = os.path.basename(_runs[-1]) if _runs else "tuning_results_20260824_140539"

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
    age_te = Xte["RIDAGEYR"].to_numpy(float)
    tert = {}
    for label, lo, hi in (("6-9", 6, 9), ("10-13", 10, 13), ("14-17", 14, 17)):
        m = (age_te >= lo) & (age_te <= hi)
        tert[label] = {"n": int(m.sum()),
                       "auc": round(float(roc_auc_score(yte[m], raw_te[m])), 3)}
    out["age_tertile_test_auc"] = tert

    miss = ((Xte["SPXNFEV1_missing"] == 1)
            & (Xte["SPXNFVC_missing"] == 1)).to_numpy()
    out["spirometry_availability_test_auc"] = {
        "note": ("'missing' = no usable baseline spirometry; from the "
                 "2026-08-27 quality gating onward this includes "
                 "measurements gated out for quality grade C/D/F"),
        "complete": {"n": int((~miss).sum()),
                     "auc": round(float(roc_auc_score(yte[~miss], raw_te[~miss])), 3)},
        "missing": {"n": int(miss.sum()),
                    "auc": round(float(roc_auc_score(yte[miss], raw_te[miss])), 3)},
    }

    os.makedirs(fin_dir, exist_ok=True)
    path = os.path.join(fin_dir, "descriptive_statistics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
