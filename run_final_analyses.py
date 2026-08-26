"""
run_final_analyses.py — post-sign-off package for the locked R3 specification.

Locked spec: tuning_results_20260824_140539 (SMOTENC-ENN primary; KM sign-off
24 Aug 2026, WN decision 24 Aug to proceed as signed).

What this does, in order:
  1. THRESHOLD + CALIBRATION LOCK (validation only)
     Isotonic calibration fit on validation probabilities of the fitted
     primary pipeline; screening threshold = the point on the calibrated
     validation ROC achieving sensitivity >= 0.80 with maximum specificity.
     Both are frozen to disk BEFORE any test evaluation.
  2. FINAL TEST EVALUATION (single pass)
     The locked pipeline + calibrator + threshold applied to the held-out
     test set ONCE. Unweighted and survey-weighted AUC / sensitivity /
     specificity / PPV / NPV.
  3. PRE-DECLARED SENSITIVITY ANALYSES (each: refit on training data with
     the declared modification, threshold re-derived on validation by the
     SAME rule, then one test evaluation)
       A. no-resampling            (sampler step removed)
       B. utilization add-back     (utilization-class vars eligible again:
                                    HUQ050/HUQ030/HUQ071/HUQ090/PFQ041)
       C. age- and sex-matched     (1:1 nearest-age within sex, training set)
       D. age-dependent vars out   (BMXWT, SPXNFET, fev1_fvc_ratio,
                                    family_spirometry_interaction removed;
                                    bmi_z_cdc kept — it is age-referenced)
       E. missing-spirometry       (primary evaluated on the 718-type
                                    subgroup, plus refit excluding them)
  4. Everything written to outputs/final_analyses_<runid>/ as JSON + pkl,
     traceable to the locked run directory.

Run from the repo root (full run):
    python run_final_analyses.py
Smoke test (small models, no files written to outputs/):
    python run_final_analyses.py --smoke
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

# [2026-08-26] auto-detect the newest tuning run (the corrective run,
# once notebook 04 has been re-executed); falls back to the prior lock.
import glob as _glob
_runs = sorted(_glob.glob(os.path.join(HERE, "notebooks", "tuning_results_*")))
LOCKED_RUN = os.path.basename(_runs[-1]) if _runs else "tuning_results_20260824_140539"
RANDOM_STATE = 42
SENS_TARGET = 0.80

from sklearn.feature_selection import f_classif                     # noqa: E402
from sklearn.isotonic import IsotonicRegression                     # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve                # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline               # noqa: E402
from catboost import CatBoostClassifier                             # noqa: E402

from asthma_pipeline import (                                       # noqa: E402
    NHANESCleaner, ClinicalFeatureEngineer, ProtectedSelectKBest,
    AutoSMOTENCENN, preprocessing_steps, apply_correlation_pruning,
    LEAKY_PROXIES, AGE_RESTRICTED_VARS, IDENTIFIERS,
    PROTOCOL_ROUTING_VARS, UTILIZATION_PROXIES, PROTECTED_FEATURES,
)


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------

def binary_metrics(y, prob, threshold, w=None):
    y = np.asarray(y, float)
    yhat = (np.asarray(prob) >= threshold).astype(float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)

    def wsum(mask):
        return float(w[mask].sum())

    pos, neg = y == 1, y == 0
    tp, fn = wsum((yhat == 1) & pos), wsum((yhat == 0) & pos)
    tn, fp = wsum((yhat == 0) & neg), wsum((yhat == 1) & neg)
    out = {
        "auc": float(roc_auc_score(y, prob, sample_weight=w)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y)),
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def lock_threshold(y_val, prob_val_cal):
    """Highest-specificity validation point with sensitivity >= SENS_TARGET."""
    fpr, tpr, thr = roc_curve(y_val, prob_val_cal)
    idx = np.argmax(tpr >= SENS_TARGET)      # first (lowest-fpr) point reaching target
    return float(thr[idx]), float(tpr[idx]), float(1 - fpr[idx])


def evaluate(tag, pipe, calibrator, threshold, Xv, yv, swv, Xt, yt, swt):
    pv = calibrator.predict(pipe.predict_proba(Xv)[:, 1])
    pt = calibrator.predict(pipe.predict_proba(Xt)[:, 1])
    return {
        "analysis": tag,
        "validation": {"unweighted": binary_metrics(yv, pv, threshold),
                       "survey_weighted": binary_metrics(yv, pv, threshold, swv)},
        "test": {"unweighted": binary_metrics(yt, pt, threshold),
                 "survey_weighted": binary_metrics(yt, pt, threshold, swt)},
    }


def fit_calibrator(y_val, prob_val):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_val, y_val)
    return iso


def build_pipeline(feature_names, best_params, resample=True):
    steps = preprocessing_steps() + [
        ("feature_selection", ProtectedSelectKBest(
            f_classif, k=20, feature_names=feature_names,
            protect=[p for p in PROTECTED_FEATURES if p in feature_names])),
    ]
    if resample:
        steps.append(("smote_enn", AutoSMOTENCENN(random_state=RANDOM_STATE)))
    steps.append(("classifier", CatBoostClassifier(**best_params)))
    return ImbPipeline(steps)


def variant_run(tag, Xtr, ytr, Xv, yv, swv, Xt, yt, swt, fn, bp, resample=True):
    """Refit -> calibrate on val -> threshold on val -> single test pass."""
    pipe = build_pipeline(fn, bp, resample=resample)
    pipe.fit(Xtr, ytr)
    prob_val = pipe.predict_proba(Xv)[:, 1]
    cal = fit_calibrator(yv, prob_val)
    thr, s, sp = lock_threshold(yv, cal.predict(prob_val))
    res = evaluate(tag, pipe, cal, thr, Xv, yv, swv, Xt, yt, swt)
    res["selected_features"] = pipe.named_steps["feature_selection"].get_selected_names()
    res["threshold_locked_on_validation"] = {"threshold": round(thr, 4),
                                             "val_sens": round(s, 4),
                                             "val_spec": round(sp, 4)}
    return res, pipe, cal


# ---------------------------------------------------------------------------
# data preparation variants (replayed identically to the locked notebook)
# ---------------------------------------------------------------------------

def prepare(exclusions):
    """Replay notebook-04 prep from 03_cleaned with a given exclusion list."""
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(os.path.join(HERE, "data", "processed", "03_cleaned.parquet"))
    df = df.drop(columns=[c for c in ("NHANES_CYCLE",) if c in df.columns])
    y = df["MCQ010"].copy()
    sw = df["WTMEC2YR"]
    X = df[[c for c in df.columns
            if c not in ["MCQ010", "WTMEC2YR", "WTINT2YR", "SDMVPSU", "SDMVSTRA",
                         "SEQN", "NHANES_CYCLE"]]].copy()
    y = (y == 1).astype(float)
    y[df["MCQ010"].isna()] = np.nan
    m = y.notna() & sw.notna() & (sw > 0)
    X, y, sw = X[m].reset_index(drop=True), y[m].reset_index(drop=True), sw[m].reset_index(drop=True)
    ped = (X["RIDAGEYR"] >= 6) & (X["RIDAGEYR"] < 18)
    X, y, sw = X[ped].reset_index(drop=True), y[ped].reset_index(drop=True), sw[ped].reset_index(drop=True)
    X = X.drop(columns=[c for c in X.columns if c in exclusions])

    X_tmp, X_te, y_tmp, y_te, sw_tmp, sw_te = train_test_split(
        X, y, sw, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_tr, X_va, y_tr, y_va, sw_tr, sw_va = train_test_split(
        X_tmp, y_tmp, sw_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp)

    cleaner, fe = NHANESCleaner(), ClinicalFeatureEngineer()
    Xtr_f = fe.fit_transform(cleaner.fit_transform(X_tr))
    Xva_f = fe.transform(cleaner.transform(X_va))
    Xte_f = fe.transform(cleaner.transform(X_te))
    Xtr_f, Xva_f, Xte_f, pruned = apply_correlation_pruning(Xtr_f, Xva_f, Xte_f)
    return (Xtr_f, y_tr.reset_index(drop=True), sw_tr.reset_index(drop=True),
            Xva_f, y_va.reset_index(drop=True), sw_va.reset_index(drop=True),
            Xte_f, y_te.reset_index(drop=True), sw_te.reset_index(drop=True),
            Xtr_f.columns.tolist(), pruned)


def match_age_sex(Xtr, ytr, seed=RANDOM_STATE):
    """1:1 nearest-age matching within sex, controls to cases, no replacement.

    Uses RIAGENDR (binary-recoded by the cleaner) and RIDAGEYR from the
    engineered training frame. Returns positional indices to keep.
    """
    rng = np.random.default_rng(seed)
    age = Xtr["RIDAGEYR"].to_numpy()
    sex = Xtr["RIAGENDR"].to_numpy()
    y = np.asarray(ytr)
    keep = []
    for s in np.unique(sex[~np.isnan(sex)]):
        cases = np.where((y == 1) & (sex == s))[0]
        ctrls = list(np.where((y == 0) & (sex == s))[0])
        rng.shuffle(cases)
        for c in cases:
            if not ctrls:
                break
            d = np.abs(age[ctrls] - age[c])
            j = int(np.argmin(d))
            keep += [c, ctrls.pop(j)]
    return sorted(keep)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny models, results to /tmp, no outputs/ writes")
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)
    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    study = joblib.load(os.path.join(run_dir, "catboost_study.pkl"))
    bp = study.best_params.copy()
    bp.update({"random_state": RANDOM_STATE, "verbose": False, "thread_count": -1})
    if args.smoke:
        bp.update({"iterations": 40, "depth": 4})

    Xtr, ytr, swtr = art["X_train_feat"], art["y_train"], art["sw_train"]
    Xva, yva, swva = art["X_val_feat"], art["y_val"], art["sw_val"]
    Xte, yte, swte = art["X_test_feat"], art["y_test"], art["sw_test"]
    FN = art["feature_names"]

    out_dir = ("/tmp/final_analyses_smoke" if args.smoke else
               os.path.join(HERE, "outputs", f"final_analyses_{LOCKED_RUN.split('_', 2)[2]}"))
    os.makedirs(out_dir, exist_ok=True)
    results = {"locked_run": LOCKED_RUN, "generated": datetime.now().isoformat(timespec="seconds"),
               "sensitivity_target": SENS_TARGET, "smoke": bool(args.smoke), "analyses": {}}

    # ---- 1+2: PRIMARY — use the run's fitted pipeline (exact provenance) ----
    print("PRIMARY (locked SMOTENC pipeline from the run artifact)")
    primary = joblib.load(os.path.join(run_dir, "catboost_best_model.pkl"))
    prob_val = primary.predict_proba(Xva)[:, 1]
    cal = fit_calibrator(yva, prob_val)
    thr, s, sp = lock_threshold(yva, cal.predict(prob_val))
    joblib.dump({"calibrator": cal, "threshold": thr,
                 "rule": f"validation sens>={SENS_TARGET}, max specificity",
                 "locked_at": datetime.now().isoformat(timespec="seconds"),
                 "locked_before_test_evaluation": True},
                os.path.join(out_dir, "locked_threshold_calibration.pkl"))
    print(f"  threshold locked on validation: {thr:.4f} (val sens {s:.3f}, spec {sp:.3f})")

    res = evaluate("primary_smotenc", primary, cal, thr, Xva, yva, swva, Xte, yte, swte)
    res["threshold_locked_on_validation"] = {"threshold": round(thr, 4),
                                             "val_sens": round(s, 4), "val_spec": round(sp, 4)}
    results["analyses"]["primary"] = res
    t = res["test"]["unweighted"]
    print(f"  TEST (single pass): AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  "
          f"spec {t['specificity']:.3f}  ppv {t['ppv']:.3f}  npv {t['npv']:.3f}")

    # ---- 3A: no-resampling ------------------------------------------------
    print("A. no-resampling")
    res, _, _ = variant_run("no_resampling", Xtr, ytr, Xva, yva, swva, Xte, yte, swte,
                            FN, bp, resample=False)
    results["analyses"]["no_resampling"] = res
    t = res["test"]["unweighted"]
    print(f"  TEST: AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}")

    # ---- 3B: utilization add-back ----------------------------------------
    print("B. utilization add-back (eligible again: " + "/".join(UTILIZATION_PROXIES) + ")")
    excl = (LEAKY_PROXIES + AGE_RESTRICTED_VARS + IDENTIFIERS + PROTOCOL_ROUTING_VARS)
    prep = prepare(excl)                     # note: UTILIZATION_PROXIES not excluded
    bXtr, bytr, _, bXva, byva, bswva, bXte, byte_, bswte, bFN, _ = prep
    res, _, _ = variant_run("utilization_addback", bXtr, bytr, bXva, byva, bswva,
                            bXte, byte_, bswte, bFN, bp, resample=True)
    added = [f for f in res["selected_features"] if f in UTILIZATION_PROXIES]
    res["utilization_variables_selected"] = added
    results["analyses"]["utilization_addback"] = res
    t = res["test"]["unweighted"]
    print(f"  TEST: AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}"
          f"  | utilization vars selected: {added}")

    # ---- 3C: age- and sex-matched training -------------------------------
    print("C. age- and sex-matched (1:1 nearest-age within sex, training set)")
    keep = match_age_sex(Xtr, ytr)
    res, _, _ = variant_run("age_sex_matched",
                            Xtr.iloc[keep], ytr.iloc[keep],
                            Xva, yva, swva, Xte, yte, swte, FN, bp, resample=True)
    res["matched_training_n"] = len(keep)
    results["analyses"]["age_sex_matched"] = res
    t = res["test"]["unweighted"]
    print(f"  matched training n={len(keep)} | TEST: AUC {t['auc']:.3f}  "
          f"sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}")

    # ---- 3D: age-dependent variables removed ------------------------------
    print("D. age-dependent anthropometric/spirometric variables removed")
    drop = ["BMXWT", "SPXNFET", "fev1_fvc_ratio", "family_spirometry_interaction"]
    dXtr = Xtr.drop(columns=[c for c in drop if c in Xtr.columns])
    dXva = Xva.drop(columns=[c for c in drop if c in Xva.columns])
    dXte = Xte.drop(columns=[c for c in drop if c in Xte.columns])
    dFN = dXtr.columns.tolist()
    res, _, _ = variant_run("age_dependent_removed", dXtr, ytr, dXva, yva, swva,
                            dXte, yte, swte, dFN, bp, resample=True)
    res["removed"] = drop
    results["analyses"]["age_dependent_removed"] = res
    t = res["test"]["unweighted"]
    print(f"  TEST: AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}")

    # ---- 3E: missing-spirometry children ----------------------------------
    print("E. children missing baseline spirometry (SPXNFEV1 & SPXNFVC)")
    miss_tr = (Xtr["SPXNFEV1_missing"] == 1) & (Xtr["SPXNFVC_missing"] == 1)
    miss_va = (Xva["SPXNFEV1_missing"] == 1) & (Xva["SPXNFVC_missing"] == 1)
    miss_te = (Xte["SPXNFEV1_missing"] == 1) & (Xte["SPXNFVC_missing"] == 1)

    pv = cal.predict(primary.predict_proba(Xva[miss_va])[:, 1])
    pt = cal.predict(primary.predict_proba(Xte[miss_te])[:, 1])
    sub = {
        "analysis": "missing_spirometry_subgroup(primary model)",
        "n_train/val/test": [int(miss_tr.sum()), int(miss_va.sum()), int(miss_te.sum())],
        "validation": binary_metrics(yva[miss_va], pv, thr, np.asarray(swva)[np.asarray(miss_va)]),
        "test": binary_metrics(yte[miss_te], pt, thr, np.asarray(swte)[np.asarray(miss_te)]),
    }
    results["analyses"]["missing_spirometry_subgroup"] = sub
    print(f"  subgroup n (train/val/test): {sub['n_train/val/test']} | "
          f"TEST AUC {sub['test']['auc']:.3f}")

    res, _, _ = variant_run("missing_spirometry_excluded",
                            Xtr[~miss_tr], ytr[~miss_tr.values],
                            Xva[~miss_va], yva[~miss_va.values], np.asarray(swva)[~miss_va.values],
                            Xte[~miss_te], yte[~miss_te.values], np.asarray(swte)[~miss_te.values],
                            FN, bp, resample=True)
    results["analyses"]["missing_spirometry_excluded"] = res
    t = res["test"]["unweighted"]
    print(f"  excluded-refit TEST: AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  "
          f"spec {t['specificity']:.3f}")

    # ---- write -------------------------------------------------------------
    with open(os.path.join(out_dir, "final_analyses_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results written to {out_dir}/")
    print("Reminder: this script performed the ONE test-set evaluation pass for the "
          "locked specification and pre-declared analyses. Do not re-run against the "
          "test set after making further modeling decisions.")


if __name__ == "__main__":
    main()
