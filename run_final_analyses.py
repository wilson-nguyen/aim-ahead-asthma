"""
run_final_analyses.py — post-sign-off package for the locked R3 specification.

Locked spec: SMOTENC-ENN primary (KM sign-off 24 Aug 2026), amended by the
27 Aug 2026 KM rulings (spirometry quality gating A/B, URDNALLC excluded,
raw-score AUC reporting); runs against the newest tuning_results_* produced
by the re-executed notebook 04.

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
       E. missing-spirometry       (primary evaluated on the no-usable-
                                    spirometry subgroup, plus refit
                                    excluding them)
       F. quality grades A/B/C     ([2026-08-27 KM ruling] primary gates
                                    spirometry to grades A/B; this arm
                                    widens to A/B/C to test whether
                                    admitting grade C changes the results)
  4. Everything written to outputs/final_analyses_<runid>/ as JSON + pkl,
     traceable to the locked run directory.

Reporting rule [2026-08-27 KM ruling]: "auc" is computed from RAW model
scores (calibration-independent discrimination); sensitivity/specificity/
PPV/NPV are at the locked threshold on isotonic-calibrated scores, whose
own quality is summarized separately in "calibration_test" (Brier score,
logistic recalibration intercept/slope). "auc_calibrated_scores" is kept
for comparison with previously circulated numbers.

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

# [2026-08-31] the analysis of record is PINNED. A newer scratch run can no
# longer silently become the evaluation target; override only with --run.
PINNED_RUN = "tuning_results_20260831_103201"
LOCKED_RUN = PINNED_RUN
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
    MEASUREMENT_VALIDITY_EXCLUSIONS, apply_spirometry_quality_gating,
    PRIMARY_ALLOWED_GRADES, SENSITIVITY_ALLOWED_GRADES,
)

# Full primary-specification exclusion list (arm B removes the utilization
# class from this; the measurement-validity exclusions apply in EVERY arm).
FULL_EXCLUSIONS = (LEAKY_PROXIES + AGE_RESTRICTED_VARS + IDENTIFIERS
                   + PROTOCOL_ROUTING_VARS + UTILIZATION_PROXIES
                   + MEASUREMENT_VALIDITY_EXCLUSIONS)


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------

def binary_metrics(y, prob, threshold, w=None, raw_scores=None):
    """Threshold metrics on `prob` (calibrated scores). [2026-08-27 ruling]
    If `raw_scores` is given, "auc" is computed from the RAW model scores
    (calibration-independent discrimination) and the calibrated-score AUC
    is kept as "auc_calibrated_scores"; otherwise "auc" is from `prob`."""
    y = np.asarray(y, float)
    yhat = (np.asarray(prob) >= threshold).astype(float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)

    def wsum(mask):
        return float(w[mask].sum())

    pos, neg = y == 1, y == 0
    tp, fn = wsum((yhat == 1) & pos), wsum((yhat == 0) & pos)
    tn, fp = wsum((yhat == 0) & neg), wsum((yhat == 1) & neg)
    out = {
        "auc": float(roc_auc_score(
            y, prob if raw_scores is None else raw_scores, sample_weight=w)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y)),
    }
    if raw_scores is not None:
        out["auc_calibrated_scores"] = float(
            roc_auc_score(y, prob, sample_weight=w))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def calibration_summary(y, prob_cal):
    """Separate calibration assessment [2026-08-27 ruling]: Brier score and
    logistic-recalibration intercept/slope of y on logit(calibrated p)."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, float)
    p = np.clip(np.asarray(prob_cal, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y)
    return {"brier": round(float(np.mean((p - y) ** 2)), 4),
            "intercept": round(float(lr.intercept_[0]), 4),
            "slope": round(float(lr.coef_[0][0]), 4)}


def lock_threshold(y_val, prob_val_cal):
    """Highest-specificity validation point with sensitivity >= SENS_TARGET."""
    fpr, tpr, thr = roc_curve(y_val, prob_val_cal)
    idx = np.argmax(tpr >= SENS_TARGET)      # first (lowest-fpr) point reaching target
    return float(thr[idx]), float(tpr[idx]), float(1 - fpr[idx])


def evaluate(tag, pipe, calibrator, threshold, Xv, yv, swv, Xt, yt, swt):
    rv = pipe.predict_proba(Xv)[:, 1]           # raw scores
    rt = pipe.predict_proba(Xt)[:, 1]
    pv = calibrator.predict(rv)                 # calibrated scores
    pt = calibrator.predict(rt)
    return {
        "analysis": tag,
        "validation": {"unweighted": binary_metrics(yv, pv, threshold, raw_scores=rv),
                       "survey_weighted": binary_metrics(yv, pv, threshold, swv, raw_scores=rv)},
        "test": {"unweighted": binary_metrics(yt, pt, threshold, raw_scores=rt),
                 "survey_weighted": binary_metrics(yt, pt, threshold, swt, raw_scores=rt)},
        "calibration_test": calibration_summary(yt, pt),
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

def prepare(exclusions, allowed_grades=PRIMARY_ALLOWED_GRADES):
    """Replay notebook-04 prep from 03_cleaned with a given exclusion list.

    [2026-08-27 KM ruling] Applies spirometry quality gating right after
    load, exactly as notebook 04 cell 2 does; `allowed_grades` widens to
    A/B/C for sensitivity arm F."""
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(os.path.join(HERE, "data", "processed", "03_cleaned.parquet"))
    df = df.drop(columns=[c for c in ("NHANES_CYCLE",) if c in df.columns])
    df = apply_spirometry_quality_gating(df, allowed_grades=allowed_grades,
                                         verbose=False)
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
    global LOCKED_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny models, results to /tmp, no outputs/ writes")
    ap.add_argument("--run", default=None,
                    help=f"tuning run to evaluate (default: pinned {PINNED_RUN})")
    args = ap.parse_args()
    if args.run:
        LOCKED_RUN = (args.run if args.run.startswith("tuning_results_")
                      else f"tuning_results_{args.run}")

    run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)

    # [31 Aug] fail-closed gate: split verification must have ACCEPTED this
    # exact run before any test-set evaluation happens here.
    rep_path = os.path.join(HERE, "outputs", "split_verification_report.json")
    if not os.path.exists(rep_path):
        sys.exit("GATE: outputs/split_verification_report.json is missing. "
                 "Run verify_split_reconstruction.py first.")
    rep = json.load(open(rep_path))
    if (rep.get("run_dir") != LOCKED_RUN
            or not str(rep.get("verdict", "")).startswith("ACCEPTED")):
        sys.exit(f"GATE: split verification covers '{rep.get('run_dir')}' "
                 f"(verdict: {str(rep.get('verdict', ''))[:30]}...). Re-run "
                 f"verify_split_reconstruction.py against {LOCKED_RUN}; do not "
                 f"evaluate unless it is ACCEPTED for this run.")
    print(f"gate: split verification ACCEPTED for {LOCKED_RUN}")
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
    excl = [c for c in FULL_EXCLUSIONS if c not in UTILIZATION_PROXIES]
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

    # ---- 3E: children with no usable baseline spirometry -------------------
    # [28 Aug estimand fix] AUC from raw scores, unweighted and weighted
    # reported side by side with labels; the subgroup is explicitly defined.
    print("E. children with no usable baseline spirometry (neither FEV1 nor FVC)")
    miss_tr = (Xtr["SPXNFEV1_missing"] == 1) & (Xtr["SPXNFVC_missing"] == 1)
    miss_va = (Xva["SPXNFEV1_missing"] == 1) & (Xva["SPXNFVC_missing"] == 1)
    miss_te = (Xte["SPXNFEV1_missing"] == 1) & (Xte["SPXNFVC_missing"] == 1)

    rv_sub = primary.predict_proba(Xva[miss_va])[:, 1]
    rt_sub = primary.predict_proba(Xte[miss_te])[:, 1]
    pv = cal.predict(rv_sub)
    pt = cal.predict(rt_sub)
    swva_s = np.asarray(swva)[np.asarray(miss_va)]
    swte_s = np.asarray(swte)[np.asarray(miss_te)]
    sub = {
        "analysis": "no_usable_spirometry_subgroup(primary model)",
        "definition": ("neither FEV1 nor FVC usable (both availability "
                       "indicators = 1; quality-gated measurements count as "
                       "unusable). AUC from raw scores; threshold metrics on "
                       "calibrated scores."),
        "n_train/val/test": [int(miss_tr.sum()), int(miss_va.sum()), int(miss_te.sum())],
        "validation": {
            "unweighted": binary_metrics(yva[miss_va], pv, thr, raw_scores=rv_sub),
            "survey_weighted": binary_metrics(yva[miss_va], pv, thr, swva_s,
                                              raw_scores=rv_sub)},
        "test": {
            "unweighted": binary_metrics(yte[miss_te], pt, thr, raw_scores=rt_sub),
            "survey_weighted": binary_metrics(yte[miss_te], pt, thr, swte_s,
                                              raw_scores=rt_sub)},
    }
    results["analyses"]["no_usable_spirometry_subgroup"] = sub
    print(f"  subgroup n (train/val/test): {sub['n_train/val/test']} | "
          f"TEST AUC {sub['test']['unweighted']['auc']:.3f} unweighted, "
          f"{sub['test']['survey_weighted']['auc']:.3f} weighted")

    res, _, _ = variant_run("no_usable_spirometry_excluded",
                            Xtr[~miss_tr], ytr[~miss_tr.values],
                            Xva[~miss_va], yva[~miss_va.values], np.asarray(swva)[~miss_va.values],
                            Xte[~miss_te], yte[~miss_te.values], np.asarray(swte)[~miss_te.values],
                            FN, bp, resample=True)
    res["definition"] = "refit excluding children with neither FEV1 nor FVC usable"
    results["analyses"]["no_usable_spirometry_excluded"] = res
    t = res["test"]["unweighted"]
    print(f"  excluded-refit TEST: AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  "
          f"spec {t['specificity']:.3f}")

    # ---- 3F: quality grades A/B/C usable ([2026-08-27 KM ruling]) ----------
    print("F. spirometry quality grades A/B/C usable (primary gates to A/B)")
    prep = prepare(FULL_EXCLUSIONS, allowed_grades=SENSITIVITY_ALLOWED_GRADES)
    fXtr, fytr, _, fXva, fyva, fswva, fXte, fyte, fswte, fFN, _ = prep
    res, pipeF, _ = variant_run("quality_grades_ABC", fXtr, fytr, fXva, fyva, fswva,
                                fXte, fyte, fswte, fFN, bp, resample=True)
    res["allowed_grades"] = list(SENSITIVITY_ALLOWED_GRADES)
    res["note"] = ("primary analysis gates best-test FEV1/FVC to quality "
                   "grades A/B; this pre-declared arm admits grade C")

    # [28 Aug] paired uncertainty for the A/B vs A/B/C contrast: same test
    # participants (asserted), raw scores, stratified paired bootstrap.
    assert np.array_equal(np.asarray(yte, float), np.asarray(fyte, float)), \
        "arm F test outcomes differ from primary - split drift, do not compare"
    rt_primary = primary.predict_proba(Xte)[:, 1]
    rt_abc = pipeF.predict_proba(fXte)[:, 1]
    yarr = np.asarray(yte, float)
    ip, iq = np.where(yarr == 1)[0], np.where(yarr == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    n_boot = 200 if args.smoke else 2000
    diffs = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(ip, len(ip), replace=True),
                             rng.choice(iq, len(iq), replace=True)])
        diffs.append(roc_auc_score(yarr[bi], rt_primary[bi])
                     - roc_auc_score(yarr[bi], rt_abc[bi]))
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    res["paired_auc_difference_AB_minus_ABC"] = {
        "point": round(float(roc_auc_score(yarr, rt_primary)
                             - roc_auc_score(yarr, rt_abc)), 4),
        "ci95": [round(float(dlo), 4), round(float(dhi), 4)],
        "n_boot": n_boot,
        "note": ("paired stratified bootstrap of raw test scores, primary "
                 "(A/B gating) minus arm F (A/B/C); same participants, "
                 "different feature versions; CI covering 0 = no detectable "
                 "dependence on the strictness of the quality criterion"),
    }
    p = res["paired_auc_difference_AB_minus_ABC"]
    print(f"  paired AUC diff (A/B minus A/B/C): {p['point']} "
          f"({p['ci95'][0]} to {p['ci95'][1]})")
    results["analyses"]["quality_grades_ABC"] = res
    t = res["test"]["unweighted"]
    print(f"  TEST: AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}")

    # ---- write -------------------------------------------------------------
    with open(os.path.join(out_dir, "final_analyses_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results written to {out_dir}/")
    print("Reminder: this script performed the ONE test-set evaluation pass for the "
          "locked specification and pre-declared analyses. Do not re-run against the "
          "test set after making further modeling decisions.")


if __name__ == "__main__":
    main()
