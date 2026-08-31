"""
run_reduced_model_and_figures.py — R3 successor to notebook 05.

From the locked primary (tuning_results_20260824_140539 + the locked
threshold/calibration from run_final_analyses.py):

  1. SHAP values on the TRAINING data via CatBoost's native TreeSHAP,
     computed on the primary pipeline's selected 22 features.
  2. Top-10 feature ranking by mean |SHAP| (training data only — no
     validation or test involvement in the ranking).
  3. Reduced model: same preprocessing + SMOTENC + tuned hyperparameters,
     restricted to the top-10 features; calibrated and thresholded on
     validation by the locked rule (sens >= 0.80), then ONE test pass.
     Feature names are saved with the model (fixes the nameless
     model_b_top10 problem).
  4. Figures for the revised manuscript, written to outputs/figures_R3/:
       figure_roc.png            ROC, full vs reduced (test)
       figure_metrics.png        sens/spec/PPV/NPV at locked thresholds
       figure_shap_ranking.png   mean |SHAP| bar ranking (all 22)
       figure_shap_summary.png   beeswarm if `shap` is installed, else skipped
       efigure_calibration.png   reliability curves (validation and test)
       efigure_decision_curve.png net benefit vs threshold probability (test)
  5. reduced_model_bundle.pkl + reduced_model_results.json + shap artifacts.

Run from the repo root:
    python run_reduced_model_and_figures.py
Smoke test (small refit, figures + files to /tmp):
    python run_reduced_model_and_figures.py --smoke
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

# [2026-08-26] auto-detect the newest tuning run (the corrective run,
# once notebook 04 has been re-executed); falls back to the prior lock.
# [2026-08-31] pinned to the analysis of record
PINNED_RUN = "tuning_results_20260831_103201"
LOCKED_RUN = PINNED_RUN
RANDOM_STATE = 42
SENS_TARGET = 0.80

from catboost import CatBoostClassifier, Pool                        # noqa: E402
from sklearn.isotonic import IsotonicRegression                      # noqa: E402
from sklearn.metrics import roc_curve, roc_auc_score                 # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline                # noqa: E402

from asthma_pipeline import AutoSMOTENCENN, preprocessing_steps      # noqa: E402
from run_final_analyses import (                                     # noqa: E402
    binary_metrics, fit_calibrator, lock_threshold, calibration_summary,
)
# [2026-08-27 KM ruling] AUC is reported from RAW model scores; threshold
# metrics use the calibrated scores; calibration is assessed separately.

# Display labels for figures (verified against project label sources).
LABELS = {
    "RDQ070": "Wheezing (past yr)", "MCQ300B": "Family history of asthma",
    "HUQ010": "General health condition", "HUQ020": "Health vs 1 yr ago",
    "RDQ140": "Dry cough at night (past yr)", "AGQ030": "Hay fever (past yr)",
    "PFQ020": "Activity limitation", "PFQ041": "Special Ed/Early Intervention services",
    "fev1_fvc_ratio": "FEV1/FVC ratio", "SPXNFET": "Forced expiratory time",
    "family_spirometry_interaction": "Family history × lung function",
    "bmi_z_cdc": "CDC BMI-for-age z-score", "BMXWT": "Weight",
    "DMDHHSIZ": "Household size",
    "cotinine_log": "Serum cotinine (log)", "URDNALLC": "Urinary NNAL below detection limit (indicator)",
    "RIDRETH1": "Race/Hispanic origin",
    "DMDCITZN": "Citizenship", "FIALANG": "Family interview language",
    "DMDBORN_US": "Born in US", "DMDHRBR_US": "HH reference person born in US",
    "HIQ011": "Health insurance coverage",
    "SPXNFEV1_missing": "No usable FEV1", "SPXNFVC_missing": "No usable FVC",
}


def lab(f):
    return LABELS.get(f, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)
    fin_dir = os.path.join(HERE, "outputs", f"final_analyses_{LOCKED_RUN.split('_', 2)[2]}")
    out_dir = "/tmp/reduced_smoke" if args.smoke else os.path.join(
        HERE, "outputs", f"reduced_model_{LOCKED_RUN.split('_', 2)[2]}")
    fig_dir = "/tmp/figures_R3_smoke" if args.smoke else os.path.join(HERE, "outputs", "figures_R3")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    primary = joblib.load(os.path.join(run_dir, "catboost_best_model.pkl"))
    lock = joblib.load(os.path.join(fin_dir, "locked_threshold_calibration.pkl"))
    cal_full, thr_full = lock["calibrator"], lock["threshold"]

    study = joblib.load(os.path.join(run_dir, "catboost_study.pkl"))
    bp = study.best_params.copy()
    bp.update({"random_state": RANDOM_STATE, "verbose": False, "thread_count": -1})
    if args.smoke:
        bp.update({"iterations": 40, "depth": 4})

    Xtr, ytr = art["X_train_feat"], art["y_train"]
    Xva, yva, swva = art["X_val_feat"], art["y_val"], art["sw_val"]
    Xte, yte, swte = art["X_test_feat"], art["y_test"], art["sw_test"]

    # ---- 1. SHAP on training, primary's selected features ------------------
    sel = primary.named_steps["feature_selection"]
    names = sel.get_selected_names()
    Xtr_sel = Xtr                                   # imputer→scaler→selector,
    for _, step in primary.steps[:-1]:              # skipping sampler steps —
        if hasattr(step, "transform"):              # mirrors predict-time flow
            Xtr_sel = step.transform(Xtr_sel)
    clf = primary.named_steps["classifier"]
    shap_vals = clf.get_feature_importance(
        data=Pool(Xtr_sel, label=np.asarray(ytr)), type="ShapValues")[:, :-1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranking = [(names[i], float(mean_abs[i])) for i in order]
    top10 = [names[i] for i in order[:10]]
    # [Spec compliance, 25 Aug] Section F of the signed exclusion log protects
    # the spirometry-availability indicators in EVERY model, so the reduced
    # model is top-10 + the two indicators (12 features), not bare top-10.
    # Otherwise missing spirometry would be median-imputed invisibly again.
    reduced_features = top10 + [p for p in ("SPXNFEV1_missing", "SPXNFVC_missing")
                                if p not in top10]
    print("Top-10 by mean |SHAP| (training):")
    for f, v in ranking[:10]:
        print(f"  {v:7.4f}  {f}  ({lab(f)})")
    np.save(os.path.join(out_dir, "shap_values_train_full.npy"), shap_vals)
    with open(os.path.join(out_dir, "shap_ranking.json"), "w") as f:
        json.dump({"ranking": ranking, "top10": top10,
                   "basis": "mean |SHAP| on training data, locked primary"}, f, indent=2)

    # ---- 2. Reduced model on top-10 ---------------------------------------
    print(f"\nReduced model (top-10 + {2} protected indicators = "
          f"{len(top10)+2} features, threshold rule on validation)")
    red = ImbPipeline(preprocessing_steps() + [
        ("smote_enn", AutoSMOTENCENN(random_state=RANDOM_STATE)),
        ("classifier", CatBoostClassifier(**bp)),
    ])
    red.fit(Xtr[reduced_features], ytr)
    prob_val_r = red.predict_proba(Xva[reduced_features])[:, 1]
    cal_r = fit_calibrator(yva, prob_val_r)
    thr_r, s_r, sp_r = lock_threshold(yva, cal_r.predict(prob_val_r))
    print(f"  reduced threshold locked on validation: {thr_r:.4f} "
          f"(val sens {s_r:.3f}, spec {sp_r:.3f})")

    raw_te_r = red.predict_proba(Xte[reduced_features])[:, 1]
    prob_te_r = cal_r.predict(raw_te_r)
    prob_va_r = cal_r.predict(prob_val_r)
    reduced_results = {
        "top10": top10,
        "reduced_features": reduced_features,
        "threshold": round(thr_r, 4),
        "validation": {"unweighted": binary_metrics(yva, prob_va_r, thr_r,
                                                    raw_scores=prob_val_r),
                       "survey_weighted": binary_metrics(yva, prob_va_r, thr_r, swva,
                                                         raw_scores=prob_val_r)},
        "test": {"unweighted": binary_metrics(yte, prob_te_r, thr_r,
                                              raw_scores=raw_te_r),
                 "survey_weighted": binary_metrics(yte, prob_te_r, thr_r, swte,
                                                   raw_scores=raw_te_r)},
        "calibration_test": calibration_summary(yte, prob_te_r),
    }
    t = reduced_results["test"]["unweighted"]
    print(f"  reduced TEST (single pass): AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  "
          f"spec {t['specificity']:.3f}")
    joblib.dump({"pipeline": red, "calibrator": cal_r, "threshold": thr_r,
                 "feature_names": reduced_features, "locked_run": LOCKED_RUN,
                 "created": datetime.now().isoformat(timespec="seconds")},
                os.path.join(out_dir, "reduced_model_bundle.pkl"))

    # primary test probabilities for figures
    raw_te_f = primary.predict_proba(Xte)[:, 1]
    raw_va_f = primary.predict_proba(Xva)[:, 1]
    prob_te_f = cal_full.predict(raw_te_f)
    prob_va_f = cal_full.predict(raw_va_f)
    full_test = binary_metrics(yte, prob_te_f, thr_full, raw_scores=raw_te_f)

    with open(os.path.join(out_dir, "reduced_model_results.json"), "w") as f:
        json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
                   "locked_run": LOCKED_RUN, "smoke": args.smoke,
                   "full_model_test": full_test, "reduced": reduced_results}, f, indent=2)

    # ---- 3. Figures --------------------------------------------------------
    print("\nWriting figures ->", fig_dir)

    # ROC full vs reduced (test) — raw scores, matching the reported AUC
    plt.figure(figsize=(6.5, 6))
    for prob, label, style in ((raw_te_f, "Full model (22 features)", "-"),
                               (raw_te_r, "Reduced model (top 10 + indicators)", "--")):
        fpr, tpr, _ = roc_curve(yte, prob)
        plt.plot(fpr, tpr, style, lw=2,
                 label=f"{label}, AUC {roc_auc_score(yte, prob):.3f}")
    plt.plot([0, 1], [0, 1], ":", color="gray", lw=1)
    plt.xlabel("1 − Specificity"); plt.ylabel("Sensitivity")
    plt.title("Held-out test set"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_roc.png"), dpi=300); plt.close()

    # Metrics bars at locked thresholds (test)
    red_t = reduced_results["test"]["unweighted"]
    ms = ["sensitivity", "specificity", "ppv", "npv"]
    x = np.arange(len(ms)); w = 0.36
    plt.figure(figsize=(7, 5))
    plt.bar(x - w / 2, [full_test[m] for m in ms], w, label="Full (22)")
    plt.bar(x + w / 2, [red_t[m] for m in ms], w, label="Reduced (12)")
    plt.xticks(x, [m.upper() if len(m) == 3 else m.capitalize() for m in ms])
    plt.ylim(0, 1); plt.ylabel("Value")
    plt.title("Test performance at locked thresholds (sens ≥ 0.80 rule)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_metrics.png"), dpi=300); plt.close()

    # SHAP ranking bar (all 22)
    plt.figure(figsize=(8, 7))
    yy = np.arange(len(ranking))[::-1]
    plt.barh(yy, [v for _, v in ranking])
    plt.yticks(yy, [lab(f) for f, _ in ranking], fontsize=9)
    plt.xlabel("Mean |SHAP| (training data)")
    plt.title("Feature importance — locked primary model")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_shap_ranking.png"), dpi=300); plt.close()

    # Beeswarm if shap available
    try:
        import shap as shap_pkg
        plt.figure()
        shap_pkg.summary_plot(shap_vals, Xtr_sel,
                              feature_names=[lab(n) for n in names],
                              show=False, max_display=22)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "figure_shap_summary.png"),
                    dpi=300, bbox_inches="tight")
        plt.close("all")
        print("  beeswarm written (shap package found)")
    except ImportError:
        print("  shap package not installed — beeswarm skipped (bar ranking covers it)")

    # Calibration (reliability) curves
    def reliability(y, p, bins=10):
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = 0, 1
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        obs = [np.asarray(y)[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
        exp = [p[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
        return np.array(exp), np.array(obs)

    plt.figure(figsize=(6, 6))
    for y_, p_, label in ((yva, prob_va_f, "Validation"), (yte, prob_te_f, "Test")):
        e, o = reliability(np.asarray(y_), np.asarray(p_))
        plt.plot(e, o, "o-", label=label)
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("Predicted probability (calibrated)"); plt.ylabel("Observed prevalence")
    plt.title("Calibration — full model"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "efigure_calibration.png"), dpi=300); plt.close()

    # Decision curve (test)
    ths = np.linspace(0.02, 0.6, 59)
    n = len(yte); prev = np.asarray(yte).mean()
    nb_model, nb_all = [], []
    for pt in ths:
        yhat = prob_te_f >= pt
        tp = float(((yhat == 1) & (np.asarray(yte) == 1)).sum())
        fp = float(((yhat == 1) & (np.asarray(yte) == 0)).sum())
        nb_model.append(tp / n - fp / n * pt / (1 - pt))
        nb_all.append(prev - (1 - prev) * pt / (1 - pt))
    plt.figure(figsize=(7, 5))
    plt.plot(ths, nb_model, lw=2, label="Full model")
    plt.plot(ths, nb_all, "--", label="Screen all")
    plt.axhline(0, color="gray", lw=1, label="Screen none")
    plt.ylim(-0.05, max(nb_model) + 0.05)
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title("Decision curve — held-out test set"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "efigure_decision_curve.png"), dpi=300); plt.close()

    print(f"\nDone. Artifacts: {out_dir}/ | Figures: {fig_dir}/")
    # [25 Aug] Deliberately NOT creating notebooks/.nb05_unlocked: notebook 05
    # remains permanently superseded by this script, and its stale guard must
    # keep blocking execution of the pre-R3 code.


if __name__ == "__main__":
    main()
