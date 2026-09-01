"""
redraw_shap_figures.py — regenerate ALL manuscript figures from the saved
artifacts, plot-only.

No model is fitted, no threshold chosen, no decision made: fitted pipelines,
calibrators, and locked thresholds are loaded from the committed artifacts
and predictions are recomputed deterministically for plotting.

History: first version (28 Aug) redrew only the two SHAP figures after a
label fix. Extended 31 Aug to all six figures so titles describe the test
split accurately ("historically reused internal split" rather than
"held-out"), per external review.

Run from the repo root:
    python redraw_shap_figures.py
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

# [2026-08-31] pinned to the analysis of record
PINNED_RUN = "tuning_results_20260831_103201"
LOCKED_RUN = PINNED_RUN

from run_reduced_model_and_figures import lab                        # noqa: E402
from run_final_analyses import binary_metrics, require_verified      # noqa: E402

TEST_LABEL = "Test set (historically reused internal split)"


def main():
    require_verified(LOCKED_RUN)
    run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)
    rid = LOCKED_RUN.split("_", 2)[2]
    fin_dir = os.path.join(HERE, "outputs", f"final_analyses_{rid}")
    red_dir = os.path.join(HERE, "outputs", f"reduced_model_{rid}")
    fig_dir = os.path.join(HERE, "outputs", "figures_R3")

    from sklearn.metrics import roc_curve, roc_auc_score

    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    primary = joblib.load(os.path.join(run_dir, "catboost_best_model.pkl"))
    lock = joblib.load(os.path.join(fin_dir, "locked_threshold_calibration.pkl"))
    bundle = joblib.load(os.path.join(red_dir, "reduced_model_bundle.pkl"))
    sh = json.load(open(os.path.join(red_dir, "shap_ranking.json")))
    shap_vals = np.load(os.path.join(red_dir, "shap_values_train_full.npy"))

    Xva, yva = art["X_val_feat"], np.asarray(art["y_val"], float)
    Xte, yte = art["X_test_feat"], np.asarray(art["y_test"], float)
    rf = bundle["feature_names"]

    raw_te_f = primary.predict_proba(Xte)[:, 1]
    raw_va_f = primary.predict_proba(Xva)[:, 1]
    prob_te_f = lock["calibrator"].predict(raw_te_f)
    prob_va_f = lock["calibrator"].predict(raw_va_f)
    raw_te_r = bundle["pipeline"].predict_proba(Xte[rf])[:, 1]
    prob_te_r = bundle["calibrator"].predict(raw_te_r)

    full_test = binary_metrics(yte, prob_te_f, lock["threshold"], raw_scores=raw_te_f)
    red_test = binary_metrics(yte, prob_te_r, bundle["threshold"], raw_scores=raw_te_r)

    # ---- ROC, full vs reduced (raw scores, matching the reported AUC) ------
    plt.figure(figsize=(6.5, 6))
    for prob, label, style in ((raw_te_f, "Full model (22 features)", "-"),
                               (raw_te_r, "Reduced model (top 10 + indicators)", "--")):
        fpr, tpr, _ = roc_curve(yte, prob)
        plt.plot(fpr, tpr, style, lw=2,
                 label=f"{label}, AUC {roc_auc_score(yte, prob):.3f}")
    plt.plot([0, 1], [0, 1], ":", color="gray", lw=1)
    plt.xlabel("1 − Specificity"); plt.ylabel("Sensitivity")
    plt.title(TEST_LABEL); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_roc.png"), dpi=300); plt.close()
    print("redrawn: figure_roc.png")

    # ---- Metrics bars at locked thresholds ---------------------------------
    ms = ["sensitivity", "specificity", "ppv", "npv"]
    x = np.arange(len(ms)); w = 0.36
    plt.figure(figsize=(7, 5))
    plt.bar(x - w / 2, [full_test[m] for m in ms], w, label="Full (22)")
    plt.bar(x + w / 2, [red_test[m] for m in ms], w, label="Reduced (12)")
    plt.xticks(x, [m.upper() if len(m) == 3 else m.capitalize() for m in ms])
    plt.ylim(0, 1); plt.ylabel("Value")
    plt.title(f"{TEST_LABEL}\nperformance at validation-locked thresholds")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_metrics.png"), dpi=300); plt.close()
    print("redrawn: figure_metrics.png")

    # ---- SHAP ranking bar (all 22) -----------------------------------------
    ranking = [(f, float(v)) for f, v in sh["ranking"]]
    plt.figure(figsize=(8, 7))
    yy = np.arange(len(ranking))[::-1]
    plt.barh(yy, [v for _, v in ranking])
    plt.yticks(yy, [lab(f) for f, _ in ranking], fontsize=9)
    plt.xlabel("Mean |SHAP| (training data)")
    plt.title("Feature importance — locked primary model")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_shap_ranking.png"), dpi=300)
    plt.close()
    print("redrawn: figure_shap_ranking.png")

    # ---- Beeswarm (transform-only walk) ------------------------------------
    try:
        import shap as shap_pkg
        names = primary.named_steps["feature_selection"].get_selected_names()
        X = art["X_train_feat"]
        for _, step in primary.steps[:-1]:
            if hasattr(step, "transform"):
                X = step.transform(X)
        plt.figure()
        shap_pkg.summary_plot(shap_vals, X, feature_names=[lab(n) for n in names],
                              show=False, max_display=22)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "figure_shap_summary.png"),
                    dpi=300, bbox_inches="tight")
        plt.close("all")
        print("redrawn: figure_shap_summary.png")
    except ImportError:
        print("shap package not installed — beeswarm skipped")

    # ---- Calibration (reliability) curves ----------------------------------
    def reliability(y, p, bins=10):
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = 0, 1
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        obs = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
        exp = [p[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
        return np.array(exp), np.array(obs)

    plt.figure(figsize=(6, 6))
    for y_, p_, label in ((yva, prob_va_f, "Validation"),
                          (yte, prob_te_f, "Test (reused internal split)")):
        e, o = reliability(np.asarray(y_), np.asarray(p_))
        plt.plot(e, o, "o-", label=label)
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("Predicted probability (calibrated)")
    plt.ylabel("Observed prevalence")
    plt.title("Calibration — full model"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "efigure_calibration.png"), dpi=300)
    plt.close()
    print("redrawn: efigure_calibration.png")

    # ---- Decision curve ----------------------------------------------------
    ths = np.linspace(0.02, 0.6, 59)
    n = len(yte); prev = yte.mean()
    nb_model, nb_all = [], []
    for pt in ths:
        yhat = prob_te_f >= pt
        tp = float(((yhat == 1) & (yte == 1)).sum())
        fp = float(((yhat == 1) & (yte == 0)).sum())
        nb_model.append(tp / n - fp / n * pt / (1 - pt))
        nb_all.append(prev - (1 - prev) * pt / (1 - pt))
    plt.figure(figsize=(7, 5))
    plt.plot(ths, nb_model, lw=2, label="Full model")
    plt.plot(ths, nb_all, "--", label="Screen all")
    plt.axhline(0, color="gray", lw=1, label="Screen none")
    plt.ylim(-0.05, max(nb_model) + 0.05)
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title(f"Decision curve — {TEST_LABEL.lower()}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "efigure_decision_curve.png"), dpi=300)
    plt.close()
    print("redrawn: efigure_decision_curve.png")


if __name__ == "__main__":
    main()
