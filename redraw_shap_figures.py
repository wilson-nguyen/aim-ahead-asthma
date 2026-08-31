"""
redraw_shap_figures.py — regenerate the two SHAP figures with display labels.

Purpose: DMDHHSIZ (household size) entered the top-10 in the final run and
had no display label, so figure_shap_ranking.png / figure_shap_summary.png
showed the raw variable code. This script redraws ONLY those two figures
from the saved run artifacts. No model is fitted, no predictions are made,
no test data is touched — plotting only.

Run from the repo root, after run_reduced_model_and_figures.py:
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

run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)
red_dir = os.path.join(HERE, "outputs", f"reduced_model_{LOCKED_RUN.split('_', 2)[2]}")
fig_dir = os.path.join(HERE, "outputs", "figures_R3")

sh = json.load(open(os.path.join(red_dir, "shap_ranking.json")))
ranking = [(f, float(v)) for f, v in sh["ranking"]]
shap_vals = np.load(os.path.join(red_dir, "shap_values_train_full.npy"))

# ranking bar (all 22)
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

# beeswarm (transform-only walk to rebuild the plotted training matrix)
try:
    import shap as shap_pkg
    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    primary = joblib.load(os.path.join(run_dir, "catboost_best_model.pkl"))
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
