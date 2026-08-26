"""
run_uncertainty.py — bootstrap confidence intervals for the locked results.

Computes uncertainty for the already-reported point estimates WITHOUT any new
modeling or threshold decisions: predictions are regenerated deterministically
from the locked pipelines, calibrators, and thresholds, then resampled.

  - Stratified bootstrap (default 2000 resamples, seed 42) of the TEST set
    for: primary (22 features), reduced (top-10 + 2 indicators), and the
    no-resampling sensitivity model is NOT refit here — only artifacts that
    were saved are used (primary + reduced). CIs for the sensitivity arms can
    be added the same way if their pipelines are persisted.
  - Metrics: AUC, sensitivity, specificity, PPV, NPV at the locked thresholds,
    unweighted and survey-weighted (weights resampled with the rows).
  - Paired full-vs-reduced AUC difference with bootstrap CI (same resample
    indices for both models — a paired comparison).

Output: outputs/final_analyses_<runid>/uncertainty_bootstrap.json

Run from the repo root:
    python run_uncertainty.py            (2000 resamples, ~1-2 min)
    python run_uncertainty.py --smoke    (200 resamples, /tmp output)
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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

# [2026-08-26] auto-detect the newest tuning run (the corrective run,
# once notebook 04 has been re-executed); falls back to the prior lock.
import glob as _glob
_runs = sorted(_glob.glob(os.path.join(HERE, "notebooks", "tuning_results_*")))
LOCKED_RUN = os.path.basename(_runs[-1]) if _runs else "tuning_results_20260824_140539"
SEED = 42

from sklearn.metrics import roc_auc_score                            # noqa: E402
from run_final_analyses import binary_metrics                        # noqa: E402


def boot_ci(stat_fn, n, n_boot, rng, strata):
    """Stratified bootstrap CI: resample indices within outcome strata."""
    idx_pos = np.where(strata == 1)[0]
    idx_neg = np.where(strata == 0)[0]
    stats = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                             rng.choice(idx_neg, len(idx_neg), replace=True)])
        stats.append(stat_fn(bi))
    arr = np.array(stats, dtype=float)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return arr, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    n_boot = 200 if args.smoke else args.n_boot

    run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)
    fin_dir = os.path.join(HERE, "outputs", f"final_analyses_{LOCKED_RUN.split('_', 2)[2]}")
    red_dir = os.path.join(HERE, "outputs", f"reduced_model_{LOCKED_RUN.split('_', 2)[2]}")
    out_path = ("/tmp/uncertainty_smoke.json" if args.smoke else
                os.path.join(fin_dir, "uncertainty_bootstrap.json"))

    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    primary = joblib.load(os.path.join(run_dir, "catboost_best_model.pkl"))
    lock = joblib.load(os.path.join(fin_dir, "locked_threshold_calibration.pkl"))
    red_bundle = joblib.load(os.path.join(red_dir, "reduced_model_bundle.pkl"))

    Xte, yte, swte = art["X_test_feat"], np.asarray(art["y_test"], float), np.asarray(art["sw_test"], float)

    prob_full = lock["calibrator"].predict(primary.predict_proba(Xte)[:, 1])
    thr_full = lock["threshold"]
    rf = red_bundle["feature_names"]
    prob_red = red_bundle["calibrator"].predict(
        red_bundle["pipeline"].predict_proba(Xte[rf])[:, 1])
    thr_red = red_bundle["threshold"]

    rng = np.random.default_rng(SEED)
    results = {"generated": datetime.now().isoformat(timespec="seconds"),
               "locked_run": LOCKED_RUN, "n_boot": n_boot, "seed": SEED,
               "n_test": int(len(yte)), "models": {}}

    for tag, prob, thr in (("primary_22", prob_full, thr_full),
                           ("reduced_top10_plus_indicators", prob_red, thr_red)):
        point_u = binary_metrics(yte, prob, thr)
        point_w = binary_metrics(yte, prob, thr, swte)

        def stat(bi, prob=prob, thr=thr):
            mu = binary_metrics(yte[bi], prob[bi], thr)
            mw = binary_metrics(yte[bi], prob[bi], thr, swte[bi])
            return [mu["auc"], mu["sensitivity"], mu["specificity"], mu["ppv"], mu["npv"],
                    mw["auc"], mw["sensitivity"], mw["specificity"]]

        _, lo, hi = boot_ci(stat, len(yte), n_boot, rng, yte)
        keys = ["auc", "sensitivity", "specificity", "ppv", "npv",
                "auc_weighted", "sensitivity_weighted", "specificity_weighted"]
        pts = [point_u["auc"], point_u["sensitivity"], point_u["specificity"],
               point_u["ppv"], point_u["npv"],
               point_w["auc"], point_w["sensitivity"], point_w["specificity"]]
        results["models"][tag] = {
            k: {"point": round(p, 4), "ci95": [round(float(l), 4), round(float(h), 4)]}
            for k, p, l, h in zip(keys, pts, lo, hi)}
        m = results["models"][tag]
        print(f"{tag}: AUC {m['auc']['point']} ({m['auc']['ci95'][0]}-{m['auc']['ci95'][1]})  "
              f"sens {m['sensitivity']['point']} ({m['sensitivity']['ci95'][0]}-"
              f"{m['sensitivity']['ci95'][1]})  "
              f"spec {m['specificity']['point']} ({m['specificity']['ci95'][0]}-"
              f"{m['specificity']['ci95'][1]})")

    # paired full-vs-reduced AUC difference (same bootstrap indices)
    def diff_stat(bi):
        return [roc_auc_score(yte[bi], prob_full[bi]) - roc_auc_score(yte[bi], prob_red[bi])]
    _, dlo, dhi = boot_ci(diff_stat, len(yte), n_boot, rng, yte)
    d_point = roc_auc_score(yte, prob_full) - roc_auc_score(yte, prob_red)
    results["paired_auc_difference_full_minus_reduced"] = {
        "point": round(float(d_point), 4),
        "ci95": [round(float(dlo[0]), 4), round(float(dhi[0]), 4)],
        "note": "paired stratified bootstrap; CI covering 0 = no detectable difference",
    }
    p = results["paired_auc_difference_full_minus_reduced"]
    print(f"paired AUC diff (full - reduced): {p['point']} ({p['ci95'][0]} to {p['ci95'][1]})")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
