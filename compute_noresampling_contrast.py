"""
compute_noresampling_contrast.py — paired uncertainty for the no-resampling
sensitivity arm versus the primary model.

The no-resampling variant's pipeline was not persisted by run_final_analyses,
so this script refits it DETERMINISTICALLY (fixed seed, no sampler, same
tuned hyperparameters and feature-selection rule) and verifies the refit
reproduces the committed arm-A test metrics before computing anything. No
new modeling decision is made: the arm was declared and reported already;
this quantifies the already-reported contrast with paired uncertainty.

Run from the repo root, after run_final_analyses.py:
    python compute_noresampling_contrast.py

Writes: outputs/final_analyses_<runid>/noresampling_contrast.json
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

# [2026-08-31] pinned to the analysis of record
PINNED_RUN = "tuning_results_20260831_103201"
LOCKED_RUN = PINNED_RUN
SEED = 42

from sklearn.metrics import roc_auc_score                            # noqa: E402
from run_final_analyses import (                                     # noqa: E402
    require_verified, build_pipeline, fit_calibrator, lock_threshold,
    binary_metrics,
)


def main():
    require_verified(LOCKED_RUN)
    run_dir = os.path.join(HERE, "notebooks", LOCKED_RUN)
    fin_dir = os.path.join(HERE, "outputs",
                           f"final_analyses_{LOCKED_RUN.split('_', 2)[2]}")

    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    primary = joblib.load(os.path.join(run_dir, "catboost_best_model.pkl"))
    study = joblib.load(os.path.join(run_dir, "catboost_study.pkl"))
    bp = study.best_params.copy()
    bp.update({"random_state": SEED, "verbose": False, "thread_count": -1})

    Xtr, ytr = art["X_train_feat"], art["y_train"]
    Xva, yva = art["X_val_feat"], np.asarray(art["y_val"], float)
    Xte, yte = art["X_test_feat"], np.asarray(art["y_test"], float)
    FN = art["feature_names"]

    # deterministic refit of arm A (no resampling)
    pipe = build_pipeline(FN, bp, resample=False)
    pipe.fit(Xtr, ytr)
    prob_val = pipe.predict_proba(Xva)[:, 1]
    cal = fit_calibrator(yva, prob_val)
    thr, _, _ = lock_threshold(yva, cal.predict(prob_val))
    raw_a = pipe.predict_proba(Xte)[:, 1]
    m = binary_metrics(yte, cal.predict(raw_a), thr, raw_scores=raw_a)

    # verification against the committed arm-A results before proceeding
    committed = json.load(open(os.path.join(
        fin_dir, "final_analyses_results.json")))["analyses"]["no_resampling"]
    ref = committed["test"]["unweighted"]
    for k in ("auc", "sensitivity", "specificity"):
        if abs(m[k] - ref[k]) > 5e-4:
            sys.exit(f"ABORT: deterministic refit does not reproduce the "
                     f"committed no-resampling arm ({k}: {m[k]} vs {ref[k]}). "
                     f"Environment differs from the machine of record; do not "
                     f"report this contrast from here.")
    print(f"refit verified against committed arm A: AUC {m['auc']} "
          f"(committed {ref['auc']})")

    raw_p = primary.predict_proba(Xte)[:, 1]
    ip, iq = np.where(yte == 1)[0], np.where(yte == 0)[0]
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        bi = np.concatenate([rng.choice(ip, len(ip), replace=True),
                             rng.choice(iq, len(iq), replace=True)])
        diffs.append(roc_auc_score(yte[bi], raw_p[bi])
                     - roc_auc_score(yte[bi], raw_a[bi]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = {
        "primary_minus_noresampling_auc": {
            "point": round(float(roc_auc_score(yte, raw_p)
                                 - roc_auc_score(yte, raw_a)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_boot": 2000, "seed": SEED,
            "note": ("paired stratified bootstrap of raw test scores; "
                     "conditional on the fitted models and the historically "
                     "reused internal split. A negative value favors the "
                     "no-resampling variant. Reported for disclosure; the "
                     "pre-declared resampling-based primary is retained "
                     "rather than switched post hoc."),
        },
        "refit_verified_against_committed_arm": True,
    }
    path = os.path.join(fin_dir, "noresampling_contrast.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    p = out["primary_minus_noresampling_auc"]
    print(f"paired AUC diff (primary - no resampling): {p['point']} "
          f"({p['ci95'][0]} to {p['ci95'][1]})")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
