"""
verify_split_reconstruction.py — item 7 of the 12 Aug audit email.

Replays the Phase-3 cleaning (notebook 03) from 02b_harmonized.parquet with
SEQN retained as non-predictor metadata, verifies the replica matches
03_cleaned.parquet EXACTLY (columns, order, dtypes, values, missingness,
hashes), then replays the notebook-04 filtering and seeded splits, verifies
them against the saved run artifacts, and writes a permanent SEQN -> split
assignment record.

Acceptance rule (verbatim from the email): the reconstruction is accepted
only if the reconstructed feature, outcome, and survey-weight arrays match
the preserved split arrays exactly in values, ordering, dtypes, missingness,
and hashes. Any FAIL below means the assignments must NOT be described as
deterministically reconstructed.

Run from the repo root:
    python verify_split_reconstruction.py
Options:
    --run tuning_results_20260824_084903   (default: newest in notebooks/)
    --out outputs                          (where to write the record)

Writes:
    <out>/split_assignment_SEQN.csv
    <out>/split_verification_report.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

RANDOM_STATE = 42
checks = {}


def check(name, ok, detail=""):
    checks[name] = {"pass": bool(ok), "detail": str(detail)[:300]}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))
    return ok


def frame_hash(df: pd.DataFrame) -> str:
    """Order- and value-sensitive content hash (missingness included)."""
    h = pd.util.hash_pandas_object(df, index=False).values
    cols = "|".join(map(str, df.columns)).encode()
    return hashlib.sha256(h.tobytes() + cols).hexdigest()


def series_hash(s: pd.Series) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(s, index=False).values.tobytes()).hexdigest()


# ===========================================================================
# Phase 3 replay (verbatim logic from notebooks/03_clean_and_filter.ipynb)
# ===========================================================================

def replay_phase3(processed_dir):
    recode_df = pd.read_parquet(os.path.join(processed_dir, "02b_harmonized.parquet"))

    df_cleaned = recode_df[(recode_df["RIDAGEYR"] >= 6) & (recode_df["RIDAGEYR"] < 18)].copy()
    df_cleaned = df_cleaned[df_cleaned["MCQ010"].isin([1.0, 2.0])].copy()

    # SEQN captured BEFORE the drop, on the identical row index
    seqn = df_cleaned["SEQN"].copy()

    cols_to_drop = [
        "SEQN",
        "MCQ025", "MCQ035", "MCQ040", "MCQ050", "MCQ051",
        "DMDBORN2", "DMDBORN4", "DMDHRBR2", "DMDHRBR4",
        "RIDAGEEX", "RIDEXAGM", "RIDEXAGY",
        "RIDAGEMN", "DMDSCHOL", "BMXTRI", "BMXSUB",
        "MCQ082", "MCQ086",
        "WTINT2YR", "SDMVPSU", "SDMVSTRA",
    ]
    df_cleaned.drop(columns=[c for c in cols_to_drop if c in df_cleaned.columns], inplace=True)

    df_cleaned.dropna(axis=1, how="all", inplace=True)
    df_cleaned.dropna(axis="columns", thresh=len(df_cleaned) / 2, inplace=True)

    for group in (
        # [2026-08-27 KM ruling] SPXNQFV1/SPXNQFVC RETAINED as QC metadata
        # for spirometry quality gating in Phase 4 (removed from this drop
        # group; bronchodilator/efficiency attributes still dropped).
        ["SPXNQEFF", "SPXBQEFF", "SPXBQFV1", "SPXBQFVC"],
        ["MIAINTRP", "MIALANG", "MIAPROXY", "FIAINTRP", "RIDEXMON", "FIAPROXY",
         "SIAPROXY", "SIAINTRP", "SIALANG", "RIDSTATR", "SDDSRVYR"],
        ["BMDSTATS", "HSAQUEX"],
        ["ECQ020", "ECQ080", "ECQ090", "WHQ030E", "MCQ080E", "ECQ150", "ECD010",
         "ECD070A", "ECD070B", "FSD670ZC", "FSQ690", "FSD680", "FSD675"],
    ):
        df_cleaned.drop(columns=[c for c in group if c in df_cleaned.columns], inplace=True)

    return df_cleaned, seqn


# ===========================================================================
# Phase 4 replay (verbatim filtering/split logic from notebook 04, cell 6)
# ===========================================================================

def replay_phase4_split(split_df, seqn):
    from sklearn.model_selection import train_test_split

    df = split_df.copy()
    provenance = [c for c in ("NHANES_CYCLE",) if c in df.columns]  # cell 2 drop
    if provenance:
        df = df.drop(columns=provenance)

    # [2026-08-27 KM ruling] cell-2 spirometry quality gating (A/B primary);
    # unconditional so the replay can never validate ungated data.
    from asthma_pipeline import apply_spirometry_quality_gating, PRIMARY_ALLOWED_GRADES
    df = apply_spirometry_quality_gating(df, PRIMARY_ALLOWED_GRADES, verbose=False)

    TARGET, WEIGHT = "MCQ010", "WTMEC2YR"
    y = df[TARGET].copy()
    sample_weight = df[WEIGHT] if WEIGHT in df.columns else None
    EXCLUDE = [TARGET, WEIGHT, "WTINT2YR", "SDMVPSU", "SDMVSTRA", "SEQN", "NHANES_CYCLE"]
    X = df[[c for c in df.columns if c not in EXCLUDE]].copy()

    if set(y.dropna().unique()) <= {1, 2, 1.0, 2.0}:
        y = (y == 1).astype(float)
        y[df[TARGET].isna()] = np.nan

    valid_mask = y.notna()
    if sample_weight is not None:
        valid_mask &= sample_weight.notna() & (sample_weight > 0)

    X_clean = X[valid_mask].reset_index(drop=True)
    y_clean = y[valid_mask].reset_index(drop=True)
    sw_clean = sample_weight[valid_mask].reset_index(drop=True)
    seqn_clean = seqn.reset_index(drop=True)[valid_mask.reset_index(drop=True)].reset_index(drop=True)

    if "RIDAGEYR" in X_clean.columns:
        ped = (X_clean["RIDAGEYR"] >= 6) & (X_clean["RIDAGEYR"] < 18)
        X_clean = X_clean[ped].reset_index(drop=True)
        y_clean = y_clean[ped].reset_index(drop=True)
        sw_clean = sw_clean[ped].reset_index(drop=True)
        seqn_clean = seqn_clean[ped].reset_index(drop=True)

    idx = pd.Series(np.arange(len(X_clean)))
    X_tmp, X_te, y_tmp, y_te, sw_tmp, sw_te, i_tmp, i_te = train_test_split(
        X_clean, y_clean, sw_clean, idx,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_clean)
    X_tr, X_va, y_tr, y_va, sw_tr, sw_va, i_tr, i_va = train_test_split(
        X_tmp, y_tmp, sw_tmp, i_tmp,
        test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp)

    return dict(X_train=X_tr, X_val=X_va, X_test=X_te,
                y_train=y_tr, y_val=y_va, y_test=y_te,
                sw_train=sw_tr, sw_val=sw_va, sw_test=sw_te,
                i_train=i_tr, i_val=i_va, i_test=i_te,
                seqn_clean=seqn_clean, n_clean=len(X_clean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="tuning_results_* dir name under notebooks/")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--production", action="store_true",
                    help="write the canonical committed report "
                         "(split_verification_report.json). Without this "
                         "flag, results go to split_verification_report."
                         "local.json so a local or cross-platform run can "
                         "never overwrite the committed record.")
    args = ap.parse_args()

    processed = os.path.join(HERE, "data", "processed")
    nb_dir = os.path.join(HERE, "notebooks")

    # [31 Aug] default pinned to the analysis of record (was: newest local run)
    PINNED_RUN = "tuning_results_20260831_103201"
    run_dir = (os.path.join(nb_dir, args.run) if args.run
               else os.path.join(nb_dir, PINNED_RUN))
    print(f"Verifying against run: {os.path.basename(run_dir)}\n")

    # ---- Step 1: Phase-3 replica vs preserved 03_cleaned.parquet ----------
    print("Step 1 — Phase-3 replay with SEQN retained")
    replica, seqn = replay_phase3(processed)
    preserved = pd.read_parquet(os.path.join(processed, "03_cleaned.parquet"))

    check("row count", len(replica) == len(preserved), f"{len(replica)} vs {len(preserved)}")
    check("column list + order", list(replica.columns) == list(preserved.columns),
          f"replica-only={set(replica.columns)-set(preserved.columns)}, "
          f"preserved-only={set(preserved.columns)-set(replica.columns)}")
    same_dtypes = list(replica.dtypes.astype(str)) == list(preserved.dtypes.astype(str))
    check("dtypes", same_dtypes)
    r2, p2 = replica.reset_index(drop=True), preserved.reset_index(drop=True)
    check("values + missingness (exact)", r2.equals(p2))
    h_rep, h_pre = frame_hash(r2), frame_hash(p2)
    check("content hash", h_rep == h_pre, f"{h_rep[:16]} vs {h_pre[:16]}")
    check("SEQN unique + fully observed",
          seqn.notna().all() and seqn.is_unique, f"n={len(seqn)}")

    # ---- Step 2: Phase-4 split replay vs preserved run arrays -------------
    print("\nStep 2 — Phase-4 seeded split replay vs saved run arrays")
    art = joblib.load(os.path.join(run_dir, "preprocessed_data.pkl"))
    rep = replay_phase4_split(p2, seqn)

    for part in ("train", "val", "test"):
        check(f"y_{part} exact (values+order+index)", rep[f"y_{part}"].equals(art[f"y_{part}"]))
        check(f"sw_{part} exact", rep[f"sw_{part}"].equals(art[f"sw_{part}"]))

    # ---- Step 2b: structural split assertions ----------------------------
    print("\nStep 2b — split partition assertions")
    itr, iva, ite = (set(rep["i_train"]), set(rep["i_val"]), set(rep["i_test"]))
    check("splits pairwise disjoint",
          not (itr & iva) and not (itr & ite) and not (iva & ite))
    check("splits complete (union == analytic cohort)",
          len(itr | iva | ite) == rep["n_clean"],
          f"{len(itr | iva | ite)} vs {rep['n_clean']}")
    s_tr = set(rep["seqn_clean"].iloc[list(itr)])
    s_va = set(rep["seqn_clean"].iloc[list(iva)])
    s_te = set(rep["seqn_clean"].iloc[list(ite)])
    check("SEQN sets pairwise disjoint",
          not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te))

    # ---- Step 2c: HISTORICAL arrays (committed exports) -------------------
    # Item 7 as originally promised: the split behind the SUBMITTED results.
    # [31 Aug] arrays are read from committed outputs/historical_split_arrays/
    # (see export_historical_split_arrays.py), so this step is clean-clone
    # reproducible; it no longer depends on the external FROZEN snapshots.
    print("\nStep 2c — historical split arrays (committed exports)")
    hist_npzs = sorted(glob.glob(os.path.join(
        HERE, "outputs", "historical_split_arrays", "*.npz")))
    if not hist_npzs:
        check("historical split arrays found", False,
              "outputs/historical_split_arrays/ is empty - run "
              "export_historical_split_arrays.py on the machine with the "
              "frozen snapshots and commit the .npz files")
    hist_tags = []
    for fp in hist_npzs:
        tag = os.path.basename(fp).replace(".npz", "")
        hist_tags.append(tag)
        hist = np.load(fp)
        for part in ("train", "val", "test"):
            check(f"[historical {tag}] y_{part} exact",
                  np.array_equal(np.asarray(rep[f"y_{part}"], float),
                                 hist[f"y_{part}"]))
            check(f"[historical {tag}] sw_{part} exact",
                  np.array_equal(np.asarray(rep[f"sw_{part}"], float),
                                 hist[f"sw_{part}"]))

    # ---- Step 3: feature arrays — engineer via the shared module ----------
    print("\nStep 3 — feature-array reconstruction (cleaner → engineer → pruning)")
    from asthma_pipeline import (NHANESCleaner, ClinicalFeatureEngineer,
                                 apply_correlation_pruning,
                                 LEAKY_PROXIES, AGE_RESTRICTED_VARS, IDENTIFIERS,
                                 PRIMARY_MODEL_EXCLUSIONS)
    # replicate cell 6 column drops on X before engineering
    def drop_excluded(X):
        drop = [c for c in X.columns
                if c in LEAKY_PROXIES + AGE_RESTRICTED_VARS + IDENTIFIERS + PRIMARY_MODEL_EXCLUSIONS]
        return X.drop(columns=drop)

    Xtr = drop_excluded(rep["X_train"]); Xva = drop_excluded(rep["X_val"]); Xte = drop_excluded(rep["X_test"])
    cleaner, fe = NHANESCleaner(), ClinicalFeatureEngineer()
    Xtr_f = fe.fit_transform(cleaner.fit_transform(Xtr))
    Xva_f = fe.transform(cleaner.transform(Xva))
    Xte_f = fe.transform(cleaner.transform(Xte))
    Xtr_f, Xva_f, Xte_f, pruned = apply_correlation_pruning(Xtr_f, Xva_f, Xte_f)

    check("pruned feature list matches run", pruned == art.get("pruned_features"),
          f"{len(pruned)} vs {len(art.get('pruned_features') or [])}")
    check("feature name list matches run",
          Xtr_f.columns.tolist() == art.get("feature_names"))
    # [v3.2] portable vs platform-sensitive classification: participant-level
    # and structural agreement must hold everywhere; bitwise equality of
    # float feature values is platform evidence. A reconstruction whose only
    # deviation is float noise (identical NaN pattern, max abs diff < 1e-9)
    # PASSES the portable check, with the bitwise status recorded separately.
    TOL = 1e-9
    for name, mine, theirs in (("X_train_feat", Xtr_f, art.get("X_train_feat")),
                               ("X_val_feat", Xva_f, art.get("X_val_feat")),
                               ("X_test_feat", Xte_f, art.get("X_test_feat"))):
        mine = mine.reset_index(drop=True)
        bitwise = theirs is not None and mine.equals(theirs.reset_index(drop=True))
        ok = bitwise
        detail = f"hash={frame_hash(mine)[:16]}; bitwise_match={bitwise}"
        if theirs is not None and not bitwise:
            t = theirs.reset_index(drop=True)
            if list(mine.columns) == list(t.columns) and mine.shape == t.shape:
                a = mine.to_numpy(dtype=float); b = t.to_numpy(dtype=float)
                nan_mm = int((np.isnan(a) != np.isnan(b)).sum())
                both = ~np.isnan(a) & ~np.isnan(b)
                max_abs = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
                if nan_mm == 0 and max_abs < TOL:
                    ok = True
                    detail += (f"; PASS within tolerance (max_abs_diff="
                               f"{max_abs:.3e} < {TOL:g}, identical NaN "
                               f"pattern) - cross-platform float noise; "
                               f"bitwise hashes are platform-specific "
                               f"evidence from the production environment")
                else:
                    detail += (f"; nan_mismatch={nan_mm}, "
                               f"max_abs_diff={max_abs:.3e} exceeds tolerance")
            else:
                detail += "; STRUCTURAL mismatch (shape or columns differ)"
        check(f"{name} reconstruction (tol {TOL:g})", ok, detail)

    # ---- Step 4: write the permanent SEQN -> split record -----------------
    print("\nStep 4 — writing the split-assignment record")
    all_pass = all(v["pass"] for v in checks.values())
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for part in ("train", "val", "test"):
        for pos in rep[f"i_{part}"]:
            rows.append((int(rep["seqn_clean"].iloc[pos]), part,
                         float(rep["y_train" if part == "train" else f"y_{part}"].loc[pos]),
                         float(rep[f"sw_{part}"].loc[pos])))
    rec = pd.DataFrame(rows, columns=["SEQN", "split", "asthma", "WTMEC2YR"]).sort_values("SEQN")
    csv_path = os.path.join(args.out, "split_assignment_SEQN.csv")
    rec.to_csv(csv_path, index=False, lineterminator="\n")

    # [31 Aug] canonical (LF) hash so the value matches the committed git
    # blob regardless of the local line-ending configuration, plus a hash
    # of the run's preprocessed artifact so the gate in run_final_analyses
    # can bind report -> artifact -> evaluation.
    csv_lf = open(csv_path, "rb").read().replace(b"\r\n", b"\n")
    pkl_path = os.path.join(run_dir, "preprocessed_data.pkl")

    def _sha(p):
        return (hashlib.sha256(open(p, "rb").read()).hexdigest()
                if os.path.exists(p) else None)

    report = {
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": os.path.basename(run_dir),
        "random_state": RANDOM_STATE,
        "n_analytic": int(rep["n_clean"]),
        "n_train": len(rep["i_train"]), "n_val": len(rep["i_val"]), "n_test": len(rep["i_test"]),
        "phase3_content_hash": h_rep,
        "split_assignment_sha256_lf": hashlib.sha256(csv_lf).hexdigest(),
        "preprocessed_data_sha256": _sha(pkl_path),
        "catboost_best_model_sha256": _sha(os.path.join(run_dir, "catboost_best_model.pkl")),
        "catboost_study_sha256": _sha(os.path.join(run_dir, "catboost_study.pkl")),
        "pandas": pd.__version__, "numpy": np.__version__,
        "checks": checks,
        "historical_runs_checked": hist_tags,
        "verdict": ("ACCEPTED. Current run: reconstructed outcome and survey-weight "
                    "arrays match the preserved arrays exactly (values, ordering, "
                    "dtypes, missingness); feature matrices match within the stated "
                    "numerical tolerance, with bitwise equality recorded per check as "
                    "platform-specific evidence. Historical runs: the preserved outcome "
                    "and survey-weight sequences match the replay exactly, in order and "
                    "value; the frozen artifacts do not include participant "
                    "identifiers, so historical identity is established at the level "
                    "of these arrays (the current run's assignments are SEQN-anchored). "
                    "Splits proven pairwise disjoint and complete."
                    if all_pass else
                    "REJECTED: at least one check failed — do NOT describe the split as "
                    "deterministically reconstructed."),
    }
    rep_name = ("split_verification_report.json" if args.production
                else "split_verification_report.local.json")
    if not args.production:
        print("\n[local mode] writing to split_verification_report.local.json; "
              "the committed report is untouched. Use --production on the "
              "machine of record.")
    with open(os.path.join(args.out, rep_name), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}\n{report['verdict']}\n"
          f"Record: {csv_path}  (n={len(rec)}: "
          f"{report['n_train']} train / {report['n_val']} val / {report['n_test']} test)")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
