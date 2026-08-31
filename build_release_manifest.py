"""
build_release_manifest.py — bind the analysis of record to verifiable hashes.

Distinct from freeze_and_manifest.py, which snapshotted the PRE-correction
state in August. This one documents the FINAL R3 analysis so a reader with a
clean clone can confirm they are looking at the same thing we reported:

  - git commit SHA, branch, and whether the working tree was clean
  - the pinned run ID and every committed result file, with SHA-256
  - input data hashes (data/processed/*.parquet, listed not copied)
  - uncommitted-but-required artifacts (model pickles), hashed so their
    identity is checkable even though they are too large for git
  - the headline numbers, read from the result JSONs rather than retyped
  - environment: Python, platform, and pinned package versions

Run from the repo root, after the final analysis scripts:
    python build_release_manifest.py

Writes: outputs/RELEASE_MANIFEST.md  (commit this file)
"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "outputs"


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git(*args, default="(unavailable)"):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, timeout=30).stdout.strip() or default
    except Exception:
        return default


def main():
    runs = sorted(glob.glob(str(REPO / "notebooks" / "tuning_results_*")))
    run_id = os.path.basename(runs[-1]).replace("tuning_results_", "")
    fin = OUT / f"final_analyses_{run_id}"
    red = OUT / f"reduced_model_{run_id}"

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", default="")
    tracked = [
        "notebooks/asthma_pipeline.py", "notebooks/pediatric_corrections.py",
        "notebooks/build_table1.py", "notebooks/03_clean_and_filter.ipynb",
        "notebooks/04_model.ipynb", "verify_split_reconstruction.py",
        "run_final_analyses.py", "run_reduced_model_and_figures.py",
        "run_uncertainty.py", "generate_descriptives.py",
        "redraw_shap_figures.py", "patch11_r3_quality_gating.py",
        "tests/test_cleaner_sentinels.py", "tests/test_quality_gating.py",
        "tests/test_cdc_bmi_age.py",
    ]
    results = [
        f"outputs/final_analyses_{run_id}/final_analyses_results.json",
        f"outputs/final_analyses_{run_id}/uncertainty_bootstrap.json",
        f"outputs/final_analyses_{run_id}/descriptive_statistics.json",
        f"outputs/reduced_model_{run_id}/reduced_model_results.json",
        f"outputs/reduced_model_{run_id}/shap_ranking.json",
        "outputs/split_verification_report.json",
        "outputs/table1_baseline.csv", "outputs/table1_baseline.md",
    ]
    uncommitted = [
        f"notebooks/tuning_results_{run_id}/preprocessed_data.pkl",
        f"notebooks/tuning_results_{run_id}/catboost_best_model.pkl",
        f"notebooks/tuning_results_{run_id}/catboost_study.pkl",
        f"outputs/final_analyses_{run_id}/locked_threshold_calibration.pkl",
        f"outputs/reduced_model_{run_id}/reduced_model_bundle.pkl",
        f"outputs/reduced_model_{run_id}/shap_values_train_full.npy",
        "outputs/split_assignment_SEQN.csv",
    ]
    figures = sorted(glob.glob(str(OUT / "figures_R3" / "*.png")))
    data = sorted(glob.glob(str(REPO / "data" / "processed" / "*.parquet")))

    def table(paths, base=REPO):
        rows = []
        for rel in paths:
            p = Path(rel) if os.path.isabs(rel) else base / rel
            shown = str(p.relative_to(base)) if p.is_absolute() and base in p.parents else str(rel)
            if p.exists():
                rows.append(f"| `{shown}` | {p.stat().st_size} | `{sha256(p)}` |")
            else:
                rows.append(f"| `{shown}` | — | (absent) |")
        return rows

    # headline numbers, read from the artifacts
    fa = json.load(open(fin / "final_analyses_results.json"))
    un = json.load(open(fin / "uncertainty_bootstrap.json"))
    rm = json.load(open(red / "reduced_model_results.json"))
    ver = json.load(open(OUT / "split_verification_report.json"))
    prim = un["models"]["primary_22"]
    redu = un["models"]["reduced_top10_plus_indicators"]

    def ci(m, k):
        return f"{m[k]['point']:.3f} ({m[k]['ci95'][0]:.3f} to {m[k]['ci95'][1]:.3f})"

    L = [
        "# Release manifest — analysis of record",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Pinned run: `tuning_results_{run_id}`",
        f"- Git commit: `{commit}` (branch `{git('rev-parse', '--abbrev-ref', 'HEAD')}`)",
        f"- Working tree at generation: {'CLEAN' if not dirty else 'MODIFIED (see below)'}",
        f"- Python {sys.version.split()[0]} on {platform.platform()}",
        "",
        "This manifest binds the reported numbers to specific file contents. To",
        "confirm a clean clone matches the analysis of record, check out the commit",
        "above and compare SHA-256 values. The manifest documents the state at that",
        "commit and is itself committed immediately afterwards, so it lives one",
        "commit later than the SHA it names.",
        "",
        "## Headline results (held-out test set, single evaluation pass)",
        "",
        "| Model | AUC (raw scores) | Sensitivity | Specificity | PPV | NPV |",
        "|---|---|---|---|---|---|",
        f"| Primary, 22 features | {ci(prim,'auc')} | {ci(prim,'sensitivity')} | "
        f"{ci(prim,'specificity')} | {ci(prim,'ppv')} | {ci(prim,'npv')} |",
        f"| Reduced, 12 features | {ci(redu,'auc')} | {ci(redu,'sensitivity')} | "
        f"{ci(redu,'specificity')} | {ci(redu,'ppv')} | {ci(redu,'npv')} |",
        "",
        f"- Paired AUC difference, full minus reduced: "
        f"{un['paired_auc_difference_full_minus_reduced']['point']} "
        f"{un['paired_auc_difference_full_minus_reduced']['ci95']}",
        f"- Operating threshold (locked on validation before any test use): "
        f"{fa['analyses']['primary']['threshold_locked_on_validation']}",
        f"- Calibration, primary (test): {fa['analyses']['primary'].get('calibration_test')}",
        f"- Calibration, reduced (test): {rm['reduced'].get('calibration_test')}",
        f"- Split verification: {len(ver['checks'])} checks, "
        f"{'all passed' if all(v['pass'] for v in ver['checks'].values()) else 'FAILURES PRESENT'}",
        "",
        "## Pre-specified sensitivity analyses (test AUC)",
        "",
        "| Analysis | AUC |",
        "|---|---|",
    ]
    for key, res in fa["analyses"].items():
        if key == "primary":
            continue
        t = res.get("test", {})
        auc = (t.get("unweighted", {}).get("auc") if "unweighted" in t else t.get("auc"))
        L.append(f"| `{key}` | {auc} |")
    abc = fa["analyses"].get("quality_grades_ABC", {}).get(
        "paired_auc_difference_AB_minus_ABC")
    if abc:
        L += ["", f"- Paired A/B minus A/B/C gating difference: {abc['point']} "
                  f"{abc['ci95']} (interval covering zero indicates no detectable "
                  f"dependence on the quality criterion's strictness)"]

    L += ["", "## Analysis code (committed)", "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(tracked),
          "", "## Result files (committed)", "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(results),
          "", "## Model artifacts (not committed; too large for git)", "",
          "These are required to reproduce predictions without refitting. Hashes",
          "let a recipient confirm that a transferred copy is the one used here.",
          "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(uncommitted),
          "", "## Figures", "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(figures),
          "", "## Input data (hashed, not committed)", "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(data)]

    if dirty:
        L += ["", "## Uncommitted changes at generation time", "", "```", dirty, "```"]

    try:
        pkgs = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                              capture_output=True, text=True, timeout=180).stdout.strip()
    except Exception as e:
        pkgs = f"(pip freeze failed: {e})"
    L += ["", "## Environment", "", "```", pkgs, "```", ""]

    path = OUT / "RELEASE_MANIFEST.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"written: {path}")
    print(f"  run {run_id} | commit {commit[:8]} | "
          f"{'clean tree' if not dirty else 'MODIFIED tree'}")


if __name__ == "__main__":
    main()
