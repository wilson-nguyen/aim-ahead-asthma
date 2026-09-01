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


def committed_sha256(relpath):
    """SHA-256 of the COMMITTED bytes (git blob content at HEAD), so the
    hash matches what a clean clone checks out regardless of the local
    line-ending configuration. [Fixed 31 Aug: the first manifest hashed
    working-tree CRLF bytes, which did not match git's stored LF bytes.]
    Returns None if the path is not tracked at HEAD."""
    try:
        r = subprocess.run(["git", "show", f"HEAD:{relpath}"], cwd=REPO,
                           capture_output=True, timeout=60)
        if r.returncode != 0:
            return None
        return hashlib.sha256(r.stdout).hexdigest()
    except Exception:
        return None


def main():
    run_id = "20260831_103201"  # [31 Aug] pinned to the analysis of record
    fin = OUT / f"final_analyses_{run_id}"
    red = OUT / f"reduced_model_{run_id}"

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", default="")
    tracked = [
        "download_nhanes.py",
        "notebooks/01_load_and_harmonize.ipynb", "notebooks/02_recode.ipynb",
        "notebooks/harmonize_cycles.py",
        "notebooks/asthma_pipeline.py", "notebooks/pediatric_corrections.py",
        "notebooks/build_table1.py", "notebooks/03_clean_and_filter.ipynb",
        "notebooks/04_model.ipynb", "verify_split_reconstruction.py",
        "run_final_analyses.py", "run_reduced_model_and_figures.py",
        "run_uncertainty.py", "generate_descriptives.py",
        "redraw_shap_figures.py", "patch11_r3_quality_gating.py",
        "patch12_nb04_header.py", "patch13_stale_notebook_text.py",
        "audit_cleaner_replacements.py", "export_historical_split_arrays.py",
        "compute_noresampling_contrast.py", "build_release_manifest.py",
        "tests/test_cleaner_sentinels.py", "tests/test_quality_gating.py",
        "tests/test_cdc_bmi_age.py",
        "README.md", "requirements.txt", "requirements-lock.txt",
    ]
    results = [
        f"outputs/final_analyses_{run_id}/final_analyses_results.json",
        f"outputs/final_analyses_{run_id}/uncertainty_bootstrap.json",
        f"outputs/final_analyses_{run_id}/descriptive_statistics.json",
        f"outputs/reduced_model_{run_id}/reduced_model_results.json",
        f"outputs/reduced_model_{run_id}/shap_ranking.json",
        f"outputs/final_analyses_{run_id}/noresampling_contrast.json",
        "outputs/split_verification_report.json",
        "outputs/cleaner_replacement_audit.json",
        "outputs/historical_split_arrays/provenance.json",
        "outputs/table1_baseline.csv", "outputs/table1_baseline.md",
    ]
    artifacts = [
        f"notebooks/tuning_results_{run_id}/preprocessed_data.pkl",
        f"notebooks/tuning_results_{run_id}/catboost_best_model.pkl",
        f"notebooks/tuning_results_{run_id}/catboost_study.pkl",
        f"outputs/final_analyses_{run_id}/locked_threshold_calibration.pkl",
        f"outputs/reduced_model_{run_id}/reduced_model_bundle.pkl",
        f"outputs/reduced_model_{run_id}/shap_values_train_full.npy",
        "outputs/split_assignment_SEQN.csv",
        "data/reference/bmiagerev.csv",
        *sorted(str(Path(p).relative_to(REPO).as_posix()) for p in
                glob.glob(str(OUT / "historical_split_arrays" / "*.npz"))),
    ]
    figures = sorted(glob.glob(str(OUT / "figures_R3" / "*.png")))
    # committed processed inputs only (01/02-stage intermediates are
    # regenerable from raw NHANES and deliberately not shipped)
    data = ["data/processed/02b_harmonized.parquet",
            "data/processed/03_cleaned.parquet"]

    def table(paths, base=REPO):
        """Tracked files: hash of committed bytes at HEAD (portable across
        line-ending configs). Untracked files: hash of working-tree bytes,
        marked as such. Paths shown with forward slashes."""
        rows = []
        for rel in paths:
            p = Path(rel) if os.path.isabs(rel) else base / rel
            shown = (p.relative_to(base).as_posix()
                     if p.is_absolute() and base in p.parents
                     else Path(rel).as_posix())
            if not p.exists():
                rows.append(f"| `{shown}` | — | (absent) |")
                continue
            h = committed_sha256(shown)
            if h is None:
                rows.append(f"| `{shown}` | {p.stat().st_size} | "
                            f"`{sha256(p)}` (working tree, untracked) |")
            else:
                rows.append(f"| `{shown}` | {p.stat().st_size} | `{h}` |")
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
        "This manifest binds the reported numbers to specific file contents. Hashes",
        "for tracked files are computed from the committed bytes at the named commit,",
        "so `sha256sum` on a clean clone reproduces them regardless of line-ending",
        "configuration. The manifest documents the state at that commit and is itself",
        "committed immediately afterwards, so it lives one commit later than the SHA",
        "it names.",
        "",
        "Test-set status: the test split is a historically reused internal holdout —",
        "it also produced previously submitted results — evaluated in this revision",
        "as a versioned batch after the specification was locked. 'Single evaluation",
        "pass' below refers to this run's locked pass, not to the split's history.",
        "",
        "## Headline results (test split, single locked evaluation pass in this run)",
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
    nrc_path = fin / "noresampling_contrast.json"
    if nrc_path.exists():
        nrc = json.load(open(nrc_path))["primary_minus_noresampling_auc"]
        L += [f"- Paired primary minus no-resampling AUC: {nrc['point']} "
              f"{nrc['ci95']} — the no-resampling variant outperforms the "
              f"primary (ENN removes roughly half the training controls; "
              f"3,202 to 1,520, 62% cases after resampling); the pre-declared "
              f"resampling-based primary is retained rather than switched "
              f"post hoc"]

    L += ["", "## Analysis code (committed)", "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(tracked),
          "", "## Result files (committed)", "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(results),
          "", "## Model artifacts and reference data (committed 31 Aug)", "",
          "Fitted pipelines, calibrators, the SHAP matrix, the split record,",
          "and the CDC LMS reference (~5 MB total) are committed so a clean",
          "clone reproduces predictions without refitting.",
          "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(artifacts),
          "", "## Figures", "",
          "| File | Bytes | SHA-256 |", "|---|---:|---|", *table(figures),
          "", "## Processed input data (committed 31 Aug)", "",
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
