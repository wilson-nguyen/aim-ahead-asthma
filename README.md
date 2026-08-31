# Explainable Machine Learning for Pediatric Asthma (NHANES)

An explainable machine-learning analysis identifying the clinical, environmental, and social factors associated with **diagnosed** pediatric asthma in U.S. national survey data (NHANES). First-author research project (Best Poster Award, 2025 AIM-AHEAD Annual Meeting); manuscript under peer review (Ms. 26-02-0197R2, *Annals of Allergy, Asthma & Immunology*).

> **Analysis of record:** run `tuning_results_20260831_103201`. Every number below comes from committed result files in `outputs/`. `outputs/RELEASE_MANIFEST.md` binds those numbers to a git commit and SHA-256 hashes. Results from earlier runs, including the AUC 0.827 figure in the originally submitted manuscript, are **superseded** — see "Revision history" below.

## Summary

- **Goal:** an interpretable classifier for **cross-sectional classification of self- or proxy-reported, physician-diagnosed asthma at the time of assessment**. This is not incident prediction, not detection of undiagnosed asthma, and not a deployable diagnostic tool.
- **Data:** NHANES 2007-2008, 2009-2010, 2011-2012; children aged 6-17; **n = 6,567** analytic sample (1,229 with reported asthma; weighted prevalence 18.8%). Outcome: `MCQ010`. Split 60/20/20 stratified (seed 42), verified participant-level identical across every version of this work.
- **Model:** CatBoost, tuned with Optuna (100 trials, 5-fold CV), class imbalance handled with categorical-aware SMOTENC + edited nearest neighbors inside folds. Model fitting is **unweighted**; survey weights are used for descriptive estimates and reported alongside as weighted evaluation.
- **Operating point:** isotonic calibration and threshold both selected on the **validation** set (first point reaching sensitivity ≥ 0.80) and frozen before any test-set evaluation.
- **Reporting:** discrimination (AUC) from **raw model scores**; threshold metrics from calibrated scores; calibration assessed separately.

### Held-out test results (single evaluation pass, n = 1,314)

| Model | AUC (95% CI) | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| Primary (22 features) | 0.779 (0.744-0.813) | 0.752 | 0.625 | 0.32 | 0.92 |
| Reduced (12 features) | 0.802 (0.769-0.834) | 0.793 | 0.672 | 0.36 | 0.93 |

The reduced model — the ten highest-SHAP features plus the two protected spirometry-availability indicators — has the higher AUC in a paired bootstrap (difference 0.023, 95% CI 0.007 to 0.041) and better calibration (slope 0.87 vs 0.71) while using half as many variables. **Neither model meets the originally stated 0.80 sensitivity / 0.70 specificity targets as point estimates on held-out data**; both sensitivity intervals include 0.80. This is reported plainly rather than tuned away.

CIs are stratified bootstrap, 2,000 resamples, seed 42.

## Top 10 predictors (mean |SHAP|, training data only)

1. Wheezing in chest, past year (`RDQ070`)
2. Family history × lung function, *engineered interaction* (`family_spirometry_interaction`)
3. General health condition (`HUQ010`)
4. Close relative had asthma (`MCQ300B`)
5. Serum cotinine, log (`cotinine_log`)
6. Household size (`DMDHHSIZ`)
7. FEV1/FVC ratio, *engineered* (`fev1_fvc_ratio`)
8. Race/Hispanic origin (`RIDRETH1`)
9. Health insurance coverage (`HIQ011`)
10. Child born in the US (`DMDBORN_US`)

*These rankings are associative, not causal. Recent wheezing and lung function are downstream of, or proxies for, an existing diagnosis and must not be read as causal risk factors. No BMI-derived variable appears in the top ten.*

## Specification decisions

Pre-specified before the final analysis; full audit trail in the revision folder's exclusion log.

- **Excluded from every model:** prior-diagnosis and treatment proxies; NHANES protocol/routing variables; age-restricted questionnaire items; identifiers and design variables.
- **Excluded from the primary model** (returned only in a declared exploratory arm): healthcare-utilization and usual-source-of-care variables that index the *opportunity* to be diagnosed.
- **Excluded on measurement validity:** `URDNALLC` (below-detection-limit comment flag whose limit varied within cycles).
- **Spirometry quality gating:** FEV1 and FVC enter only with NHANES quality attribute **A or B**; measures from the same maneuvers require both attributes acceptable (our analytic specification, not an NHANES-prescribed rule). Gated values become missing and are absorbed by two availability indicators protected through feature selection. 4,726 of 6,567 children (72.0%) have a usable FEV1/FVC pair.
- **Adiposity:** CDC 2000 LMS BMI-for-age z-score only, computed with half a month added to completed-month age per CDC program documentation.
- **Retained deliberately:** health insurance coverage and family interview language, as the social and structural factors this study is about.

**Disclosed limitations:** integer category codes are treated as numeric by the tree models (no declared `cat_features`); the cleaner's type inference and correlation pruning are fitted on the full training set rather than within inner folds; the cohort-level missingness screen precedes the split; the tuning objective maximizes CV sensitivity with a soft specificity penalty and does not itself enforce the 0.80 floor, which is applied at threshold selection; the MLP comparator is unweighted; the parallel Optuna search is preserved as committed study objects rather than being bit-reproducible.

## Pipeline

```bash
python download_nhanes.py                                  # 1. fetch raw NHANES
jupyter execute notebooks/01_load_and_harmonize.ipynb      # 2. load and merge cycles
jupyter execute notebooks/02_recode.ipynb                  # 3. recode
python notebooks/harmonize_cycles.py                       # 4. -> 02b_harmonized.parquet
jupyter execute notebooks/03_clean_and_filter.ipynb        # 5. -> 03_cleaned.parquet
jupyter execute notebooks/04_model.ipynb                   # 6. tuning (hours)
python verify_split_reconstruction.py                      # 7. gate: 32 checks must pass
python run_final_analyses.py                               # 8. locked eval + 6 sensitivity arms
python run_reduced_model_and_figures.py                    # 9. SHAP, reduced model, figures
python run_uncertainty.py                                  # 10. bootstrap CIs
python generate_descriptives.py                            # 11. descriptive statistics
cd notebooks && python build_table1.py && cd ..            # 12. Table 1
python build_release_manifest.py                           # 13. hash-bound manifest
```

Order matters: the verifier gates the test evaluation, and the uncertainty script needs the reduced-model bundle. `notebooks/05_top10_sensitivity.ipynb` is **permanently superseded** by `run_reduced_model_and_figures.py` and is blocked by a guard cell; do not run it. Regression tests: `python -m pytest tests/ -q`.

## Repository structure

- `notebooks/asthma_pipeline.py` — single source of truth for cleaning, feature engineering, exclusion lists, quality gating, resampling
- `notebooks/pediatric_corrections.py` — CDC BMI-for-age, missingness-preserving helpers, correlation pruning
- `notebooks/01`-`04` — the analysis pipeline; `05` superseded
- `run_*.py` — locked evaluation, reduced model and figures, bootstrap uncertainty
- `verify_split_reconstruction.py` — replays Phase 3 and the seeded split; 32 checks including frozen historical runs
- `generate_descriptives.py`, `notebooks/build_table1.py` — every reported descriptive number
- `build_release_manifest.py` — binds commit SHA, run ID, and file hashes
- `tests/` — regression tests for sentinel handling, quality gating, and the CDC BMI age offset
- `data/`, `outputs/` — git-ignored except the committed result JSONs and tables

Model pickles, prediction arrays, and the SHAP matrix are too large for git; `outputs/RELEASE_MANIFEST.md` hashes them so a transferred copy can be verified.

## Revision history

The R3 revision followed a full audit of the pipeline against the NHANES codebooks. Corrected: missing spirometry silently coded as "no obstruction" (718 children); blanket sentinel recoding that erased ~3,900 valid values and real measurements; adult BMI thresholds applied to children; redundant BMI forms; Methods text that did not match the code (imputation, scaling, unexecuted pruning); spirometry quality grades discarded without screening; a below-detection-limit flag treated as an exposure concentration; a CDC BMI age-offset error. Specification tightening (diagnostic-opportunity exclusions, categorical-aware resampling, locked threshold) reduced reported performance from AUC 0.827 to 0.779 for the full model; this is stated plainly in the response to reviewers rather than tuned back.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell; Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt   # exact pins: requirements-lock.txt
```

NHANES files are not tracked (`data/` is git-ignored). Download from https://wwwn.cdc.gov/nchs/nhanes/ into `data/raw/`, or run `python download_nhanes.py`.

## Tech

Python · pandas · NumPy · scikit-learn · CatBoost · SHAP · imbalanced-learn · Optuna · Jupyter

## Citation

Nguyen, W., Micheals, K., & Alwesabi, Y. "Explainable Machine Learning to Identify Clinical, Environmental, and Social Factors Associated with Diagnosed Pediatric Asthma (NHANES 2007-2012)." (Under review, 2026.)
