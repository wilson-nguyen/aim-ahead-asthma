# Explainable Machine Learning for Pediatric Asthma (NHANES)

A sensitivity-first, explainable machine-learning pipeline that identifies the clinical, environmental, and social factors associated with diagnosed pediatric asthma in U.S. national survey data (NHANES). First-author research project — **Best Poster Award, 2025 AIM-AHEAD Annual Meeting**; manuscript under peer review (revise-and-resubmit).

## Summary

- **Goal:** an interpretable, **sensitivity-first** classifier that identifies the factors most associated with a recorded diagnosis of asthma, rather than a deployable diagnostic tool.
- **Data:** NHANES cycles 2007–2008, 2009–2010, 2011–2012; children aged 6–17; **n = 6,567** analytic sample (asthma prevalence 18.7%). Outcome: `MCQ010` (ever told had asthma). Survey design weights (`WTMEC2YR`) incorporated.
- **Model:** gradient-boosted trees (**CatBoost**), tuned with **Optuna** (100 trials, 5-fold CV); 60/20/20 stratified train/validation/test split (seed 42); decision threshold tuned on validation to reach ≥ 80% sensitivity, then applied unchanged to the held-out test set.
- **Explainability:** **SHAP** values and permutation importance; a reduced **Top-10-feature model** derived from the SHAP ranking.
- **Headline result (test set, n = 1,314):** the full model reaches AUC **0.827** (95% CI 0.795–0.856) at the sensitivity-first operating point (sensitivity 0.78, specificity 0.72, NPV 0.93), and a **10-feature model performs comparably** (AUC 0.803, 95% CI 0.771–0.833), showing the top SHAP-ranked predictors capture nearly all of the signal. Predicted probabilities were isotonically calibrated (Brier 0.185 to 0.109); confidence intervals are 1,000-iteration bootstrap.
- **Cross-cycle data handling:** variables that NHANES renamed across 2007–2012 were harmonized into single cross-cycle variables, and survey-design/administrative variables (the interview weight and the masked variance pseudo-PSU/pseudo-stratum) were excluded from the predictor set; the examination weight (`WTMEC2YR`) was retained only as the survey weight.

## Pipeline

Run the pipeline in order:

1. `python download_nhanes.py` — fetch raw NHANES files
2. `notebooks/01_load_and_harmonize.ipynb` — load and merge survey cycles
3. `notebooks/02_recode.ipynb` — recode variables
4. `python notebooks/harmonize_cycles.py` — harmonize renamed cross-cycle variables and write `data/processed/02b_harmonized.parquet`
5. `notebooks/03_clean_and_filter.ipynb` — clean and define the analytic sample (children 6–17 with a valid `MCQ010` response)
6. `notebooks/04_model.ipynb` — feature engineering, Optuna tuning, train and evaluate CatBoost
7. `notebooks/05_top10_sensitivity.ipynb` — Top-10-feature sensitivity analysis + SHAP

Notebook 03 expects `data/processed/02b_harmonized.parquet`, which is created by `notebooks/harmonize_cycles.py` after notebook 02.

## Top 10 predictors (SHAP-ranked)

1. Wheezing/whistling in chest, past year (`RDQ070`)
2. Close relative had asthma (`MCQ300B`)
3. General health condition (`HUQ010`)
4. FEV1/FVC ratio — *engineered* (`fev1_fvc_ratio`)
5. Times received healthcare, past year (`HUQ050`)
6. Head cold or chest cold, past 30 days (`HSQ500`)
7. Family history × lung function — *engineered interaction* (`family_spirometry_interaction`)
8. Forced expiratory time (`SPXNFET`)
9. Crawl/walk/run/play limitations (`PFQ020`)
10. Airway obstruction indicator — *engineered* (`obstruction_indicator`)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash; PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

NHANES files are not tracked in this repository (the `data/` folder is git-ignored). Download the public data from the CDC:

https://wwwn.cdc.gov/nchs/nhanes/

Place raw files in `data/raw/`, or run `python download_nhanes.py`.

## Repository structure

- `download_nhanes.py` — fetch raw NHANES data
- `notebooks/` — the 01–05 analysis pipeline
- `data/` — raw and processed NHANES files (git-ignored)
- `outputs/` — figures, metrics, and model artifacts (git-ignored)

## Tech

Python · pandas · NumPy · scikit-learn · CatBoost · SHAP · imbalanced-learn · Optuna · Jupyter

## Citation

Nguyen, W., Micheals, K., & Alwesabi, Y. "Explainable Machine Learning to Identify Clinical, Environmental, and Social Factors Associated with Diagnosed Pediatric Asthma (NHANES 2007-2012)." (Under review, 2026.)
