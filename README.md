# Explainable Machine Learning for Pediatric Asthma Screening (NHANES)

A sensitivity-first, explainable machine-learning pipeline that screens for pediatric asthma from U.S. national survey data (NHANES), integrating clinical, environmental, and social predictors. First-author research project — **Best Poster Award, 2025 AIM-AHEAD Annual Meeting**; manuscript under peer review (revise-and-resubmit).

## Summary

- **Goal:** a screening classifier tuned for **sensitivity** (catching true asthma cases) while staying interpretable for clinical use.
- **Data:** NHANES cycles 2007–2008, 2009–2010, 2011–2012; children aged 6–17; **n = 6,567** analytic sample (asthma prevalence 18.7%). Outcome: `MCQ010` (ever told had asthma). Survey design weights (`WTMEC2YR`) incorporated.
- **Model:** gradient-boosted trees (**CatBoost**), tuned with **Optuna** (100 trials, 5-fold CV); 60/20/20 stratified train/validation/test split (seed 42); decision threshold tuned on validation to reach ≥ 80% sensitivity, then applied unchanged to the held-out test set.
- **Explainability:** **SHAP** values and permutation importance; a reduced **Top-10-feature model** derived from the SHAP ranking.
- **Headline result (test set, n = 1,314):** the full model reaches AUC **0.822** (95% CI 0.790–0.851) at 80% sensitivity, and a **10-feature model performs comparably** — AUC 0.811 (0.779–0.840) — showing the top SHAP-ranked predictors capture nearly all of the signal. Confidence intervals are 1,000-iteration bootstrap.

## Pipeline

Run the notebooks in order:

1. `download_nhanes.py` — fetch raw NHANES files
2. `notebooks/01_load_and_harmonize.ipynb` — load and merge survey cycles
3. `notebooks/02_recode.ipynb` — recode variables
4. `notebooks/03_clean_and_filter.ipynb` — clean and define the analytic sample (children 6–17 with a valid `MCQ010` response)
5. `notebooks/04_model.ipynb` — feature engineering, Optuna tuning, train and evaluate CatBoost
6. `notebooks/05_top10_sensitivity.ipynb` — Top-10-feature sensitivity analysis + SHAP

## Top 10 predictors (SHAP-ranked)

1. Wheezing/whistling in chest, past year (`RDQ070`)
2. Close relative had asthma (`MCQ300B`)
3. General health condition (`HUQ010`)
4. FEV1/FVC ratio — *engineered* (`fev1_fvc_ratio`)
5. Times received healthcare, past year (`HUQ050`)
6. Health now vs. one year ago (`HSQ500`)
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

Nguyen, W., Micheals, K., & Alwesabi, Y. "Explainable Machine Learning for Pediatric Asthma Diagnosis: Integrating Clinical, Environmental, and Social Predictors." (Under review, 2026.)
