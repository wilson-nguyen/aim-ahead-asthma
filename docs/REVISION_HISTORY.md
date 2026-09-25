# Revision history and analysis details

This page holds the detail that used to be on the front page: how the analysis behind the paper was specified, what was corrected during the revision, and the caveats that go with the reported numbers. The paper's Methods and online supplement are the primary description.

## Version of record

- Tag `v3.4-R3-final` (commit `e296ad4`), pinned run `tuning_results_20260831_103201`.
- `outputs/RELEASE_MANIFEST.md` ties the reported numbers to commit `433da14` and to SHA-256 hashes of the code, models and result files. It describes the repository as it was at that commit; some files have since moved (see the last section).
- Results from earlier runs, including the AUC of 0.827 in the originally submitted manuscript, are superseded.

## Data, outcome and model

- NHANES 2007-2008, 2009-2010 and 2011-2012; children aged 6 to 17; analytic sample of 6,567 (1,229 with reported asthma; weighted prevalence 18.8%). Outcome: `MCQ010`, a self- or proxy-reported physician diagnosis of asthma.
- Stratified 60/20/20 split, seed 42. The current run's assignments are anchored to survey sequence numbers (SEQN) in a committed record. Earlier runs reproduced the same ordered outcome and survey-weight arrays exactly, but their saved artifacts contain no participant identifiers, so their identity is established at the level of those arrays.
- CatBoost tuned with Optuna (100 trials, 5-fold cross-validation), with categorical-aware SMOTENC plus edited nearest neighbours inside the folds for class imbalance. CatBoost was kept for continuity with the originally submitted analysis rather than reselected on revision results; the balanced random forest comparator reached a validation AUC of 0.815 against CatBoost's 0.812.
- Model fitting is unweighted. Survey weights are used for descriptive estimates and for a weighted evaluation reported alongside.
- Isotonic calibration and the operating threshold were both chosen on the validation set (the first point reaching a sensitivity of at least 0.80) and locked before the test split was scored in this run.
- AUC comes from raw model scores, threshold metrics from calibrated scores, and calibration is assessed separately.

## Test-set evaluation

The test split (n = 1,314) is a reused internal holdout: it also produced the previously submitted results. In this revision it was evaluated once, as a versioned batch, after the specification was locked. It is not untouched data and not independent validation.

| Model | AUC (95% CI) | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| Full, 22 features | 0.779 (0.744-0.813) | 0.752 | 0.625 | 0.32 | 0.92 |
| Reduced, 12 features | 0.802 (0.769-0.834) | 0.793 | 0.672 | 0.36 | 0.93 |

The reduced model (the ten highest-ranked SHAP features plus the two protected spirometry-availability indicators) has the higher AUC in a paired bootstrap (difference 0.023, 95% CI 0.007 to 0.041) and a calibration slope closer to 1 (0.87 against 0.71).

A no-resampling sensitivity analysis, declared before the final evaluation batch, had a higher AUC than the primary model (0.818 against 0.779; paired difference 0.039, 95% CI 0.019 to 0.060; `outputs/final_analyses_20260831_103201/noresampling_contrast.json`). The combined SMOTENC and edited-nearest-neighbours step leaves 1,520 of 3,202 training controls (62% cases after resampling), and the contrast removes that step as a whole, so it cannot say which part is responsible. The resampling-based primary model was kept rather than switched after seeing the reused test split. Neither the full nor the reduced model met the combined targets of 0.80 sensitivity and 0.70 specificity as point estimates, although both sensitivity intervals include 0.80. The no-resampling variant met the sensitivity target (0.805) but not the specificity target (0.645). Those figures and its AUC of 0.818 are point estimates; the paired AUC difference above is the only result from this analysis with a confidence interval.

Confidence intervals are stratified bootstrap (2,000 resamples, seed 42) and are conditional on the fitted models and locked thresholds, so they do not include tuning, selection or calibration uncertainty. The survey-weighted versions resample participants with their weights and do not model the design's clustering. Because resampling is stratified by outcome, the PPV and NPV intervals are conditional on the observed test-set case mix.

Display rounding: `uncertainty_bootstrap.json` stores values to four decimals, and the release manifest rounds them again to three. For three interval bounds that second rounding is one unit off in the last digit. At full precision the full model's AUC upper bound is 0.812511 (so 0.813; the manifest shows 0.812), the reduced model's sensitivity upper bound is 0.841463 (0.841; manifest 0.842), and its PPV upper bound is 0.383486 (0.383; manifest 0.384).

## Top 10 predictors (mean absolute SHAP, training data only)

1. Wheezing in chest, past year (`RDQ070`)
2. Family history of asthma by lung function, engineered interaction (`family_spirometry_interaction`)
3. General health condition (`HUQ010`)
4. Close relative had asthma (`MCQ300B`)
5. Serum cotinine, log (`cotinine_log`)
6. Household size (`DMDHHSIZ`)
7. FEV1/FVC ratio, engineered (`fev1_fvc_ratio`)
8. Race and Hispanic origin (`RIDRETH1`)
9. Health insurance coverage (`HIQ011`)
10. Child born in the United States (`DMDBORN_US`)

These rankings are associations, not causes. Recent wheezing and lung function are downstream of, or proxies for, an existing diagnosis and should not be read as causal risk factors. No BMI-derived variable is in the top ten.

## Specification decisions

These were fixed before the final analysis. The full decision log is kept by the authors and is available on request.

- Excluded from every model: prior-diagnosis and treatment proxies, NHANES protocol and routing variables, age-restricted questionnaire items, and identifiers and design variables.
- Excluded from the primary model and returned only in a declared exploratory analysis: health-care utilization and usual-source-of-care variables, which index the opportunity to be diagnosed.
- Excluded on measurement validity: `URDNALLC`, a below-detection-limit comment flag whose detection limit varied within cycles.
- Spirometry quality: FEV1 and FVC enter only with NHANES quality attribute A or B, and measures from the same maneuvers require both attributes to be acceptable (our rule, not one NHANES prescribes). Values that fail become missing and are captured by two availability indicators kept through feature selection. 4,726 of the 6,567 children (72.0%) have a usable FEV1/FVC pair.
- BMI: the CDC 2000 BMI-for-age z-score is the only BMI-derived predictor, computed with half a month added to completed-month age as the CDC program documentation directs. Raw body weight remains a separately eligible variable (ranked 13th by SHAP) and is covered by the age-dependence sensitivity analysis.
- Kept on purpose: health insurance coverage and family interview language, as the social and structural factors the study is about.

## Known limitations of the implementation

Integer category codes are treated as numbers by the tree models (no declared `cat_features`). The cleaner's type inference and the correlation pruning are fitted on the full training set rather than within inner folds. The cohort-level missingness screen comes before the split. The tuning objective maximizes cross-validated sensitivity with a soft specificity penalty and does not itself enforce the 0.80 floor, which is applied when the threshold is chosen. The MLP comparator is unweighted. The parallel Optuna search is not bit-for-bit reproducible and is kept as a committed study object for the primary model only; the comparator studies stay local, and their cited validation metrics are committed in `weighted_validation_metrics.json`.

## Reproduction details

Every run script is pinned to `tuning_results_20260831_103201`. Any script that reads the test split first checks that the committed verification report covers that exact run, that every check passed, and that the run's saved artifacts match their recorded hashes; `run_final_analyses.py` refuses to evaluate otherwise.

Several scripts read the test split, but each model has one evaluation that decides anything (thresholds and calibration are locked on validation first). Everything else recomputes descriptive results from fixed predictions.

Retraining from the raw data:

```
python download_nhanes.py
jupyter execute notebooks/01_load_and_harmonize.ipynb
jupyter execute notebooks/02_recode.ipynb
python notebooks/harmonize_cycles.py
jupyter execute notebooks/03_clean_and_filter.ipynb
jupyter execute notebooks/04_model.ipynb      # the tuning step takes hours
```

This creates a new `tuning_results_*` directory, which is a new analysis. To evaluate it, run `verify_split_reconstruction.py --run <new-id> --production` (without `--production`, the report goes to a `.local.json` file that the check ignores), then the run scripts with the pin updated, or `run_final_analyses.py --run <new-id>`.

## Corrections made during the revision

The third revision followed a full audit of the pipeline against the NHANES codebooks. It corrected missing spirometry that had been coded as "no obstruction" (718 children); blanket recoding of nonresponse codes that erased about 3,900 valid values and real measurements; adult BMI thresholds applied to children; redundant BMI forms; Methods text that did not match the code (imputation, scaling, and correlation pruning that had not been run); spirometry quality grades that had not been screened; a below-detection-limit flag treated as an exposure concentration; and an error in the CDC BMI age offset. Tightening the specification (diagnostic-opportunity exclusions, categorical-aware resampling, a locked threshold) lowered the full model's AUC from 0.827 to 0.779, as reported in the response to reviewers.

## Files moved after the tagged release

To make the front page easier to follow, some files were moved after `v3.4-R3-final`. Nothing was deleted, and the tag keeps the original layout.

- `archive/notebook_patches/`: fifteen one-off scripts that edited the notebooks during the revision (all already applied)
- `archive/notebooks/05_top10_sensitivity.ipynb`: the original reduced-model notebook, replaced by `run_reduced_model_and_figures.py` and blocked by a guard cell
- `archive/freeze_and_manifest.py`: the utility that took a snapshot of the pipeline before the corrections
- `outputs/superseded/`: interim results from the 24, 26 and 28 August 2026 runs
