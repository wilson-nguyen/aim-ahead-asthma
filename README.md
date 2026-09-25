# Explainable machine learning for pediatric asthma (NHANES 2007-2012)

Code and results for the paper "Explainable Machine Learning to Identify Clinical, Environmental, and Social Factors Associated with Diagnosed Pediatric Asthma," accepted for publication in *Annals of Allergy, Asthma & Immunology* (in press).

## The study

We used NHANES 2007-2012 to look at which clinical, physiologic, environmental and social factors distinguish U.S. children aged 6 to 17 with a reported physician diagnosis of asthma from those without one. The analytic sample has 6,567 children, 1,229 of them with reported asthma (weighted prevalence 18.8%). A gradient-boosted model (CatBoost) was trained on 22 predictors and interpreted with SHAP. A reduced model used the ten highest-ranked predictors plus two spirometry-availability indicators.

## Main results (test set, n = 1,314)

| Model | AUC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|
| Full, 22 features | 0.779 (0.744-0.813) | 0.752 | 0.625 |
| Reduced, 12 features | 0.802 (0.769-0.834) | 0.793 | 0.672 |

The highest-ranked predictors were recent wheezing, family history of asthma combined with lung function, general health, family history of asthma, serum cotinine, household size, the FEV1/FVC ratio, race and Hispanic origin, health insurance coverage, and birth in the United States.

## Version used for the paper

The paper's results come from the tagged release `v3.4-R3-final` (run `tuning_results_20260831_103201`). `outputs/RELEASE_MANIFEST.md` records the commit and the SHA-256 hashes of the code, models and result files for that version. Later commits change only documentation, code comments and file locations.

## Reproducing the results

Python 3.12. Exact package versions are in `requirements-lock.txt`.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt

python verify_split_reconstruction.py   # rebuilds the data split and checks it against the saved one
python audit_cleaner_replacements.py    # confirms the cleaning step changes no values
python -m pytest tests/ -q              # regression tests
python redraw_shap_figures.py           # redraws the figures from the saved models
```

These steps use the fitted models and results saved in the repository and do not retrain anything. Retraining from the raw NHANES files is possible (`download_nhanes.py`, then notebooks 01 to 04), but the hyperparameter search is not bit-for-bit reproducible, so it produces a new analysis rather than the published one. See `docs/REVISION_HISTORY.md`.

## Limitations

- The design is cross-sectional. The model identifies factors associated with an existing asthma diagnosis. It does not predict new cases and is not a diagnostic or screening tool.
- The evaluation is internal. The same test split was used when the paper was first submitted, so the results need external and prospective validation.
- The outcome is a reported diagnosis, which reflects access to care as well as disease.
- SHAP rankings describe what the model relies on, not causes.

## Repository layout

- `notebooks/`: data preparation and modeling (notebooks 01 to 04) and the shared pipeline code
- `run_final_analyses.py`, `run_reduced_model_and_figures.py`, `run_uncertainty.py`: evaluation, reduced model, figures and bootstrap intervals
- `verify_split_reconstruction.py`, `audit_cleaner_replacements.py`, `tests/`: checks
- `outputs/`: results, figures, Table 1 and the release manifest; `outputs/superseded/` holds interim runs from the revision
- `archive/`: one-off scripts used while revising the notebooks
- `docs/REVISION_HISTORY.md`: specification details, corrections made during the revision, and further caveats

## Citation

Micheals K, Nguyen W, Alwesabi Y. Explainable Machine Learning to Identify Clinical, Environmental, and Social Factors Associated with Diagnosed Pediatric Asthma. *Annals of Allergy, Asthma & Immunology*. In press, 2026.

## License

MIT
