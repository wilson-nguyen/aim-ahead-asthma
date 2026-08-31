# Release manifest — analysis of record

- Generated: 2026-08-31T11:59:22
- Pinned run: `tuning_results_20260831_103201`
- Git commit: `47ea4f520d2c1e7a131f617af3631cac105a3c02` (branch `master`)
- Working tree at generation: CLEAN
- Python 3.12.13 on Windows-11-10.0.26200-SP0

This manifest binds the reported numbers to specific file contents. Hashes
for tracked files are computed from the committed bytes at the named commit,
so `sha256sum` on a clean clone reproduces them regardless of line-ending
configuration. The manifest documents the state at that commit and is itself
committed immediately afterwards, so it lives one commit later than the SHA
it names.

Test-set status: the test split is a historically reused internal holdout —
it also produced previously submitted results — evaluated in this revision
as a versioned batch after the specification was locked. 'Single evaluation
pass' below refers to this run's locked pass, not to the split's history.

## Headline results (test split, single locked evaluation pass in this run)

| Model | AUC (raw scores) | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| Primary, 22 features | 0.779 (0.744 to 0.812) | 0.752 (0.699 to 0.809) | 0.625 (0.597 to 0.654) | 0.316 (0.294 to 0.341) | 0.916 (0.899 to 0.934) |
| Reduced, 12 features | 0.802 (0.769 to 0.834) | 0.793 (0.744 to 0.842) | 0.672 (0.644 to 0.701) | 0.358 (0.334 to 0.384) | 0.934 (0.918 to 0.949) |

- Paired AUC difference, full minus reduced: -0.0232 [-0.0409, -0.0066]
- Operating threshold (locked on validation before any test use): {'threshold': 0.1134, 'val_sens': 0.8049, 'val_spec': 0.6311}
- Calibration, primary (test): {'brier': 0.1191, 'intercept': -0.3588, 'slope': 0.7128}
- Calibration, reduced (test): {'brier': 0.1127, 'intercept': -0.1211, 'slope': 0.87}
- Split verification: 32 checks, all passed

## Pre-specified sensitivity analyses (test AUC)

| Analysis | AUC |
|---|---|
| `no_resampling` | 0.8181 |
| `utilization_addback` | 0.7603 |
| `age_sex_matched` | 0.7944 |
| `age_dependent_removed` | 0.7782 |
| `no_usable_spirometry_subgroup` | 0.7935 |
| `no_usable_spirometry_excluded` | 0.7605 |
| `quality_grades_ABC` | 0.767 |

- Paired A/B minus A/B/C gating difference: 0.0117 [-0.0052, 0.0287] (interval covering zero indicates no detectable dependence on the quality criterion's strictness)

## Analysis code (committed)

| File | Bytes | SHA-256 |
|---|---:|---|
| `notebooks/asthma_pipeline.py` | 24135 | `3a9d12922ee805904526e80f8b69c198d50bdab9420826143369aeff109464a6` |
| `notebooks/pediatric_corrections.py` | 6328 | `e473434463d47073c95b0e40484ef860d4d882a4a88d252f8ab09a008d703520` |
| `notebooks/build_table1.py` | 9113 | `03cf7ee05b9e8a1a3b1a6296ad26dac419318e1f657d146d6a9898569cc0e3b3` |
| `notebooks/03_clean_and_filter.ipynb` | 11389 | `ba9690be760cb1d4676b9a8c27bec5c81517552d17573545c4c167303bc5d360` |
| `notebooks/04_model.ipynb` | 127469 | `9f767e6c42f65a4fed4a20159f135792471b8e98a4de821dfb3cac01ff7daea6` |
| `verify_split_reconstruction.py` | 16496 | `947253cdd935b1b2fbc51360e75c2cb601aa9d97229a2c33c2bd8b36216ccf97` |
| `run_final_analyses.py` | 24925 | `b53ec7b27f173a806e493092f915b014887838dd91e5aa7868d3a37cee72395a` |
| `run_reduced_model_and_figures.py` | 14534 | `0acbf6140f27f0e5d474486705ed52914dea19f604f89473726a05b54e7c99ee` |
| `run_uncertainty.py` | 6812 | `4ad2c9351ec4a2836a6736278208eeac341b951e6a2455e2ce2cd8eb6c4b4c7f` |
| `generate_descriptives.py` | 6835 | `0ad5f5d75cc4fbd2f9fcadbda05eab3ecfe2e33d1b305a00f1e0fca06869eeb9` |
| `redraw_shap_figures.py` | 2735 | `8a89bfa4a1c1ce5bde6447aa74689530e800c6043c327ff98a5e5c3201376783` |
| `patch11_r3_quality_gating.py` | 5371 | `bd8be8eab51816e75c992aa019e3c2e9d72e68b569dec80b5331b5127b695060` |
| `tests/test_cleaner_sentinels.py` | 5587 | `ab0dfd819395f6a71cf5f6e8cd327dff535b6bdd406e295acb62852de65e14fc` |
| `tests/test_quality_gating.py` | 5383 | `a6d640586dae01a1eee03fdcf7491d5417e2f5bee4663f4fc2cf97e0eea35a8b` |
| `tests/test_cdc_bmi_age.py` | 3158 | `07afceb31423dc27fa7679dac5107ec4a9786d05cf2d61e0fd71569207211387` |

## Result files (committed)

| File | Bytes | SHA-256 |
|---|---:|---|
| `outputs/final_analyses_20260831_103201/final_analyses_results.json` | 16540 | `d590dc5bcdfd344ed988c22a5a6e1cf65c12c9eba2cde32a3dd3233569ab931b` |
| `outputs/final_analyses_20260831_103201/uncertainty_bootstrap.json` | 2496 | `6cd7d34159db468c5e731afeff2a24bf47f488d57760f17fa652e0d170b8ce45` |
| `outputs/final_analyses_20260831_103201/descriptive_statistics.json` | 1584 | `f54fb12902ff3820f0f4197b42b4ea05f05eb542d325c1f6b6530dffc97c3e88` |
| `outputs/reduced_model_20260831_103201/reduced_model_results.json` | 2131 | `cb987df88c0fbbb94ffee0043b05eaa963f415fcc19a1dddf1f6ec7c5cde0103` |
| `outputs/reduced_model_20260831_103201/shap_ranking.json` | 1663 | `c298b92185478969f46f1ccbd752ff1c35241083bcbfc18be85f929f3467533c` |
| `outputs/split_verification_report.json` | 4125 | `154ce6422e4334d097a204e32e928f560fbd987b4c099056a86b6b0eee6ebb22` |
| `outputs/cleaner_replacement_audit.json` | 11767 | `fe57b39435be4a84d1b9fd494e20365a29ce90db81cc11737df239da94ec7e56` |
| `outputs/table1_baseline.csv` | 646 | `c7beef0b3f6ad2676e154eeeac071e2da67190611b4be1570828aefcdb59a40a` |
| `outputs/table1_baseline.md` | 1844 | `09462b16d70b1afa3f7fbd5f31e18ce55a85aea0c33000056f21accc04eb110d` |

## Model artifacts and reference data (committed 31 Aug)

Fitted pipelines, calibrators, the SHAP matrix, the split record,
and the CDC LMS reference (~5 MB total) are committed so a clean
clone reproduces predictions without refitting.

| File | Bytes | SHA-256 |
|---|---:|---|
| `notebooks/tuning_results_20260831_103201/preprocessed_data.pkl` | 4059933 | `2285a676c8e2ea9c7c7e81b1da1cf5d4256b853250e874ed906fcff57e9f5f41` |
| `notebooks/tuning_results_20260831_103201/catboost_best_model.pkl` | 62693 | `0250dd73d681fcacbf40ddcb43690251fa99cd8e4e4abb0b22251372c8e42d66` |
| `notebooks/tuning_results_20260831_103201/catboost_study.pkl` | 60223 | `20f60a0eb44fb2c6db003758a8f11be3974c74680a395c5cf6c1d60a4a9126ad` |
| `outputs/final_analyses_20260831_103201/locked_threshold_calibration.pkl` | 1215 | `8bdb4e21f7a70a02547bd7f517e6397a5c046b0edcc5d894ac3e4486bbbc5bde` |
| `outputs/reduced_model_20260831_103201/reduced_model_bundle.pkl` | 59285 | `992866815546bd64969504531fd0bd53b24a138ff2cb8360416ec0e6a27cf690` |
| `outputs/reduced_model_20260831_103201/shap_values_train_full.npy` | 693392 | `922a223a224e05eb16742303c8a37f56648f6316acdf0dbf7da2041085ff4821` |
| `outputs/split_assignment_SEQN.csv` | 191261 | `7f76b2425a555cb51754f4f1def5d8aadd0c8a5a9f816dedbba5063450fa9569` |
| `data/reference/bmiagerev.csv` | 72022 | `fcf2ddd1aa7b902620f6ddd6a10971b533f8f1c1653951d4a8bf18c8945df297` |

## Figures

| File | Bytes | SHA-256 |
|---|---:|---|
| `outputs/figures_R3/efigure_calibration.png` | 133198 | `e5ccf9efa8f0ee886aa88ce701f4dc0894b621191cc66061ab1be101aa0a618d` |
| `outputs/figures_R3/efigure_decision_curve.png` | 120347 | `f5e70c49e997e97e891f5d43f8361af9291d6381cf8794fb5a1843364ea6ae08` |
| `outputs/figures_R3/figure_metrics.png` | 71043 | `baa9fba4d91c3259928eb84f09689322fb7e2655dc7ea022043105af8a8f73bd` |
| `outputs/figures_R3/figure_roc.png` | 137594 | `d090856afe0913f41f5277257c4c15ea5fbfd7d0b43165548a0fd87dfdf47f6b` |
| `outputs/figures_R3/figure_shap_ranking.png` | 199086 | `8c10a056e8db9caa44f76d699c7f5103386aac1b53c615abff61bc474dab9c03` |
| `outputs/figures_R3/figure_shap_summary.png` | 456992 | `4b1c80a09b23096e4120c59185d8d0fb4df23395851ad6eab9853ce7cc0c32c0` |

## Processed input data (committed 31 Aug)

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/processed/01_combined_nhanes.parquet` | 4179785 | `2bcfc9c7a8bcb7fa92c0c69214e10957f71dca64e02cd1d917a15808bf091d3c` (working tree, untracked) |
| `data/processed/02_recoded.parquet` | 4071526 | `7054a6fb07fd0848912ef849f17f136c1ec390682f7ed586a1c4d022f5e63baa` (working tree, untracked) |
| `data/processed/02b_harmonized.parquet` | 4121522 | `04b65673360c12a16327c62a2df6d84e87cf0f427c77c293ebcd7f8b3c3b5524` |
| `data/processed/03_cleaned.parquet` | 538075 | `8a6c17b36c26486323b92270d0d1b1a93dcb8c45f489a49dc561904c29fcd84a` |

## Environment

```
alembic==1.18.4
annotated-doc==0.0.4
annotated-types==0.7.0
anthropic==0.109.2
anyio==4.13.0
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
asttokens==3.0.1
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.14.3
bleach==6.3.0
catboost==1.2.10
certifi==2026.4.22
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
cloudpickle==3.1.2
colorama==0.4.6
colorlog==6.10.1
comm==0.2.3
contourpy==1.3.3
cycler==0.12.1
debugpy==1.8.20
decorator==5.2.1
defusedxml==0.7.1
distro==1.9.0
docstring_parser==0.18.0
et_xmlfile==2.0.0
executing==2.2.1
fastjsonschema==2.21.2
filelock==3.29.4
fonttools==4.62.1
fqdn==1.5.1
fsspec==2026.4.0
graphviz==0.21
greenlet==3.4.0
h11==0.16.0
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.19.0
idna==3.13
imbalanced-learn==0.14.1
ipykernel==7.2.0
ipython==9.13.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.8
isoduration==20.11.0
jedi==0.19.2
Jinja2==3.1.6
jiter==0.15.0
joblib==1.5.3
json5==0.14.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter==1.1.1
jupyter-console==6.6.3
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_client==8.8.0
jupyter_core==5.9.1
jupyter_server==2.17.0
jupyter_server_terminals==0.5.4
jupyterlab==4.5.6
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_widgets==3.0.16
kiwisolver==1.5.0
lark==1.3.1
lightgbm==4.6.0
llvmlite==0.47.0
Mako==1.3.11
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
matplotlib-inline==0.2.1
mdurl==0.1.2
mistune==3.2.0
narwhals==2.20.0
nbclient==0.10.4
nbconvert==7.17.1
nbformat==5.10.4
nest-asyncio==1.6.0
notebook==7.5.5
notebook_shim==0.2.4
numba==0.65.1
numpy==2.4.4
openpyxl==3.1.5
optuna==4.8.0
packaging==26.2
pandas==2.3.3
pandocfilters==1.5.1
parso==0.8.6
patsy==1.0.2
pillow==12.2.0
platformdirs==4.9.6
plotly==6.7.0
prometheus_client==0.25.0
prompt_toolkit==3.0.52
psutil==7.2.2
pure_eval==0.2.3
pyarrow==24.0.0
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.20.0
pyparsing==3.3.2
pyreadstat==1.3.4
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-json-logger==4.1.0
pytz==2026.1.post1
pywinpty==3.0.3
PyYAML==6.0.3
pyzmq==27.1.0
referencing==0.37.0
requests==2.33.1
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
rpds-py==0.30.0
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
Send2Trash==2.1.0
setuptools==82.0.1
shap==0.51.0
shellingham==1.5.4
six==1.17.0
sklearn-compat==0.1.5
slicer==0.0.8
sniffio==1.3.1
soupsieve==2.8.3
SQLAlchemy==2.0.49
stack-data==0.6.3
statsmodels==0.14.6
terminado==0.18.1
threadpoolctl==3.6.0
tinycss2==1.4.0
tornado==6.5.5
tqdm==4.67.3
traitlets==5.14.3
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2026.2
uri-template==1.3.0
urllib3==2.6.3
wcwidth==0.6.0
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
widgetsnbextension==4.0.15
xgboost==3.2.0
```
