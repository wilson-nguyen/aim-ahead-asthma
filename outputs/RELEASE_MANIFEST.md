# Release manifest — analysis of record

- Generated: 2026-08-31T14:29:11
- Pinned run: `tuning_results_20260831_103201`
- Git commit: `cc7e80937e8b5971bc05d9d9a78f2e50f9238710` (branch `master`)
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
| `download_nhanes.py` | 2171 | `7332a218210013ca52ad54498e90af76acd3999aa4ddcd2efa756e1766ac8e87` |
| `notebooks/01_load_and_harmonize.ipynb` | 9943 | `e0170701e2a6d4161e37a206a339ae724925850fdbae5778272b3dcb95e0aa05` |
| `notebooks/02_recode.ipynb` | 85037 | `b95f10b0f95a436fec565813b65a8a7fa3aa6f10138a8db682f7462d2b4d6663` |
| `notebooks/harmonize_cycles.py` | 4738 | `e3d8a41ab838fb503053bc1215a7aabaf2aab71981c438a7acce2e7e812b461d` |
| `notebooks/asthma_pipeline.py` | 24882 | `bc47ecaf46b6dd5da1d31bdf0011b6128a0b381c27d5ec948bf8df6c20b9553d` |
| `notebooks/pediatric_corrections.py` | 6328 | `e473434463d47073c95b0e40484ef860d4d882a4a88d252f8ab09a008d703520` |
| `notebooks/build_table1.py` | 9113 | `03cf7ee05b9e8a1a3b1a6296ad26dac419318e1f657d146d6a9898569cc0e3b3` |
| `notebooks/03_clean_and_filter.ipynb` | 11389 | `ba9690be760cb1d4676b9a8c27bec5c81517552d17573545c4c167303bc5d360` |
| `notebooks/04_model.ipynb` | 126420 | `922debe734fab8dd0b3cf65c353ea7fb8154a2df57668a986aaa6845113010f2` |
| `verify_split_reconstruction.py` | 16992 | `8478486fbba297b4ffbba55d91bcaba23ec41541df9564bfd1986d45245eff90` |
| `run_final_analyses.py` | 26005 | `ce5d52b56a8c29042ab31b9c743fa6a3b3eed63ddf95c1bc73eead54f045dc9d` |
| `run_reduced_model_and_figures.py` | 14589 | `bca7d156a315d3534e6b0388e59e8f16a24e9201dd3629355e94f245bf9403e0` |
| `run_uncertainty.py` | 6845 | `8ba845336a4a11447472af5f5587e7ea73bf205ffbed4766039ae56922fbce79` |
| `generate_descriptives.py` | 6950 | `fd812fe9b32e842cb0e43e2779ed4c38f1715aa95fb8171842886d657dbbc1df` |
| `redraw_shap_figures.py` | 7784 | `f3a530f60bd940276b3fbfece1586b5e06a39c011e6f8bf3cf94c99957ccb340` |
| `patch11_r3_quality_gating.py` | 5371 | `bd8be8eab51816e75c992aa019e3c2e9d72e68b569dec80b5331b5127b695060` |
| `audit_cleaner_replacements.py` | 4789 | `0933866222f9f42e7a4dbb6fefe793ae702948fd4226de55824a20f57d69c7de` |
| `export_historical_split_arrays.py` | 2211 | `13e55d9b4da93dc6c78662b2929bee470b69d529cab5f1edb3a37f3684fd3c12` |
| `build_release_manifest.py` | 11221 | `5f2fa99fbd667ca019b5003f727536c0ba0d386aa3a16b02c9204db13f771f16` |
| `tests/test_cleaner_sentinels.py` | 7023 | `90833ed6b0c3fdaf6dfd17e81685fe8c67c1481528f86907edf503bda7c9ad9a` |
| `tests/test_quality_gating.py` | 5383 | `a6d640586dae01a1eee03fdcf7491d5417e2f5bee4663f4fc2cf97e0eea35a8b` |
| `tests/test_cdc_bmi_age.py` | 3158 | `07afceb31423dc27fa7679dac5107ec4a9786d05cf2d61e0fd71569207211387` |
| `README.md` | 11312 | `72bb28e9c3b448eff8d73f0ebad78d56957bbc17abfe4630c061283c752e106e` |
| `requirements.txt` | 148 | `6381e344f1addcc74c0d4c15edc3bb0fbb8b54bbdec0e34c342d1825a4d58263` |
| `requirements-lock.txt` | 2930 | `5f0f4a38ebd99a8bd8a47e972ba928e34d2a5f20da4bc12341084aa5dc8ea46f` |

## Result files (committed)

| File | Bytes | SHA-256 |
|---|---:|---|
| `outputs/final_analyses_20260831_103201/final_analyses_results.json` | 16540 | `d590dc5bcdfd344ed988c22a5a6e1cf65c12c9eba2cde32a3dd3233569ab931b` |
| `outputs/final_analyses_20260831_103201/uncertainty_bootstrap.json` | 2496 | `6cd7d34159db468c5e731afeff2a24bf47f488d57760f17fa652e0d170b8ce45` |
| `outputs/final_analyses_20260831_103201/descriptive_statistics.json` | 1584 | `f54fb12902ff3820f0f4197b42b4ea05f05eb542d325c1f6b6530dffc97c3e88` |
| `outputs/reduced_model_20260831_103201/reduced_model_results.json` | 2131 | `cb987df88c0fbbb94ffee0043b05eaa963f415fcc19a1dddf1f6ec7c5cde0103` |
| `outputs/reduced_model_20260831_103201/shap_ranking.json` | 1663 | `c298b92185478969f46f1ccbd752ff1c35241083bcbfc18be85f929f3467533c` |
| `outputs/split_verification_report.json` | 4197 | `77c8f17714a58544a0ae319793a752b81289d8c4c146335af8b433c325b5e044` |
| `outputs/cleaner_replacement_audit.json` | 11402 | `8d7b7034806f837b94eccf990212bda837c4cf855e9c40f7950a3ec44c31ad17` |
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
| `outputs/split_assignment_SEQN.csv` | 184693 | `7f76b2425a555cb51754f4f1def5d8aadd0c8a5a9f816dedbba5063450fa9569` |
| `data/reference/bmiagerev.csv` | 72022 | `fcf2ddd1aa7b902620f6ddd6a10971b533f8f1c1653951d4a8bf18c8945df297` |
| `outputs/historical_split_arrays/20260426_180035.npz` | 44852 | `6eee95bcd7bc065ca2adb8a91945879d30e079f3db9c547c03bf92717573248f` |
| `outputs/historical_split_arrays/20260625_165708.npz` | 44852 | `6eee95bcd7bc065ca2adb8a91945879d30e079f3db9c547c03bf92717573248f` |

## Figures

| File | Bytes | SHA-256 |
|---|---:|---|
| `outputs/figures_R3/efigure_calibration.png` | 139823 | `8c1ceb1dc9c4294b833b81ec19a6def5abcb99fd3426aa21f73b557a32469c07` |
| `outputs/figures_R3/efigure_decision_curve.png` | 126533 | `98b67f1787ed9ad873c3cef046dfc488a8d889e9aff2df3cdc79572109d1ac13` |
| `outputs/figures_R3/figure_metrics.png` | 81084 | `0a9760e9433c54a471dd24a299df5466b99df7b47174b87769f72083d9c11986` |
| `outputs/figures_R3/figure_roc.png` | 145753 | `0ef4379efd0aa35160b2038bfbaaec0b17482bbb198b867a62a71e49b19624a9` |
| `outputs/figures_R3/figure_shap_ranking.png` | 199086 | `8c10a056e8db9caa44f76d699c7f5103386aac1b53c615abff61bc474dab9c03` |
| `outputs/figures_R3/figure_shap_summary.png` | 455150 | `56f38a92cf9e1f6ddbdd0e1de8d716ca53ffeecb4dfeaa30ca42317f9003b744` |

## Processed input data (committed 31 Aug)

| File | Bytes | SHA-256 |
|---|---:|---|
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
iniconfig==2.3.0
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
pluggy==1.6.0
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
pytest==9.1.1
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
