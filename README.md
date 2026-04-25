# AIM-AHEAD Pediatric Asthma Classification

CatBoost/SHAP analysis of pediatric asthma classification using NHANES data.
Revised pipeline addressing reviewer feedback.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash on Windows
pip install -r requirements.txt
```

## Data

NHANES files are not tracked in this repository. Download from:
https://wwwn.cdc.gov/nchs/nhanes/

Place raw files in `data/raw/`.

## Structure

- `data/` — raw and processed NHANES files (gitignored)
- `notebooks/` — Jupyter analysis notebooks
- `src/` — reusable preprocessing and modeling functions
- `outputs/` — SHAP plots, model artifacts (gitignored)

## Author

Wilson Nguyen