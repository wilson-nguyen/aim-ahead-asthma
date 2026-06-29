# -*- coding: utf-8 -*-
"""
Harmonize NHANES 2007-2012 (cycles E/F/G) variables that NHANES renamed,
discontinued, or restricted across cycles, and remove survey-design/admin
variables that should never be model features.

Background (verified against the NHANES codebooks + the data):
  * In 2011-2012 (G) NHANES RENAMED three variables, so they do not populate
    under their old names in the pooled file:
        DMDBORN2  -> DMDBORN4   (country of birth)
        DMDHRBR2  -> DMDHRBR4   (HH ref person country of birth)
        RIDAGEEX  -> RIDEXAGM   (age in months at exam, 0-19 y)
  * In 2011-2012 (G) these were DISCONTINUED / RESTRICTED (no replacement for
    6-17 y), so they are missing for the whole G cycle:
        BMXTRI, BMXSUB (skinfolds, replaced by sagittal abdominal diameter)
        DMDSCHOL (now-attending-school item dropped)
        RIDAGEMN (restricted to 0-24 months; redundant with RIDAGEYR here)
  * MCQ082 / MCQ086 (celiac) were introduced in 2009-2010, so they are missing
    for the whole 2007-2008 (E) cycle.
  * The modeling pipeline used `feature_columns = every column except MCQ010 and
    WTMEC2YR`, which swept three survey-design/admin variables in as predictors:
        WTINT2YR, SDMVPSU, SDMVSTRA   (plus SEQN if present)

This script reads 02_recoded.parquet (which still contains BOTH the old and the
new names), builds harmonized columns with full E/F/G coverage, and writes
02b_harmonized.parquet.  It does NOT overwrite any existing file.
"""
import numpy as np, pandas as pd, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(REPO, "data", "processed", "02_recoded.parquet")
OUT  = os.path.join(REPO, "data", "processed", "02b_harmonized.parquet")

df = pd.read_parquet(SRC)
cyc = "NHANES_CYCLE"

def us_born(old, new):
    """1 = born in 50 US states/DC, 0 = born elsewhere, NaN = refused/DK/missing.
    Old (E/F) code 1=US; 2/3/4/5=elsewhere.  New (G) code 1=US; 2=elsewhere.
    Refused/DK (7/9/77/99) -> NaN in both."""
    out = np.full(len(df), np.nan)
    if old in df:
        out = np.where(df[old].isin([1]), 1.0, out)
        out = np.where(df[old].isin([2, 3, 4, 5]), 0.0, out)
    if new in df:
        out = np.where(df[new].isin([1]), 1.0, out)
        out = np.where(df[new].isin([2]), 0.0, out)
    return out

# --- 1. Harmonized country-of-birth indicators (replace renamed pairs) ---
df["DMDBORN_US"]  = us_born("DMDBORN2", "DMDBORN4")     # SP born in US
df["DMDHRBR_US"]  = us_born("DMDHRBR2", "DMDHRBR4")     # HH ref person born in US

# --- 2. Harmonized age-in-months at exam (E/F: RIDAGEEX ; G: RIDEXAGM) ---
ex = df["RIDAGEEX"] if "RIDAGEEX" in df else pd.Series(np.nan, index=df.index)
if "RIDEXAGM" in df:
    ex = ex.fillna(df["RIDEXAGM"])
df["RIDAGEEX_H"] = ex

# --- 3. Columns to drop from the candidate predictor set ---
REPLACED = ["DMDBORN2", "DMDBORN4", "DMDHRBR2", "DMDHRBR4",
            "RIDAGEEX", "RIDEXAGM", "RIDEXAGY"]          # superseded by harmonized cols
NO_CROSSCYCLE = ["RIDAGEMN", "DMDSCHOL", "BMXTRI", "BMXSUB"]  # no G coverage (redundant/discontinued)
CELIAC = ["MCQ082", "MCQ086"]                            # no E coverage (optional to drop)
DESIGN = ["WTINT2YR", "SDMVPSU", "SDMVSTRA", "SEQN"]     # survey-design/admin - never features

DROP_CELIAC = True   # set False to keep celiac items (accept E-cycle structured missingness)

# --- verification: per-cycle coverage before/after ---
g = df.groupby(cyc)
def cov(c):
    return {k: round(g.get_group(k)[c].notna().mean(), 2) for k in sorted(g.groups)}

print("Coverage AFTER harmonization (should be ~full across E/F/G):")
for c in ["DMDBORN_US", "DMDHRBR_US", "RIDAGEEX_H"]:
    print(f"  {c:12s} {cov(c)}")
print("\nOld renamed columns (now superseded):")
for c in ["DMDBORN2","DMDBORN4","DMDHRBR2","DMDHRBR4","RIDAGEEX","RIDEXAGM"]:
    if c in df: print(f"  {c:12s} {cov(c)}")

df.to_parquet(OUT)
print(f"\nWrote {OUT}  ({df.shape[0]} rows x {df.shape[1]} cols)")

# --- emit the corrected candidate-feature list for notebook 03/04 ---
drop_all = set(REPLACED + NO_CROSSCYCLE + DESIGN + (CELIAC if DROP_CELIAC else []))
print("\n=== Apply in 03_clean_and_filter (candidate columns) ===")
print("ADD harmonized:   DMDBORN_US, DMDHRBR_US, RIDAGEEX_H")
print("DROP superseded:  " + ", ".join(REPLACED))
print("DROP no-G/redund: " + ", ".join(NO_CROSSCYCLE))
print("DROP celiac (opt):" + ", ".join(CELIAC) + f"   (DROP_CELIAC={DROP_CELIAC})")
print("\n=== Apply in 04_model (feature_columns) ===")
print("Exclude target, weight, AND survey-design vars, e.g.:")
print("  EXCLUDE = ['MCQ010','WTMEC2YR','WTINT2YR','SDMVPSU','SDMVSTRA','SEQN','NHANES_CYCLE']")
print("  feature_columns = [c for c in df.columns if c not in EXCLUDE]")
