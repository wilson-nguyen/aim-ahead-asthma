"""
Pediatric corrections for the R3 revision.

Provides:
  1. cdc_bmi_z()        -- CDC BMI-for-age z-score and percentile (sex + exact age in months)
  2. safe_indicator()   -- threshold indicator that preserves NaN instead of coercing to 0
  3. safe_interaction() -- interaction term that preserves NaN instead of fillna(0)
  4. prune_correlated() -- the >0.90 correlation pruning the Methods claims

First use downloads the CDC LMS reference table and caches it in ../data/reference/.

Quick check:
    python pediatric_corrections.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

CDC_URL = "https://www.cdc.gov/growthcharts/data/zscore/bmiagerev.csv"
CACHE = Path(__file__).resolve().parent.parent / "data" / "reference" / "bmiagerev.csv"


def _load_lms() -> pd.DataFrame:
    """CDC BMI-for-age LMS parameters. Sex: 1=male, 2=female (matches NHANES RIAGENDR)."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CDC LMS reference -> {CACHE}")
        pd.read_csv(CDC_URL).to_csv(CACHE, index=False)
    lms = pd.read_csv(CACHE)
    lms.columns = [c.strip() for c in lms.columns]
    lms = lms[["Sex", "Agemos", "L", "M", "S"]]
    # CDC ships this file with a repeated header row partway through, which makes
    # every column parse as text. Coerce and drop the junk row.
    lms = lms.apply(pd.to_numeric, errors="coerce").dropna()
    return lms.sort_values(["Sex", "Agemos"])


def cdc_bmi_z(bmi, age_months, sex):
    """
    CDC BMI-for-age z-score and percentile.

    bmi        : array-like, kg/m^2
    age_months : array-like, EXACT age in months (use RIDAGEEX_H, not RIDAGEYR*12)
    sex        : array-like, 1=male, 2=female

    Returns (z, percentile). NaN inputs and out-of-range ages (<24 or >240 months)
    return NaN rather than a fabricated value.
    """
    from scipy.stats import norm

    bmi = pd.Series(np.asarray(bmi, dtype=float)).reset_index(drop=True)
    age = pd.Series(np.asarray(age_months, dtype=float)).reset_index(drop=True)
    sx = pd.Series(np.asarray(sex, dtype=float)).reset_index(drop=True)

    # Accept either NHANES coding (1=male, 2=female) or the pipeline's
    # binary-recoded form (NHANESCleaner maps `== 1`, giving 1=male, 0=female).
    observed = set(pd.unique(sx.dropna()))
    if observed and observed <= {0.0, 1.0}:
        sx = sx.map({1.0: 1.0, 0.0: 2.0})

    lms = _load_lms()
    z = pd.Series(np.nan, index=bmi.index)

    for s in (1.0, 2.0):
        ref = lms[lms.Sex == s]
        if ref.empty:
            continue
        m = (sx == s) & bmi.notna() & age.between(24, 240)
        if not m.any():
            continue
        # interpolate L, M, S at each child's exact age in months
        L = np.interp(age[m], ref.Agemos, ref.L)
        M = np.interp(age[m], ref.Agemos, ref.M)
        S = np.interp(age[m], ref.Agemos, ref.S)
        x = bmi[m].to_numpy()
        zz = np.where(np.abs(L) < 1e-7,
                      np.log(x / M) / S,
                      ((x / M) ** L - 1.0) / (L * S))
        z.loc[m] = zz

    return z.to_numpy(), norm.cdf(z.to_numpy()) * 100.0


def safe_indicator(series: pd.Series, threshold: float, direction: str = "lt") -> pd.Series:
    """
    Threshold indicator that PRESERVES missingness.

    The bug this replaces:
        (ratio < 0.8).astype(float)      # NaN < 0.8 -> False -> 0.0
    which silently records "not measured" as "no obstruction" and hides it
    from the imputer. Here missing stays NaN so imputation handles it.
    """
    out = (series < threshold) if direction == "lt" else (series > threshold)
    return out.astype(float).where(series.notna(), np.nan)


def safe_interaction(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Interaction that PRESERVES missingness.

    Replaces  a.fillna(0) * b.fillna(0), where 0 conflates "no exposure",
    "missing exposure", and "missing measurement" into one value.
    """
    return (a * b).where(a.notna() & b.notna(), np.nan)


def prune_correlated(df: pd.DataFrame, threshold: float = 0.90, protect=()):
    """
    Drop one of each pair of features with |r| > threshold (the step the
    Methods claims). Catches the BMXBMI / bmi_zscore affine duplicate (r = 1.0).

    Returns (pruned_df, dropped_list).
    """
    num = df.select_dtypes(include=[np.number])
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = []
    for col in upper.columns:
        if col in protect:
            continue
        partners = upper.index[upper[col] > threshold].tolist()
        partners = [p for p in partners if p not in dropped]
        if partners:
            dropped.append(col)
    return df.drop(columns=dropped), dropped


if __name__ == "__main__":
    # sanity checks
    z, p = cdc_bmi_z([16.0, 22.0, 30.0, np.nan], [120, 120, 120, 120], [1, 1, 1, 1])
    print("10-year-old boys, BMI 16 / 22 / 30 / missing")
    print("  z          :", np.round(z, 3))
    print("  percentile :", np.round(p, 1), " (expect ~25th, ~95th, ~99th, nan)")

    s = pd.Series([0.75, 0.85, np.nan])
    print("\nsafe_indicator(<0.8):", safe_indicator(s, 0.8).tolist(), "(expect [1.0, 0.0, nan])")
    print("safe_interaction   :", safe_interaction(pd.Series([1, 0, np.nan]), s).tolist())

    d = pd.DataFrame({"bmi": [20, 25, 30, 22.0]})
    d["bmi_z"] = (d.bmi - d.bmi.mean()) / d.bmi.std()   # the affine duplicate
    d["other"] = [1, 5, 2, 9.0]
    _, dropped = prune_correlated(d)
    print("\nprune_correlated dropped:", dropped, "(expect one of bmi / bmi_z)")
