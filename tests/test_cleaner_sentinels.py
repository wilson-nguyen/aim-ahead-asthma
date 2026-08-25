"""
Regression tests for the codebook-verified NHANESCleaner (rewritten 25 Aug 2026).

Guards against the two defects found in review:
  1. Blanket sentinel-scrubbing of continuous variables (erased 99 real
     values: ages of exactly 99 months, FEV of 999 mL, weight 99 kg, ...).
  2. Blanket {7,9,77} scrubbing of categoricals where those are valid codes
     (grade levels, income brackets, household size, curve counts —
     ~3,900 real values).

Run from the repo root:
    python -m pytest tests/test_cleaner_sentinels.py -q
or:
    python tests/test_cleaner_sentinels.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "notebooks"))

from asthma_pipeline import NHANESCleaner  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "processed", "03_cleaned.parquet")


def _clean(df):
    cl = NHANESCleaner().fit(df)
    return cl, cl.transform(df)


# ---------------------------------------------------------------------------
# synthetic-frame unit tests
# ---------------------------------------------------------------------------

def _synth(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        # continuous with legitimate sentinel-looking values
        "RIDAGEEX_H": np.r_[np.full(5, 99.0), rng.uniform(72, 215, n - 5)],
        "SPXNFEV1": np.r_[np.full(3, 999.0), rng.uniform(600, 4000, n - 3)],
        "BMXWT": np.r_[np.full(2, 99.0), rng.uniform(15, 120, n - 2)],
        # income category: 1-15 valid incl. 7 and 9; 77/99 refused/DK
        "INDFMIN2": rng.choice([1, 2, 3, 7, 9, 14, 15, 77, 99], n,
                               p=[.1, .1, .1, .2, .2, .1, .1, .05, .05]),
        # household size: 1-7 valid, 7 = "7 or more", no refused codes
        "DMDHHSIZ": rng.choice([1, 2, 3, 4, 5, 6, 7], n),
        # small categorical 1-5 with rare refused/DK 7/9
        "HUQ010": rng.choice([1, 2, 3, 4, 5, 7, 9], n,
                             p=[.3, .3, .2, .1, .08, .01, .01]),
        # binary 1/2
        "MCQ300B": rng.choice([1.0, 2.0], n),
    })


def test_continuous_values_preserved():
    df = _synth()
    _, out = _clean(df)
    assert (out["RIDAGEEX_H"] == 99.0).sum() == 5, "ages of 99 months erased"
    assert (out["SPXNFEV1"] == 999.0).sum() == 3, "FEV of 999 mL erased"
    assert (out["BMXWT"] == 99.0).sum() == 2, "weight of 99 kg erased"


def test_income_codes_7_and_9_valid_77_99_sentinel():
    df = _synth()
    _, out = _clean(df)
    assert (out["INDFMIN2"] == 7).sum() > 0, "income bracket 7 erased"
    assert (out["INDFMIN2"] == 9).sum() > 0, "income bracket 9 erased"
    assert not out["INDFMIN2"].isin([77, 99]).any(), "77/99 not scrubbed"
    assert out["INDFMIN2"].isna().sum() == df["INDFMIN2"].isin([77, 99]).sum()


def test_household_size_topcode_7_survives():
    df = _synth()
    _, out = _clean(df)
    assert (out["DMDHHSIZ"] == 7).sum() == (df["DMDHHSIZ"] == 7).sum()


def test_small_categorical_rare_7_9_scrubbed():
    df = _synth()
    _, out = _clean(df)
    assert not out["HUQ010"].isin([7, 9]).any()


def test_binary_recode_preserves_missingness():
    df = _synth()
    df.loc[:4, "MCQ300B"] = np.nan
    _, out = _clean(df)
    assert out["MCQ300B"].isna().sum() == 5
    assert set(out["MCQ300B"].dropna().unique()) == {0.0, 1.0}


def test_ambiguous_unlisted_variable_raises():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        # unlisted categorical whose "7" is 15% of values -> must raise
        "FAKE_TOPCODED": rng.choice([1, 2, 3, 4, 5, 6, 7], 300,
                                    p=[.14, .14, .14, .14, .14, .15, .15]),
    })
    try:
        NHANESCleaner().fit(df)
    except ValueError as e:
        assert "FAKE_TOPCODED" in str(e)
    else:
        raise AssertionError("ambiguous variable did not raise")


# ---------------------------------------------------------------------------
# real-data regression tests (skipped if the parquet is absent)
# ---------------------------------------------------------------------------

def test_real_data_38_children_at_99_months_survive():
    if not os.path.exists(DATA):
        return
    df = pd.read_parquet(DATA)
    an = df[df.WTMEC2YR > 0]
    X = an.drop(columns=[c for c in ("MCQ010", "WTMEC2YR") if c in an.columns])
    _, out = _clean(X)
    assert (out["RIDAGEEX_H"] == 99).sum() == 38, \
        "the 38 children aged exactly 99 months must survive cleaning"
    assert (out["SPXNFEV1"] == 999).sum() == 1
    assert (out["DMDEDUC3"].isin([7, 9])).sum() >= 900, \
        "7th/9th-grade codes must survive"
    assert (out["INDFMIN2"].isin([7, 9])).sum() >= 900, \
        "income brackets 7/9 must survive"
    assert (out["DMDHHSIZ"] == 7).sum() >= 800, \
        "household size '7 or more' must survive"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
