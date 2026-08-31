"""
Regression tests for spirometry quality gating [2026-08-27 KM ruling].

Primary analysis: best-test FEV1/FVC usable only with quality grade A or B
(SPXNQFV1/SPXNQFVC); pre-declared sensitivity arm widens to A/B/C. Shared-
curve measures (FET, PEF, FEF25-75, EV) require BOTH grades allowed. Grade
columns are QC metadata only and are dropped by the gating function.

Run from the repo root:
    python -m pytest tests/test_quality_gating.py -q
or:
    python tests/test_quality_gating.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "notebooks"))

from asthma_pipeline import (  # noqa: E402
    apply_spirometry_quality_gating,
    PRIMARY_ALLOWED_GRADES,
    SENSITIVITY_ALLOWED_GRADES,
    PRIMARY_MODEL_EXCLUSIONS,
)

DATA = os.path.join(HERE, "..", "data", "processed", "03_cleaned.parquet")


def _skip(msg):
    """Skip that works under pytest AND the plain-python runner."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest
        pytest.skip(msg)
    print("SKIPPED " + msg)
    return True


def _frame():
    # rows: (FEV1 grade, FVC grade) = (A,A) (B,B) (C,C) (D,D) (F,A) (A,D) ('',''), no-spiro
    return pd.DataFrame({
        "SPXNFEV1": [2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, np.nan, np.nan],
        "SPXNFVC":  [2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, np.nan, np.nan],
        "SPXNFET":  [6.0, 6.1, 6.2, 6.3, 6.4, 6.5, np.nan, np.nan],
        "SPXNPEF":  [400.0, 410.0, 420.0, 430.0, 440.0, 450.0, np.nan, np.nan],
        "SPXNQFV1": ["A", "B", "C", "D", "F", "A", "", None],
        "SPXNQFVC": ["A", "B", "C", "D", "A", "D", "", None],
        "RIDAGEYR": [8.0] * 8,
    })


def test_primary_ab_gating():
    out = apply_spirometry_quality_gating(_frame(), PRIMARY_ALLOWED_GRADES,
                                          verbose=False)
    # rows 0,1 (A/B) keep everything
    assert out.loc[0:1, ["SPXNFEV1", "SPXNFVC", "SPXNFET"]].notna().all().all()
    # rows 2,3 (C,C / D,D): all gated in the primary
    assert out.loc[2:3, ["SPXNFEV1", "SPXNFVC", "SPXNFET", "SPXNPEF"]].isna().all().all()
    # row 4 (F,A): FEV1 gated, FVC kept, shared-curve gated (both required)
    assert np.isnan(out.loc[4, "SPXNFEV1"]) and out.loc[4, "SPXNFVC"] == 2900.0
    assert np.isnan(out.loc[4, "SPXNFET"])
    # row 5 (A,D): FEV1 kept, FVC gated, shared-curve gated
    assert out.loc[5, "SPXNFEV1"] == 2500.0 and np.isnan(out.loc[5, "SPXNFVC"])
    assert np.isnan(out.loc[5, "SPXNPEF"])
    # grade columns dropped
    assert "SPXNQFV1" not in out.columns and "SPXNQFVC" not in out.columns


def test_sensitivity_abc_keeps_grade_c():
    out = apply_spirometry_quality_gating(_frame(), SENSITIVITY_ALLOWED_GRADES,
                                          verbose=False)
    assert out.loc[2, "SPXNFEV1"] == 2200.0 and out.loc[2, "SPXNFVC"] == 2700.0
    assert out.loc[2, "SPXNFET"] == 6.2
    # D/F still gated
    assert out.loc[3, ["SPXNFEV1", "SPXNFVC"]].isna().all()
    assert np.isnan(out.loc[4, "SPXNFEV1"])


def test_missing_rows_stay_missing():
    out = apply_spirometry_quality_gating(_frame(), PRIMARY_ALLOWED_GRADES,
                                          verbose=False)
    assert out.loc[6:7, ["SPXNFEV1", "SPXNFVC", "SPXNFET"]].isna().all().all()


def test_raises_without_grade_columns():
    df = _frame().drop(columns=["SPXNQFV1", "SPXNQFVC"])
    try:
        apply_spirometry_quality_gating(df, PRIMARY_ALLOWED_GRADES, verbose=False)
    except ValueError as e:
        assert "SPXNQFV1" in str(e)
    else:
        raise AssertionError("gating must refuse to run without grade columns")


def test_urdnallc_in_primary_exclusions():
    assert "URDNALLC" in PRIMARY_MODEL_EXCLUSIONS


def test_real_data_gating_counts():
    """Pinned against 02b_harmonized on the 6,567 analytic cohort:
    A/B: FEV1 usable 5,189; FVC usable 4,859; both usable 4,726.
    A/B/C: both usable 5,440 (1,127 without usable ratio spirometry).
    Runs only after notebook 03 has been re-executed with the grade
    columns retained (patched 2026-08-27)."""
    if not os.path.exists(DATA):
        return _skip("test_real_data_gating_counts (03_cleaned.parquet absent)")
    df = pd.read_parquet(DATA)
    if "SPXNQFV1" not in df.columns:
        return _skip("test_real_data_gating_counts "
                     "(03_cleaned predates the grade-retention patch)")
    an = df[df.WTMEC2YR > 0]
    assert len(an) == 6567
    ab = apply_spirometry_quality_gating(an, PRIMARY_ALLOWED_GRADES, verbose=False)
    assert int(ab["SPXNFEV1"].notna().sum()) == 5189
    assert int(ab["SPXNFVC"].notna().sum()) == 4859
    assert int((ab["SPXNFEV1"].notna() & ab["SPXNFVC"].notna()).sum()) == 4726
    abc = apply_spirometry_quality_gating(an, SENSITIVITY_ALLOWED_GRADES, verbose=False)
    n_both = int((abc["SPXNFEV1"].notna() & abc["SPXNFVC"].notna()).sum())
    assert n_both == 5440 and len(an) - n_both == 1127
    return False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = n_skip = 0
    for fn in fns:
        skipped = fn()
        if skipped:
            n_skip += 1
        else:
            print(f"PASS {fn.__name__}")
            n_pass += 1
    print(f"\n{n_pass} tests passed" + (f", {n_skip} skipped" if n_skip else ""))
