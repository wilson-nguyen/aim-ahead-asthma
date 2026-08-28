"""
Regression tests for the CDC BMI-for-age half-month correction [28 Aug 2026].

CDC growth-chart SAS-program instruction: "If only the completed number of
months is known (as in NHANES), add 0.5 to the age." NHANES RIDAGEEX_H is
completed months; the LMS reference rows are month midpoints. cdc_bmi_z now
adds 0.5 internally (completed_months=True, the default).

Run from the repo root:
    python -m pytest tests/test_cdc_bmi_age.py -q
or:
    python tests/test_cdc_bmi_age.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "notebooks"))

from pediatric_corrections import cdc_bmi_z  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "processed", "03_cleaned.parquet")


def test_completed_months_adds_half():
    z_completed, _ = cdc_bmi_z([18.0], [99], [1])                       # default
    z_exact, _ = cdc_bmi_z([18.0], [99.5], [1], completed_months=False)
    assert np.isclose(z_completed[0], z_exact[0]), \
        "completed-month input must evaluate the reference at month + 0.5"
    z_old, _ = cdc_bmi_z([18.0], [99.0], [1], completed_months=False)
    assert not np.isclose(z_completed[0], z_old[0]), \
        "the correction must actually shift the evaluation age"


def test_shift_is_small_but_real():
    rng = np.random.default_rng(0)
    ages = rng.integers(72, 216, 500)
    bmis = rng.uniform(14, 30, 500)
    sexes = rng.choice([1, 2], 500)
    z_new, _ = cdc_bmi_z(bmis, ages, sexes)
    z_old, _ = cdc_bmi_z(bmis, ages, sexes, completed_months=False)
    d = np.abs(z_new - z_old)
    assert 0 < np.nanmean(d) < 0.05 and np.nanmax(d) < 0.1


def test_real_data_weighted_obesity():
    """Pinned on the 6,567 analytic cohort with the corrected ages:
    weighted obesity (>=95th pct, nonmissing denominators)
    asthma 24.1%, controls 17.8% (was 17.9% under the uncorrected ages)."""
    if not os.path.exists(DATA):
        print("SKIPPED test_real_data_weighted_obesity (03_cleaned.parquet absent)")
        return True
    df = pd.read_parquet(DATA)
    an = df[df.WTMEC2YR > 0]
    _, pct = cdc_bmi_z(an.BMXBMI, an.RIDAGEEX_H, an.RIAGENDR)
    pct = np.asarray(pct, float)
    g = (an.MCQ010 == 1).to_numpy()
    w = an.WTMEC2YR.to_numpy(float)
    known = np.isfinite(pct)
    assert int(known.sum()) == 6519
    vals = {}
    for name, sel in (("asthma", g), ("ctrl", ~g)):
        vals[name] = round(100 * w[sel & known & (pct >= 95)].sum()
                           / w[sel & known].sum(), 1)
    assert vals == {"asthma": 24.1, "ctrl": 17.8}, vals


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = n_skip = 0
    for fn in fns:
        if fn():
            n_skip += 1
        else:
            print(f"PASS {fn.__name__}")
            n_pass += 1
    print(f"\n{n_pass} tests passed" + (f", {n_skip} skipped" if n_skip else ""))
