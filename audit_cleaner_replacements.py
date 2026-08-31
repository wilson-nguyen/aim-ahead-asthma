"""
audit_cleaner_replacements.py — per-variable evidence of what NHANESCleaner
actually replaces on the analytic modeling frame.

Context [31 Aug 2026]: external review noted that variables without an
explicit CATEGORICAL_SENTINELS entry fall back to a width-rule heuristic
guarded by a 2% ambiguity threshold, and demonstrated on synthetic data
that a legitimate rare code could in principle be scrubbed. This audit
answers the question that matters for THIS analysis: what does the fitted
cleaner replace on the real frame the models see?

It replays the notebook-04 preparation (gating, exclusions, row filters),
fits the cleaner on the frame, and records every variable's classification
and every value the transform would replace, with counts.

Finding at the analysis of record (run 20260831_103201): ZERO values are
replaced — Phase 2 recoding already handled questionnaire nonresponse
codes upstream, so the model-stage cleaner is a defensive no-op on real
data. This file regenerates that evidence rather than asserting it.

Run from the repo root:
    python audit_cleaner_replacements.py

Writes: outputs/cleaner_replacement_audit.json (commit this file)
"""
import json
import os
import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "notebooks"))

from asthma_pipeline import (  # noqa: E402
    NHANESCleaner, apply_spirometry_quality_gating, PRIMARY_ALLOWED_GRADES,
    PRIMARY_MODEL_EXCLUSIONS, LEAKY_PROXIES, AGE_RESTRICTED_VARS, IDENTIFIERS,
    CATEGORICAL_SENTINELS,
)


def main():
    df = pd.read_parquet(os.path.join(HERE, "data", "processed", "03_cleaned.parquet"))
    df = df.drop(columns=[c for c in ("NHANES_CYCLE",) if c in df.columns])
    df = apply_spirometry_quality_gating(df, PRIMARY_ALLOWED_GRADES, verbose=False)
    y, sw = df["MCQ010"], df["WTMEC2YR"]
    X = df[[c for c in df.columns
            if c not in ["MCQ010", "WTMEC2YR", "WTINT2YR", "SDMVPSU",
                         "SDMVSTRA", "SEQN"]]]
    m = y.isin([1.0, 2.0]) & sw.notna() & (sw > 0)
    X = X[m].reset_index(drop=True)
    excl = LEAKY_PROXIES + AGE_RESTRICTED_VARS + IDENTIFIERS + PRIMARY_MODEL_EXCLUSIONS
    X = X.drop(columns=[c for c in X.columns if c in excl])

    cl = NHANESCleaner().fit(X)
    variables = {}
    total = 0
    for col in X.columns:
        if col in cl.binary_cols_:
            n = int(X[col].isin([7, 9]).sum())
            variables[col] = {"class": "binary", "codes_scrubbed": [7, 9],
                              "values_replaced": n}
        elif col in cl.categorical_cols_:
            sents = sorted(cl.sentinel_map_.get(col, set()))
            n = int(X[col].isin(sents).sum()) if sents else 0
            variables[col] = {
                "class": "categorical",
                "sentinel_source": ("explicit codebook entry"
                                    if col in CATEGORICAL_SENTINELS
                                    else "width rule (2% ambiguity guard)"),
                "sentinel_codes": sents,
                "values_replaced": n,
            }
        else:
            variables[col] = {"class": "continuous (no replacement by design)",
                              "values_replaced": 0}
        total += variables[col]["values_replaced"]

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "frame": "notebook-04 analytic frame (gated, excluded, filtered), "
                 f"n={len(X)}, {X.shape[1]} columns",
        "total_values_replaced": total,
        "interpretation": (
            "0 means the model-stage cleaner changes nothing on real data: "
            "Phase 2 recoding already handled nonresponse codes, and the "
            "cleaner is a defensive regression guard. Any nonzero count "
            "must be adjudicated against the NHANES codebook before the "
            "affected run is treated as final."),
        "variables": variables,
    }
    path = os.path.join(HERE, "outputs", "cleaner_replacement_audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"total values replaced on the analytic frame: {total}")
    nonzero = {k: v for k, v in variables.items() if v["values_replaced"]}
    if nonzero:
        print("NONZERO replacements (adjudicate against the codebook):")
        for k, v in nonzero.items():
            print(f"  {k}: {v}")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
