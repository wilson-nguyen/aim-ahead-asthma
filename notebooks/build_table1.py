"""
Build Table 1 -- baseline characteristics, survey-weighted, on the 6,567 analytic cohort.

Fixes two problems in the submitted Table 1:
  1. Mixed denominators (6,784 eligible vs 6,567 analytic). This uses ONE cohort
     throughout and prints the sample flow explicitly.
  2. Raw BMI reported for growing children. This adds CDC BMI-for-age percentile
     categories (Reviewer #3, comment 2) alongside the mean.

Estimates are weighted by WTMEC2YR. Standard errors are Taylor-linearized using
SDMVSTRA / SDMVPSU. [Fixed 25 Aug 2026: the design variables are dropped in the
Phase-3 cleaning, so the earlier version silently fell back to weighted-only SEs
for every row. They are now rejoined through the verified SEQN participant
mapping (replay of Phase 3), and the linearization is actually performed, with
domain (subgroup) estimation over the full design and single-PSU strata
centered at the overall mean.]

Run from the notebooks/ directory:
    python build_table1.py

Writes: ../outputs/table1_baseline.csv  and  ../outputs/table1_baseline.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pediatric_corrections import cdc_bmi_z

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verify_split_reconstruction import replay_phase3  # noqa: E402

DATA = Path("../data/processed/03_cleaned.parquet")
OUT = Path("../outputs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)

# ---- rejoin survey design variables via the verified SEQN mapping ----------
replica, seqn = replay_phase3(str(Path("../data/processed")))
assert len(replica) == len(df) and list(replica.columns) == list(df.columns), \
    "Phase-3 replay does not match 03_cleaned - do not proceed"
design = (pd.read_parquet(Path("../data/processed/02b_harmonized.parquet"),
                          columns=["SEQN", "SDMVSTRA", "SDMVPSU"])
          .set_index("SEQN"))
df = df.reset_index(drop=True)
df["SDMVSTRA"] = design["SDMVSTRA"].reindex(seqn.to_numpy()).to_numpy()
df["SDMVPSU"] = design["SDMVPSU"].reindex(seqn.to_numpy()).to_numpy()
assert df["SDMVSTRA"].notna().all() and df["SDMVPSU"].notna().all(), \
    "design variables incomplete after SEQN join"

# ---- sample flow -----------------------------------------------------------
n_eligible = len(df)
an = df[df["WTMEC2YR"] > 0].copy()
n_analytic = len(an)
print("SAMPLE FLOW")
print(f"  Eligible children (aged 6-17, valid MCQ010): {n_eligible:,}")
print(f"  Excluded, nonpositive examination weight:    {n_eligible - n_analytic:,}")
print(f"  ANALYTIC COHORT:                             {n_analytic:,}\n")

an["asthma"] = (an["MCQ010"] == 1).astype(int)
w = an["WTMEC2YR"].to_numpy()
STRATA = an["SDMVSTRA"].to_numpy()
PSU = an["SDMVPSU"].to_numpy()
_psu_per_stratum = pd.DataFrame({"h": STRATA, "j": PSU}).groupby("h")["j"].nunique()
print(f"Design: {(_psu_per_stratum.index.size)} strata, "
      f"{int(_psu_per_stratum.sum())} PSUs, "
      f"{int((_psu_per_stratum == 1).sum())} single-PSU strata "
      f"(centered at overall mean)\n")

# ---- CDC BMI-for-age -------------------------------------------------------
z, pct = cdc_bmi_z(an["BMXBMI"], an["RIDAGEEX_H"], an["RIAGENDR"])
an["bmi_z_cdc"] = z
an["bmi_pct_cdc"] = pct
an["bmi_cat"] = pd.cut(pct, [-0.1, 5, 85, 95, 100.1],
                       labels=["Underweight (<5th)", "Healthy (5th-85th)",
                               "Overweight (85th-95th)", "Obese (>=95th)"])

def wmean(x, wt):
    m = np.isfinite(x) & np.isfinite(wt)
    return np.average(x[m], weights=wt[m]) if m.sum() else np.nan

def wse(x, wt, domain=None):
    """Taylor-linearized SE of the weighted mean under the survey design.

    Domain estimation over the FULL design: units outside the domain (or with
    the item missing) contribute zero to the linearized scores but their PSUs
    remain in the variance computation. Single-PSU strata are handled by
    centering the PSU total at the overall mean of PSU totals (conservative).
    """
    x = np.asarray(x, float)
    wt = np.asarray(wt, float)
    d = np.isfinite(x) & np.isfinite(wt)
    if domain is not None:
        d &= np.asarray(domain, bool)
    if not d.any():
        return np.nan
    W = wt[d].sum()
    mu = np.average(x[d], weights=wt[d])
    zi = np.where(d, wt * np.where(d, x - mu, 0.0), 0.0) / W

    frame = pd.DataFrame({"z": zi, "h": STRATA, "j": PSU})
    psu_tot = frame.groupby(["h", "j"])["z"].sum()
    var = 0.0
    grand_mean = psu_tot.mean()
    for h, tots in psu_tot.groupby(level=0):
        n_h = len(tots)
        if n_h > 1:
            var += n_h / (n_h - 1) * np.sum((tots.values - tots.values.mean()) ** 2)
        else:
            var += (tots.values[0] - grand_mean) ** 2   # lonely-PSU, centered
    return np.sqrt(var)

def wpct(mask, wt, denom=None):
    """Weighted % of `mask` WITHIN `denom` (defaults to everyone)."""
    d = np.isfinite(wt) if denom is None else (np.isfinite(wt) & denom)
    tot = np.sum(wt[d])
    return 100 * np.sum(wt[d & mask]) / tot if tot else np.nan

rows = []

def add_cont(label, col, dp=1):
    x = an[col].to_numpy(float)
    r = {"Variable": label, "Missing (n)": int(an[col].isna().sum())}
    for name, sub in [("Overall", np.ones(len(an), bool)),
                      ("No Asthma", an.asthma.values == 0),
                      ("Asthma", an.asthma.values == 1)]:
        r[name] = f"{wmean(x[sub], w[sub]):.{dp}f} ({wse(x, w, domain=sub):.{dp+1}f})"
    rows.append(r)

def add_cat(label, col, level):
    mask = (an[col] == level).to_numpy()
    r = {"Variable": f"{label}: {level}", "Missing (n)": int(an[col].isna().sum())}
    for name, sub in [("Overall", np.ones(len(an), bool)),
                      ("No Asthma", an.asthma.values == 0),
                      ("Asthma", an.asthma.values == 1)]:
        r[name] = f"{wpct(mask, w, denom=sub):.1f}%"
    rows.append(r)

add_cont("Age, years", "RIDAGEYR")
_fem = (an.RIAGENDR == 2).values
rows.append({"Variable": "Female, %", "Missing (n)": 0,
             "Overall": f"{wpct(_fem, w):.1f}%",
             "No Asthma": f"{wpct(_fem, w, denom=an.asthma.values == 0):.1f}%",
             "Asthma": f"{wpct(_fem, w, denom=an.asthma.values == 1):.1f}%"})
add_cont("Body Mass Index, kg/m2", "BMXBMI")
add_cont("BMI-for-age z-score (CDC)", "bmi_z_cdc")
for lev in ["Underweight (<5th)", "Healthy (5th-85th)",
            "Overweight (85th-95th)", "Obese (>=95th)"]:
    add_cat("BMI-for-age category", "bmi_cat", lev)
for col, lab in [("FEV1/FVC ratio", None)]:
    pass
if {"SPXNFEV1", "SPXNFVC"} <= set(an.columns):
    m = an.SPXNFEV1.notna() & an.SPXNFVC.notna() & (an.SPXNFVC > 0)
    an["fev1_fvc"] = np.nan
    an.loc[m, "fev1_fvc"] = an.loc[m, "SPXNFEV1"] / an.loc[m, "SPXNFVC"]
    add_cont("FEV1/FVC ratio", "fev1_fvc", dp=3)
if "INDFMPIR" in an.columns:
    add_cont("Family income-to-poverty ratio", "INDFMPIR")

t1 = pd.DataFrame(rows)[["Variable", "Overall", "No Asthma", "Asthma", "Missing (n)"]]

hdr = (f"Table 1. Baseline characteristics of the analytic cohort "
       f"(N = {n_analytic:,}), survey-weighted.\n"
       f"Weighted estimates use WTMEC2YR. Continuous variables: weighted mean "
       f"(Taylor-linearized SE using SDMVSTRA/SDMVPSU; single-PSU strata centered "
       f"at the overall mean). Categorical: weighted %.\n"
       f"Sample flow: {n_eligible:,} eligible; {n_eligible - n_analytic:,} excluded "
       f"for nonpositive examination weight; {n_analytic:,} analytic.\n"
       f"Unweighted counts: no asthma {int((an.asthma == 0).sum()):,}, "
       f"asthma {int((an.asthma == 1).sum()):,} "
       f"(weighted prevalence {wpct(an.asthma.values == 1, w):.1f}%).\n")

t1.to_csv(OUT / "table1_baseline.csv", index=False)

def to_md(frame):
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(frame.columns)
    widths = [max(len(str(c)), *(len(str(v)) for v in frame[c])) for c in cols]
    head = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    rule = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([head, rule] + body)

(OUT / "table1_baseline.md").write_text(hdr + "\n" + to_md(t1), encoding="utf-8")

print(hdr)
print(t1.to_string(index=False))
print(f"\nSaved -> {OUT/'table1_baseline.csv'} and {OUT/'table1_baseline.md'}")
