"""
PATCH 9 -- append the four pre-specified sensitivity analyses to nb05.

Implements the analyses named in Analysis_decisions_prespecified.md, which answer
Reviewer #3's "would the results differ" questions with evidence:

  S1  performance by age group (+ case/control age standardized difference)
  S2  spirometry-complete subsample
  S3  complete-BMI subsample
  S4  age-balanced test subsample

All are evaluation-only on the existing fitted model -- no refit, no retuning,
and the held-out test set is never used to choose anything.

Run WITH THE NOTEBOOK CLOSED:
    python patch9_sensitivity.py

Backup -> notebooks/05_top10_sensitivity.ipynb.bak9
"""
import json, shutil, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "05_top10_sensitivity.ipynb")
shutil.copy(PATH, PATH + ".bak9")
nb = json.load(open(PATH, encoding="utf-8"))

CODE = r'''# [R3] Pre-specified sensitivity analyses (evaluation-only, no refit)
import numpy as np, json
from sklearn.metrics import roc_auc_score, confusion_matrix

IDX = {n: i for i, n in enumerate(feature_names)}
Xte = np.asarray(X_test)
yte = np.asarray(y_test).ravel()

# tuned threshold: sensitivity >= 0.80 on VALIDATION, then applied unchanged
_pv = model_a.predict_proba(np.asarray(X_val))[:, 1]
_yv = np.asarray(y_val).ravel()
_grid = np.linspace(0.01, 0.99, 981)
_ok = [t for t in _grid if ((_pv >= t)[_yv == 1].mean() >= 0.80)]
THR = max(_ok) if _ok else 0.5
p_te = model_a.predict_proba(Xte)[:, 1]
print(f"Threshold (validation, sensitivity >= 0.80): {THR:.4f}\n")

def perf(mask, label):
    m = np.asarray(mask).ravel()
    if m.sum() < 50 or len(np.unique(yte[m])) < 2:
        print(f"{label:38s} n={m.sum():5d}  (too few / single class)")
        return None
    yy, pp = yte[m], p_te[m]
    pred = (pp >= THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    r = dict(n=int(m.sum()), cases=int(yy.sum()),
             auc=roc_auc_score(yy, pp),
             sens=tp / (tp + fn) if tp + fn else np.nan,
             spec=tn / (tn + fp) if tn + fp else np.nan,
             ppv=tp / (tp + fp) if tp + fp else np.nan)
    print(f"{label:38s} n={r['n']:5d}  AUC={r['auc']:.3f}  "
          f"sens={r['sens']:.3f}  spec={r['spec']:.3f}  PPV={r['ppv']:.3f}")
    return r

results = {}
print("=" * 92)
print("OVERALL (reference)")
print("=" * 92)
results["overall"] = perf(np.ones(len(yte), bool), "Full held-out test set")

# ---- S1: performance by age group -----------------------------------------
print("\n" + "=" * 92)
print("S1. PERFORMANCE BY AGE GROUP  (Reviewer #3, comment 1)")
print("=" * 92)
age = Xte[:, IDX["RIDAGEYR"]]          # RobustScaler is monotonic; tertiles are valid
q1, q2 = np.quantile(age, [1/3, 2/3])
for lab, m in [("Youngest tertile", age <= q1),
               ("Middle tertile", (age > q1) & (age <= q2)),
               ("Oldest tertile", age > q2)]:
    results[f"age_{lab}"] = perf(m, lab)

# case/control age difference, standardized
a1, a0 = age[yte == 1], age[yte == 0]
sd = (a1.mean() - a0.mean()) / np.sqrt((a1.var(ddof=1) + a0.var(ddof=1)) / 2)
print(f"\nStandardized difference in age (cases vs controls): {sd:.3f}")
print("  |d| < 0.10 is conventionally considered negligible imbalance.")

# ---- S2 / S3: completeness subsamples --------------------------------------
def flag_missing(col):
    """Missingness indicator survives scaling as two distinct values; take the higher."""
    v = Xte[:, IDX[col]]
    u = np.unique(v)
    return v > u.mean() if len(u) == 2 else v > np.median(v)

print("\n" + "=" * 92)
print("S2/S3. COMPLETENESS SUBSAMPLES  (Reviewer #3, comments 1-2)")
print("=" * 92)
if "SPXNFEV1_missing" in IDX:
    results["spiro_complete"] = perf(~flag_missing("SPXNFEV1_missing"), "Spirometry complete")
    results["spiro_missing"] = perf(flag_missing("SPXNFEV1_missing"), "Spirometry missing")
if "BMXBMI_missing" in IDX:
    results["bmi_complete"] = perf(~flag_missing("BMXBMI_missing"), "BMI complete")

# ---- S4: age-balanced test subsample ---------------------------------------
print("\n" + "=" * 92)
print("S4. AGE-BALANCED TEST SUBSAMPLE  (Reviewer #3, comment 1)")
print("=" * 92)
rng = np.random.default_rng(42)
bins = np.quantile(age, np.linspace(0, 1, 11))
keep = []
for lo, hi in zip(bins[:-1], bins[1:]):
    inb = np.where((age >= lo) & (age <= hi))[0]
    ci, co = inb[yte[inb] == 1], inb[yte[inb] == 0]
    k = min(len(ci), len(co))
    if k:
        keep += list(rng.choice(ci, k, replace=False)) + list(rng.choice(co, k, replace=False))
mask = np.zeros(len(yte), bool); mask[keep] = True
results["age_balanced"] = perf(mask, "Age-balanced (1:1 within age deciles)")
if mask.sum():
    ab = age[mask]; b1, b0 = ab[yte[mask] == 1], ab[yte[mask] == 0]
    sdb = (b1.mean() - b0.mean()) / np.sqrt((b1.var(ddof=1) + b0.var(ddof=1)) / 2)
    print(f"  standardized age difference after balancing: {sdb:.3f} (was {sd:.3f})")

with open(OUTPUTS_DIR / "sensitivity_analyses.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved -> {OUTPUTS_DIR / 'sensitivity_analyses.json'}")
'''

if not any(c.get("id") == "r3-sensitivity" for c in nb["cells"]):
    nb["cells"].append({"cell_type": "code", "execution_count": None,
                        "id": "r3-sensitivity", "metadata": {}, "outputs": [],
                        "source": CODE.splitlines(keepends=True)})
    msg = "appended sensitivity cell"
else:
    for c in nb["cells"]:
        if c.get("id") == "r3-sensitivity":
            c["source"] = CODE.splitlines(keepends=True)
    msg = "replaced existing sensitivity cell"

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(msg)
try:
    compile(CODE, "cell", "exec")
    print("cell compiles cleanly.")
except SyntaxError as e:
    print(f"!! syntax error: {e}")
print(f"Backup: {PATH}.bak9")
print("Reopen nb05 and run the new last cell (no need to Run All).")
