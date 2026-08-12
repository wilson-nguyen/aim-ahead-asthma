"""
PATCH 5c -- systematic .fillna(0) sweep on CONTINUOUS MEASURED variables.

Patches 5 and 5b fixed the named defects. This removes the remaining
`.fillna(0)` calls on continuous physiological / anthropometric / lab /
socioeconomic measures inside the feature-engineering classes.

Why: zero is not a neutral value for a lung volume, a body measurement, or a
lab result. Zero-filling makes "not measured" look like an extreme observation
and hides it from the imputer.

Binary exposure flags (smoke_exposure_heavy, low_income, obese, likely_pubertal,
MCQ300B, RDQ070, RIAGENDR, RIDRETH1) are intentionally LEFT ALONE, because 0
there means "not exposed" and is defensible.

Removing .fillna(0) also fixes products automatically: NaN * x propagates NaN,
so interaction terms become missing rather than silently zero.

Run WITH THE NOTEBOOK CLOSED:
    python patch5c_fillna_sweep.py

Backup -> notebooks/04_model.ipynb.bak5c
"""
import json, shutil, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "04_model.ipynb")
shutil.copy(PATH, PATH + ".bak5c")
nb = json.load(open(PATH, encoding="utf-8"))

# continuous measured variables: zero is NOT a valid value
CONTINUOUS = [
    # spirometry
    "SPXNFEV1", "SPXNFVC", "SPXNPEF", "SPXNFET", "SPXNFEV3", "SPXNFEV6",
    "fev1_fvc_ratio",
    # anthropometry
    "BMXBMI", "BMXHT", "BMXWT", "BMXWAIST", "BMXARMC", "BMXLEG", "BMXARML",
    # labs
    "LBXCOT", "LBXWBCSI", "LBXEOPCT", "LBXTHC", "cotinine_log",
    # continuous socio / demographic
    "INDFMPIR", "RIDAGEYR",
]

pattern = re.compile(
    r"X_df\['(" + "|".join(re.escape(v) for v in CONTINUOUS) + r")'\]\.fillna\(0\)"
)

touched, changes = [], []
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "class ClinicalFeatureEngineer" not in src:
        continue
    hits = pattern.findall(src)
    if not hits:
        continue
    new_src = pattern.sub(r"X_df['\1']", src)
    cell["source"] = new_src.splitlines(keepends=True)
    touched.append(i)
    for v in sorted(set(hits)):
        changes.append(f"cell {i}: {v} (x{hits.count(v)})")

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"cells modified: {touched}")
print(f"total .fillna(0) removed: {sum(int(c.split('x')[-1].rstrip(')')) for c in changes)}")
print("\nBY VARIABLE:")
for c in changes:
    print("   -", c)
print("\nLEFT ALONE (binary flags, 0 = 'not exposed' is valid):")
print("   smoke_exposure_heavy, low_income, obese, likely_pubertal,")
print("   MCQ300B, RDQ070, RIAGENDR, RIDRETH1")
print(f"\nBackup: {PATH}.bak5c")
print("Now reopen nb04 and Run All.")
