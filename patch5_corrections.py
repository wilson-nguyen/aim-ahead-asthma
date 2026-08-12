"""
PATCH 5 -- apply the pre-specified defect corrections to nb04.

IMPORTANT: nb04 defines ClinicalFeatureEngineer TWICE. Python uses the SECOND
definition, which is the one the final model path instantiates. This patch targets
that second copy only.

Run WITH THE NOTEBOOK CLOSED:
    python patch5_corrections.py

Backup -> notebooks/04_model.ipynb.bak5
Then reopen nb04 and Run All.
"""
import json, shutil, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "04_model.ipynb")
shutil.copy(PATH, PATH + ".bak5")
nb = json.load(open(PATH, encoding="utf-8"))

# --- locate the LAST cell defining ClinicalFeatureEngineer (the effective one) ---
target = None
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") == "code" and "class ClinicalFeatureEngineer" in "".join(c.get("source", [])):
        target = i
if target is None:
    raise SystemExit("ERROR: ClinicalFeatureEngineer not found.")
print(f"Patching cell index {target} (the effective, second definition).")

src = "".join(nb["cells"][target]["source"])

REPLACEMENTS = [
    # D1: preserve missingness instead of NaN < 0.8 -> False -> 0.0
    ("X_df['obstruction_indicator'] = (X_df['fev1_fvc_ratio'] < 0.8).astype(float)",
     "X_df['obstruction_indicator'] = safe_indicator(X_df['fev1_fvc_ratio'], 0.8)"),

    # D3: zero is not a neutral value for physiological measures
    ("X_df['fev1_log'] = np.log1p(X_df['SPXNFEV1'].fillna(0))",
     "X_df['fev1_log'] = np.log1p(X_df['SPXNFEV1'])"),
    ("X_df['cotinine_log'] = np.log1p(X_df['LBXCOT'].fillna(0))",
     "X_df['cotinine_log'] = np.log1p(X_df['LBXCOT'])"),
    ("X_df['bmi_log'] = np.log1p(X_df['BMXBMI'].fillna(0))",
     "X_df['bmi_log'] = np.log1p(X_df['BMXBMI'])"),

    # D5: adult BMI cutoff -> CDC BMI-for-age; D4: drop the affine duplicate
    ("X_df['obese'] = (X_df['BMXBMI'] >= 30).astype(float)",
     "_z_cdc, _pct_cdc = cdc_bmi_z(X_df['BMXBMI'], X_df.get('RIDAGEEX_H'), X_df.get('RIAGENDR'))\n"
     "            X_df['bmi_z_cdc'] = _z_cdc\n"
     "            X_df['obese'] = np.where(np.isnan(_pct_cdc), np.nan, (_pct_cdc >= 95).astype(float))"),

    # D2: interaction must preserve missingness
    ("X_df['family_spirometry_interaction'] = X_df['MCQ300B'].fillna(0) * X_df['fev1_fvc_ratio'].fillna(0)",
     "X_df['family_spirometry_interaction'] = safe_interaction(X_df['MCQ300B'], X_df['fev1_fvc_ratio'])"),
]

applied, missing = [], []
for old, new in REPLACEMENTS:
    if old in src:
        src = src.replace(old, new, 1)
        applied.append(old.split("=")[0].strip())
    else:
        missing.append(old.split("=")[0].strip())

# D4: remove the affine-duplicate bmi_zscore lines entirely
out_lines = []
for line in src.splitlines(keepends=True):
    if "bmi_zscore" in line:
        out_lines.append(line.replace(line.lstrip(), "# [R3 correction] removed affine duplicate of BMXBMI: " + line.lstrip()))
        applied.append("bmi_zscore (removed)")
    else:
        out_lines.append(line)
src = "".join(out_lines)

# add the import at the top of the cell
IMPORT = ("# [R3 corrections] pediatric + missingness-safe helpers\n"
          "from pediatric_corrections import safe_indicator, safe_interaction, cdc_bmi_z\n\n")
if "from pediatric_corrections import" not in src:
    src = IMPORT + src

nb["cells"][target]["source"] = src.splitlines(keepends=True)
json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("\nAPPLIED:")
for a in applied:
    print("   +", a)
if missing:
    print("\nNOT FOUND (check these manually):")
    for m in missing:
        print("   !", m)
print(f"\nBackup: {PATH}.bak5")
print("Now reopen nb04 and Run All.")
print("\nEXPECT: the selected 20 features will change. That is intended -- the")
print("feature matrix is genuinely different now. nb05 must be rerun afterward.")
