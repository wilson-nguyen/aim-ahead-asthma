"""
PATCH 5b -- apply the same corrections to the FIRST ClinicalFeatureEngineer.

Patch 5 fixed the second (effective) class used by the final model. The first class
is the definition in scope during the model-search phase, so it must match or the
search will tune hyperparameters on the uncorrected feature space.

Run WITH THE NOTEBOOK CLOSED:
    python patch5b_first_class.py

Backup -> notebooks/04_model.ipynb.bak5b
"""
import json, shutil, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "04_model.ipynb")
shutil.copy(PATH, PATH + ".bak5b")
nb = json.load(open(PATH, encoding="utf-8"))

# exact-string corrections (applied to every remaining occurrence)
PAIRS = [
    ("X_df['obstruction_indicator'] = (X_df['fev1_fvc_ratio'] < 0.8).astype(float)",
     "X_df['obstruction_indicator'] = safe_indicator(X_df['fev1_fvc_ratio'], 0.8)"),
    ("X_df['obstruction_mild'] = ((X_df['fev1_fvc_ratio'] >= 0.7) & (X_df['fev1_fvc_ratio'] < 0.8)).astype(float)",
     "X_df['obstruction_mild'] = (safe_indicator(X_df['fev1_fvc_ratio'], 0.8) * (1 - safe_indicator(X_df['fev1_fvc_ratio'], 0.7)))"),
    ("X_df['obstruction_severe'] = (X_df['fev1_fvc_ratio'] < 0.6).astype(float)",
     "X_df['obstruction_severe'] = safe_indicator(X_df['fev1_fvc_ratio'], 0.6)"),
    ("X_df['fev1_log'] = np.log1p(X_df['SPXNFEV1'].fillna(0))",
     "X_df['fev1_log'] = np.log1p(X_df['SPXNFEV1'])"),
    ("X_df['fev1_sqrt'] = np.sqrt(X_df['SPXNFEV1'].fillna(0))",
     "X_df['fev1_sqrt'] = np.sqrt(X_df['SPXNFEV1'])"),
    ("X_df['pef_log'] = np.log1p(X_df['SPXNPEF'].fillna(0))",
     "X_df['pef_log'] = np.log1p(X_df['SPXNPEF'])"),
    ("X_df['cotinine_log'] = np.log1p(X_df['LBXCOT'].fillna(0))",
     "X_df['cotinine_log'] = np.log1p(X_df['LBXCOT'])"),
    ("X_df['bmi_log'] = np.log1p(X_df['BMXBMI'].fillna(0))",
     "X_df['bmi_log'] = np.log1p(X_df['BMXBMI'])"),
    ("X_df['bmi_sqrt'] = np.sqrt(X_df['BMXBMI'].fillna(0))",
     "X_df['bmi_sqrt'] = np.sqrt(X_df['BMXBMI'])"),
    ("X_df['family_spirometry_interaction'] = X_df['MCQ300B'].fillna(0) * X_df['fev1_fvc_ratio'].fillna(0)",
     "X_df['family_spirometry_interaction'] = safe_interaction(X_df['MCQ300B'], X_df['fev1_fvc_ratio'])"),
    ("X_df['family_obstruction_interaction'] = X_df['MCQ300B'].fillna(0) * X_df['obstruction_indicator'].fillna(0)",
     "X_df['family_obstruction_interaction'] = safe_interaction(X_df['MCQ300B'], X_df['obstruction_indicator'])"),
    ("X_df['gender_spirometry_interaction'] = X_df['RIAGENDR'].fillna(0) * X_df['fev1_fvc_ratio'].fillna(0)",
     "X_df['gender_spirometry_interaction'] = safe_interaction(X_df['RIAGENDR'], X_df['fev1_fvc_ratio'])"),
    ("X_df['wheeze_spirometry_interaction'] = X_df['RDQ070'].fillna(0) * X_df['fev1_fvc_ratio'].fillna(0)",
     "X_df['wheeze_spirometry_interaction'] = safe_interaction(X_df['RDQ070'], X_df['fev1_fvc_ratio'])"),
    # adult BMI cutoffs -> CDC BMI-for-age (>=95th percentile)
    ("X_df['obese'] = (X_df['BMXBMI'] >= 30).astype(float)",
     "_z_cdc, _pct_cdc = cdc_bmi_z(X_df['BMXBMI'], X_df.get('RIDAGEEX_H'), X_df.get('RIAGENDR'))\n"
     "            X_df['bmi_z_cdc'] = _z_cdc\n"
     "            X_df['obese'] = np.where(np.isnan(_pct_cdc), np.nan, (_pct_cdc >= 95).astype(float))"),
]

# adult-cutoff dummies that have no pediatric meaning -> comment out
DROP_PATTERNS = [
    r"X_df\['underweight'\] = \(X_df\['BMXBMI'\] < 18\.5\)",
    r"X_df\['normal_weight'\] = ",
    r"X_df\['overweight'\] = \(\(X_df\['BMXBMI'\] >= 25\)",
    r"X_df\['severely_obese'\] = \(X_df\['BMXBMI'\] >= 35\)",
    r"X_df\['bmi_zscore'\] = ",
]

IMPORT = "from pediatric_corrections import safe_indicator, safe_interaction, cdc_bmi_z\n"
applied, dropped, cells_touched = [], [], []

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "class ClinicalFeatureEngineer" not in src:
        continue
    before = src
    for old, new in PAIRS:
        if old in src:
            n = src.count(old)
            src = src.replace(old, new)
            applied.append(f"cell {i}: {old.split('=')[0].strip()} (x{n})")
    out = []
    for line in src.splitlines(keepends=True):
        if any(re.search(p, line) for p in DROP_PATTERNS) and not line.lstrip().startswith("#"):
            out.append(line.replace(line.lstrip(), "# [R3 correction] " + line.lstrip()))
            dropped.append(f"cell {i}: {line.strip()[:60]}")
        else:
            out.append(line)
    src = "".join(out)
    if "from pediatric_corrections import" not in src:
        src = "# [R3 corrections]\n" + IMPORT + "\n" + src
    if src != before:
        cell["source"] = src.splitlines(keepends=True)
        cells_touched.append(i)

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"cells modified: {cells_touched}")
print("\nREPLACED:")
for a in applied:
    print("   +", a)
print("\nCOMMENTED OUT (adult cutoffs / duplicate):")
for d in dropped:
    print("   -", d)
print(f"\nBackup: {PATH}.bak5b")
print("Now reopen nb04 and Run All.")
