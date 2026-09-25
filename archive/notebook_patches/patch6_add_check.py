"""
PATCH 6 -- insert the bmi_z_cdc verification cell into nb04.

Places the check immediately after the final preprocessing cell (the one that
creates X_train_feat / X_train_final), so a broken CDC z-score surfaces before
the final model fit rather than at the very end.

Run WITH THE NOTEBOOK CLOSED:
    python patch6_add_check.py

Backup -> notebooks/04_model.ipynb.bak6
"""
import json, shutil, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "04_model.ipynb")
shutil.copy(PATH, PATH + ".bak6")
nb = json.load(open(PATH, encoding="utf-8"))

CHECK = '''# [R3 correction] verify the CDC BMI-for-age z-score built correctly
import numpy as np

if 'bmi_z_cdc' not in X_train_feat.columns:
    print("!! bmi_z_cdc MISSING from the feature matrix.")
    print("   RIDAGEEX_H present:", 'RIDAGEEX_H' in X_train_clean.columns)
    print("   RIAGENDR  present:", 'RIAGENDR' in X_train_clean.columns)
else:
    z = X_train_feat['bmi_z_cdc']
    n_ok = np.isfinite(z).sum()
    print(f"bmi_z_cdc non-missing: {n_ok} of {len(z)} ({n_ok/len(z)*100:.1f}%)")
    print(z.describe())
    if n_ok == 0:
        print("\\n!! ALL NaN -- RIDAGEEX_H or RIAGENDR was dropped upstream.")
        print("   Stop and report this before relying on the run.")
    elif n_ok / len(z) < 0.90:
        print("\\n!! Coverage below 90% -- expected ~99%. Worth investigating.")
    else:
        print("\\nOK: coverage as expected (~99%).")
'''

# insert after the cell that builds the final preprocessed training matrix
anchor = None
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") == "code":
        s = "".join(c.get("source", []))
        if "X_train_final = scaler.fit_transform(X_train_imp)" in s:
            anchor = i
            break

if anchor is None:
    nb["cells"].append({"cell_type": "code", "execution_count": None,
                        "id": "check-bmi-z-cdc", "metadata": {}, "outputs": [],
                        "source": CHECK.splitlines(keepends=True)})
    where = "appended at end (anchor cell not found)"
else:
    nb["cells"].insert(anchor + 1, {"cell_type": "code", "execution_count": None,
                                    "id": "check-bmi-z-cdc", "metadata": {}, "outputs": [],
                                    "source": CHECK.splitlines(keepends=True)})
    where = f"inserted after cell {anchor} (final preprocessing)"

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("check cell", where)
print(f"Backup: {PATH}.bak6")
print("Now reopen nb04 and Run All.")
