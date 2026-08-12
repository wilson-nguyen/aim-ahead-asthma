"""
PATCH 8 -- nb05: derive TOP_10_FEATURES from DEVELOPMENT data instead of hardcoding.

Implements pre-specified decision D7. The hardcoded list is stale after the
corrections (bmi_zscore removed, bmi_z_cdc added) and its provenance was never
reproducible. This derives the top 10 from full-model SHAP on training +
validation data, never the held-out test set.

Also makes DISPLAY_LABELS lookups safe and adds a label for bmi_z_cdc.

Run WITH THE NOTEBOOK CLOSED:
    python patch8_nb05_derive_top10.py

Backup -> notebooks/05_top10_sensitivity.ipynb.bak8
"""
import json, shutil, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "05_top10_sensitivity.ipynb")
shutil.copy(PATH, PATH + ".bak8")
nb = json.load(open(PATH, encoding="utf-8"))

DERIVE = '''# [R3 correction, decision D7] Derive the top 10 from DEVELOPMENT data
# (training + validation). Not hardcoded, and never from the held-out test set.
import numpy as np
from catboost import Pool as _Pool

_sel = model_a.named_steps['feature_selection']
_clf = model_a.named_steps['classifier']
_mask = _sel.get_support()
_sel_names = [feature_names[i] for i in range(len(feature_names)) if _mask[i]]

_Xdev = np.vstack([np.asarray(X_train), np.asarray(X_val)])
_sv = np.asarray(_clf.get_feature_importance(_Pool(_sel.transform(_Xdev)),
                                             type="ShapValues"))[:, :-1]
_imp = np.abs(_sv).mean(axis=0)
_order = np.argsort(_imp)[::-1]
TOP_10_FEATURES = [_sel_names[i] for i in _order[:10]]

print(f"SelectKBest retained {len(_sel_names)} features.")
print("Top 10 by full-model SHAP on development data (train + val):")
for _r, _i in enumerate(_order[:10], 1):
    print(f"  {_r:2d}. {_sel_names[_i]:38s} {_imp[_i]:.4f}")
print()
'''

target = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if re.search(r"TOP_10_FEATURES\s*=\s*\[", src):
        target = i
        # replace the literal list assignment with the derivation
        src = re.sub(r"TOP_10_FEATURES\s*=\s*\[.*?\n\]\n", DERIVE, src, count=1, flags=re.S)
        cell["source"] = src.splitlines(keepends=True)
        break

if target is None:
    raise SystemExit("ERROR: hardcoded TOP_10_FEATURES not found.")
print(f"replaced hardcoded list in cell {target}")

# make label lookups safe + add bmi_z_cdc
label_fix = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    orig = src
    if "DISPLAY_LABELS = {" in src and "bmi_z_cdc" not in src:
        src = src.replace(
            "DISPLAY_LABELS = {\n",
            "DISPLAY_LABELS = {\n"
            "    'bmi_z_cdc': 'BMI-for-Age Z-Score (CDC)',\n", 1)
    # DISPLAY_LABELS[f] -> DISPLAY_LABELS.get(f, f) so a new feature cannot KeyError
    src = src.replace("DISPLAY_LABELS[f] for f in TOP_10_FEATURES",
                      "DISPLAY_LABELS.get(f, f) for f in TOP_10_FEATURES")
    if src != orig:
        cell["source"] = src.splitlines(keepends=True)
        label_fix += 1

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"label-handling cells updated: {label_fix}")

bad = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    s = "".join(cell.get("source", []))
    if s.strip().startswith(("!", "%")):
        continue
    try:
        compile(s, f"cell{i}", "exec")
    except SyntaxError as e:
        bad += 1
        print(f"  !! cell {i}: {e.msg} (line {e.lineno})")
print("all code cells compile cleanly." if bad == 0 else f"{bad} cell(s) failed to compile")
print(f"\nBackup: {PATH}.bak8")
print("Now reopen nb05, RESTART THE KERNEL, and Run All.")
