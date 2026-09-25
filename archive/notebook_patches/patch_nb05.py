"""
One-time patch for notebooks/05_top10_sensitivity.ipynb.

Run this WITH THE NOTEBOOK CLOSED, from anywhere:
    python patch_nb05.py

It (1) removes the stray pasted "corrections" cell, (2) replaces TOP_10_FEATURES
with the model's actual top ten, (3) adds the HUQ020 and DMDHRBR_US labels to
DISPLAY_LABELS, and (4) appends a cell that regenerates the full-model Figures
3, 4, 6. A backup is written to ...ipynb.bak first.

Then reopen the notebook and Run All.
"""
import json, re, shutil, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "05_top10_sensitivity.ipynb")
shutil.copy(PATH, PATH + ".bak")
nb = json.load(open(PATH, encoding="utf-8"))

CORRECT_TOP10 = ["RDQ070", "MCQ300B", "HUQ010", "fev1_fvc_ratio", "family_spirometry_interaction",
                 "HUQ050", "HUQ020", "DMDHRBR_US", "SPXNFET", "PFQ020"]
NEW_LABELS = {"HUQ020": "Health Now Compared With 1 Year Ago",
              "DMDHRBR_US": "Household Reference Person Born in the US"}

def to_lines(s):
    return s.splitlines(keepends=True)

FIG_CODE = r'''# Regenerate full-model Figures 3, 4, 6 (SelectKBest features, corrected labels)
import numpy as np, matplotlib.pyplot as plt, shap
from catboost import Pool
from sklearn.inspection import permutation_importance

selector = model_a.named_steps['feature_selection']
clf      = model_a.named_steps['classifier']
mask     = selector.get_support()
sel_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
disp = [DISPLAY_LABELS.get(f, f) for f in sel_names]

Xs = selector.transform(np.asarray(X_test))
sv = np.asarray(clf.get_feature_importance(Pool(Xs), type="ShapValues"))[:, :-1]
mean_abs = np.abs(sv).mean(axis=0); order = np.argsort(mean_abs)[::-1]

idx = order[:20][::-1]
plt.figure(figsize=(9, 10)); plt.barh([disp[i] for i in idx], mean_abs[idx], color="#1f77e0")
plt.xlabel("Mean Absolute SHAP Value"); plt.title("Feature Importance Ranking")
plt.tight_layout(); plt.savefig("outputs/figure_4_full_ranking_CORRECTED.png", dpi=300); plt.close()

plt.figure(); shap.summary_plot(sv, Xs, feature_names=disp, max_display=20, show=False)
plt.title("Feature Impact on Asthma Prediction"); plt.tight_layout()
plt.savefig("outputs/figure_6_full_summary_CORRECTED.png", dpi=300, bbox_inches="tight"); plt.close()

perm = permutation_importance(model_a, X_test, y_test, n_repeats=10, random_state=42)
perm_sel = np.array([perm.importances_mean[i] for i in range(len(feature_names)) if mask[i]])
sn = mean_abs / mean_abs.max(); pn = perm_sel / max(perm_sel.max(), 1e-9)
top = order[:15][::-1]; yy = np.arange(len(top))
plt.figure(figsize=(10, 8))
plt.barh(yy + 0.2, [sn[i] for i in top], height=0.4, label="SHAP Importance")
plt.barh(yy - 0.2, [pn[i] for i in top], height=0.4, label="Permutation Importance")
plt.yticks(yy, [disp[i] for i in top]); plt.xlabel("Normalized Importance")
plt.title("Feature Importance Comparison: SHAP vs Permutation"); plt.legend()
plt.tight_layout(); plt.savefig("outputs/figure_3_full_shap_vs_perm_CORRECTED.png", dpi=300); plt.close()

top10 = [sel_names[i] for i in order[:10]]
print("model top10:", top10)
print("matches TOP_10_FEATURES?:", set(top10) == set(TOP_10_FEATURES))
print("Saved corrected Figures 3, 4, 6 to outputs/.")
'''

out, log = [], []
for cell in nb["cells"]:
    s = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and "Reduced-model correction (align everything" in s:
        log.append("removed pasted correction cell")
        continue
    if "TOP_10_FEATURES = [" in s and "manuscript Figure 4 to actual column names" in s:
        newlist = "TOP_10_FEATURES = [\n" + "".join("    %r,\n" % f for f in CORRECT_TOP10) + "]\n"
        s = re.sub(r"TOP_10_FEATURES = \[.*?\n\]\n", newlist, s, count=1, flags=re.S)
        cell["source"] = to_lines(s)
        log.append("fixed TOP_10_FEATURES")
    if "DISPLAY_LABELS = {" in s:
        add = "".join("    %r: %r,\n" % (k, v) for k, v in NEW_LABELS.items())
        s = s.replace("DISPLAY_LABELS = {\n", "DISPLAY_LABELS = {\n" + add, 1)
        cell["source"] = to_lines(s)
        log.append("added DISPLAY_LABELS entries")
    out.append(cell)

out.append({"cell_type": "code", "execution_count": None, "id": "regen-full-model-figs",
            "metadata": {}, "outputs": [], "source": to_lines(FIG_CODE)})
log.append("appended full-model figure cell")

nb["cells"] = out
json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("PATCH COMPLETE:")
for x in log:
    print("  -", x)
print("Backup saved to:", PATH + ".bak")
print("Now reopen the notebook and Run All.")
