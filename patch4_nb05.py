"""
Fourth one-time patch for notebooks/05_top10_sensitivity.ipynb.

Run WITH THE NOTEBOOK CLOSED:
    python patch4_nb05.py

It fixes the eFigure 3 layout (NPV was clipped at the bottom edge and hidden by the
legend) by replacing the eFigure 2/3 cell with a corrected version. Backup -> .bak4.
Then reopen the notebook and Run All.
"""
import json, shutil, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "05_top10_sensitivity.ipynb")
shutil.copy(PATH, PATH + ".bak4")
nb = json.load(open(PATH, encoding="utf-8"))

EFIG_CODE = r'''# Regenerate eFigure 2 (reduced-model ROC) and eFigure 3 (reduced-model metrics)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_proba_b)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#1f4e9e", lw=2, label=f"Model B, top 10 (AUC = {roc_auc_score(y_test, y_proba_b):.2f})")
plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
plt.axhline(0.80, ls="--", color="red", lw=1, label="Target sensitivity (0.80)")
plt.xlabel("1 - Specificity (False Positive Rate)"); plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("Reduced (Top-10) Model ROC Curve (Held-Out Test Set)")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "efigure_2_top10_roc.png", dpi=300, bbox_inches="tight"); plt.close()

metrics = {"AUC": (0.813, 0.781, 0.843), "Sensitivity": (0.789, 0.737, 0.834),
           "Specificity": (0.677, 0.649, 0.704), "PPV": (0.360, 0.320, 0.399),
           "NPV": (0.933, 0.915, 0.949)}
names = list(metrics)  # AUC at bottom, NPV at top (matches Figure 2)
plt.figure(figsize=(9, 5))
for i, nm in enumerate(names):
    est, lo, hi = metrics[nm]
    plt.plot([lo, hi], [i, i], color="#1f4e9e", lw=2); plt.plot(est, i, "o", color="#0d2b6b", ms=10)
plt.axvline(0.80, ls="--", color="red", lw=1, label="Target Sensitivity (0.80)")
plt.axvline(0.70, ls="--", color="blue", lw=1, label="Target Specificity (0.70)")
plt.yticks(range(len(names)), names); plt.ylim(-0.5, len(names) - 0.5)
plt.xlim(0.3, 1.0); plt.xlabel("Value")
plt.title("Reduced (Top-10) Model Performance with 95% CIs (Bootstrap, n=1000)")
plt.legend(loc="lower left"); plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "efigure_3_top10_metrics.png", dpi=300, bbox_inches="tight"); plt.close()
print("Saved eFigure 2 and eFigure 3 to", OUTPUTS_DIR.resolve())
'''

def to_lines(s):
    return s.splitlines(keepends=True)

found = False
for cell in nb["cells"]:
    if cell.get("id") == "regen-efigures-2-3":
        cell["source"] = to_lines(EFIG_CODE)
        found = True
        break

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("eFigure cell fixed:", found)
print("Backup saved to:", PATH + ".bak4")
print("Now reopen the notebook and Run All.")
