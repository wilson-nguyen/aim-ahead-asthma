"""
Second one-time patch for notebooks/05_top10_sensitivity.ipynb.

Run WITH THE NOTEBOOK CLOSED:
    python patch2_nb05.py

It adds the nine remaining feature labels to the top of the figure cell
(id "regen-full-model-figs"), so the regenerated Figures 3, 4, 6 show readable
names instead of raw NHANES codes. A backup is written to ...ipynb.bak2 first.

Then reopen the notebook and Run All.
"""
import json, shutil, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "05_top10_sensitivity.ipynb")
shutil.copy(PATH, PATH + ".bak2")
nb = json.load(open(PATH, encoding="utf-8"))

LABELS_CODE = (
    "DISPLAY_LABELS.update({\n"
    "    \"BMXBMI\": \"Body Mass Index (kg/m²)\", \"BMXWAIST\": \"Waist Circumference (cm)\",\n"
    "    \"BMXARMC\": \"Arm Circumference (cm)\", \"bmi_zscore\": \"BMI Z-Score (Age-Adjusted)\",\n"
    "    \"HUQ090\": \"Saw Mental Health Professional (Past Year)\",\n"
    "    \"AGQ030\": \"Episode of Hay Fever (Past Year)\", \"RDQ140\": \"Dry Cough at Night (Past Year)\",\n"
    "    \"SPDBRONC\": \"Bronchodilator / 2nd-Test Eligibility\", \"ENQ020\": \"Spirometry Eligibility Screen\",\n"
    "})\n"
    "\n"
)

found = False
for cell in nb["cells"]:
    if cell.get("id") == "regen-full-model-figs":
        found = True
        existing = cell["source"] if isinstance(cell["source"], list) else [cell["source"]]
        if "DISPLAY_LABELS.update" not in "".join(existing):
            cell["source"] = LABELS_CODE.splitlines(keepends=True) + existing
        break

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("figure cell found and updated:", found)
print("Backup saved to:", PATH + ".bak2")
print("Now reopen the notebook and Run All.")
