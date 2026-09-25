"""
patch13_stale_notebook_text.py — remove contradictory pre-R3 content from
the rendered notebooks (external review, 31 Aug, v3.2 round).

  1. notebooks/04_model.ipynb cell 1 (markdown): still described the old
     two-step Model A/B selection and a test AUC of 0.83. Replaced with a
     pointer to the analysis of record.
  2. notebooks/05_top10_sensitivity.ipynb: permanently superseded and
     guard-blocked, but its STORED OUTPUTS (AUC 0.826, 112 features,
     HUQ050, the obstruction indicator) still rendered on GitHub. All
     outputs are cleared and a SUPERSEDED banner becomes the first cell.

Markdown/output changes only; no code cell of notebook 04 is touched, so
the executed analysis-of-record outputs there remain intact. Guarded and
idempotent. Run from the repo root:
    python patch13_stale_notebook_text.py
"""
import json
import sys

MARK04 = "superseded description removed 31 Aug 2026"
MARK05 = "SUPERSEDED NOTEBOOK — DO NOT RUN, DO NOT CITE"

applied = []

# --- 1. notebook 04, cell 1 markdown ---------------------------------------
NB4 = "notebooks/04_model.ipynb"
with open(NB4, encoding="utf-8") as f:
    nb = json.load(f)
c1 = nb["cells"][1]
src = "".join(c1["source"])
if MARK04 in src:
    print("[already applied] nb04 cell 1")
elif c1["cell_type"] == "markdown" and ("model selection" in src.lower()
                                        or "0.83" in src):
    c1["source"] = (
        f"*Pre-R3 model-selection narrative ({MARK04}): the description that "
        "stood here referenced the superseded two-step Model A/B screening "
        "and its AUC 0.83 result. The analysis of record is run "
        "`tuning_results_20260831_103201`; see the repository README and "
        "`outputs/RELEASE_MANIFEST.md` for the current specification, "
        "results, and verification gates.*\n").splitlines(keepends=True)
    with open(NB4, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    applied.append("nb04 cell 1")
    print("[patched] nb04 cell 1 markdown replaced")
else:
    sys.exit("REFUSED: nb04 cell 1 not in expected state - check manually.")

# --- 2. notebook 05: clear outputs, prepend SUPERSEDED banner ---------------
NB5 = "notebooks/05_top10_sensitivity.ipynb"
with open(NB5, encoding="utf-8") as f:
    nb = json.load(f)
first = "".join(nb["cells"][0]["source"]) if nb["cells"] else ""
changed = False
n_cleared = 0
for c in nb["cells"]:
    if c.get("cell_type") == "code" and c.get("outputs"):
        c["outputs"] = []
        c["execution_count"] = None
        n_cleared += 1
        changed = True
if MARK05 not in first:
    banner = {
        "cell_type": "markdown", "metadata": {},
        "source": (
            f"# {MARK05}\n\n"
            "This notebook is permanently superseded by "
            "`run_reduced_model_and_figures.py` (25 Aug 2026) and blocked "
            "by an unconditional guard cell. Its historical code reflects "
            "the pre-correction pipeline (112 features, utilization "
            "variables, the fixed obstruction indicator) and its outputs "
            "have been cleared so no superseded numbers render here. The "
            "analysis of record is run `tuning_results_20260831_103201`; "
            "see the README and `outputs/RELEASE_MANIFEST.md`.\n"
        ).splitlines(keepends=True),
    }
    nb["cells"].insert(0, banner)
    changed = True
if changed:
    with open(NB5, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    applied.append(f"nb05 ({n_cleared} outputs cleared, banner ensured)")
    print(f"[patched] nb05: {n_cleared} cell outputs cleared, banner ensured")
else:
    print("[already applied] nb05")

print(f"\nDone: {applied if applied else 'nothing to do'}")
