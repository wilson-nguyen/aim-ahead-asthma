"""
patch12_nb04_header.py — replace notebook 04's stale header markdown.

The header still described the original "full faithful port" and Model A/B
framing from before the R3 corrections (external review, 31 Aug). This
patches cell 0's markdown only; no code cell and no stored output changes,
so the executed analysis-of-record outputs remain untouched.

Guarded and idempotent. Run from the repo root:
    python patch12_nb04_header.py
"""
import json
import sys

NB = "notebooks/04_model.ipynb"
MARKER = "Analysis of record: `tuning_results_20260831_103201`"

NEW_HEADER = f"""# Phase 4: Modeling — R3 analysis of record

{MARKER}. This notebook performs feature engineering, Optuna tuning, and
model fitting for the corrected R3 specification: spirometry quality gating
(grades A/B), pre-specified exclusions, categorical-aware SMOTENC-ENN
resampling, and unweighted fitting with survey-weighted evaluation reported
alongside. The authoritative description of the pipeline, its verification
gates, and its disclosed limitations is the repository README and
`outputs/RELEASE_MANIFEST.md`.

Notes:
- Cell 3 is the ARCHIVED pre-R3 exploratory leaderboard (raw cell, not
  executed) kept for provenance; the production path is the standalone
  tuning cell below it.
- Re-executing this notebook creates a NEW timestamped run: the parallel
  Optuna search is not bit-reproducible, so a re-execution is a new
  analysis, not a reproduction of the analysis of record.
- Evaluation happens in the run scripts (`run_final_analyses.py` and
  successors), each pinned and gated on the split-verification report.
"""

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

cell0 = nb["cells"][0]
src = "".join(cell0["source"])
if MARKER in src:
    print("[already applied] nb04 header")
    sys.exit(0)
if cell0["cell_type"] != "markdown" or "Phase 4" not in src:
    sys.exit("REFUSED: cell 0 is not the expected header - check manually.")

cell0["source"] = NEW_HEADER.splitlines(keepends=True)
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("[patched] nb04 header markdown replaced (code and outputs untouched)")
