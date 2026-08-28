"""
patch11_r3_quality_gating.py — apply the 2026-08-27 KM rulings to the notebooks.

Khamron's email of 27 Aug 2026:
  1. Spirometry quality gating: primary analysis uses best-test quality
     grades A/B only (SPXNQFV1/SPXNQFVC); A/B/C is the pre-declared
     sensitivity arm; grade variables and curve counts are QC only.
  2. URDNALLC excluded (below-LOD comment flag; LOD varied within cycles).
     [Implemented in asthma_pipeline.py, no notebook change needed.]

What this patches:
  - notebooks/03_clean_and_filter.ipynb (cell 2): RETAIN SPXNQFV1/SPXNQFVC
    in 03_cleaned.parquet (they were being dropped with the other quality
    attributes). Row membership is unchanged, so the split is identical.
  - notebooks/04_model.ipynb (cell 2): apply A/B quality gating right after
    loading 03_cleaned, before any modeling; grade columns are dropped there.
  - notebooks/04_model.ipynb (cell 6): import PRIMARY_ALLOWED_GRADES and
    record the gating rule in both saved config dicts.

Guarded and idempotent: refuses to run if a cell is in an unknown state;
does nothing if a change is already applied. Run from the repo root:
    python patch11_r3_quality_gating.py
Then re-run notebook 03 and notebook 04.
"""
import json
import sys

CHANGES_APPLIED = []
PROBLEMS = []


def patch_cell(nb_path, cell_idx, old, new, label):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    src = "".join(nb["cells"][cell_idx]["source"])
    if new in src:
        print(f"  [already applied] {label}")
        return
    if old not in src:
        PROBLEMS.append(f"{label}: anchor text not found in {nb_path} cell "
                        f"{cell_idx} — cell is in an unknown state, NOT patched")
        print(f"  [REFUSED] {label}")
        return
    src = src.replace(old, new, 1)
    nb["cells"][cell_idx]["source"] = src.splitlines(keepends=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    CHANGES_APPLIED.append(label)
    print(f"  [patched] {label}")


NB03 = "notebooks/03_clean_and_filter.ipynb"
NB04 = "notebooks/04_model.ipynb"

print("Patch 11 — spirometry quality gating (KM ruling 2026-08-27)")

# --- 1. nb03 cell 2: retain the two best-test grade columns ---------------
patch_cell(
    NB03, 2,
    "Spirometry_Quality_Variables = ['SPXNQEFF', 'SPXBQEFF', 'SPXNQFV1', "
    "'SPXBQFV1', 'SPXBQFVC', 'SPXNQFVC']",
    "# [2026-08-27 KM ruling] SPXNQFV1/SPXNQFVC RETAINED as QC metadata for\n"
    "# spirometry quality gating in Phase 4 (primary analysis: grades A/B\n"
    "# usable). Bronchodilator/efficiency attributes still dropped.\n"
    "Spirometry_Quality_Variables = ['SPXNQEFF', 'SPXBQEFF', 'SPXBQFV1', "
    "'SPXBQFVC']",
    "nb03 cell 2: retain SPXNQFV1/SPXNQFVC",
)

# --- 2. nb04 cell 2: gate right after loading 03_cleaned ------------------
patch_cell(
    NB04, 2,
    "# Sanity check: report any remaining non-numeric columns "
    "so we can spot issues early",
    "# [2026-08-27 KM ruling] Spirometry quality gating: primary analysis\n"
    "# uses best-test grades A/B only; C/D/F and ungraded are treated as not\n"
    "# measured (captured by the availability indicators). The grade columns\n"
    "# are consumed here (QC only) and dropped before modeling.\n"
    "from asthma_pipeline import apply_spirometry_quality_gating, "
    "PRIMARY_ALLOWED_GRADES\n"
    "split_df = apply_spirometry_quality_gating(split_df, "
    "allowed_grades=PRIMARY_ALLOWED_GRADES)\n"
    "\n"
    "# Sanity check: report any remaining non-numeric columns "
    "so we can spot issues early",
    "nb04 cell 2: apply A/B quality gating after load",
)

# --- 3. nb04 cell 6: import the constant, record the rule in both configs -
patch_cell(
    NB04, 6,
    "    PRIMARY_MODEL_EXCLUSIONS, PROTECTED_FEATURES,\n)",
    "    PRIMARY_MODEL_EXCLUSIONS, PROTECTED_FEATURES, PRIMARY_ALLOWED_GRADES,\n)",
    "nb04 cell 6: import PRIMARY_ALLOWED_GRADES",
)
patch_cell(
    NB04, 6,
    "    'primary_model_exclusions': PRIMARY_MODEL_EXCLUSIONS,\n"
    "    'protected_features': PROTECTED_FEATURES,",
    "    'primary_model_exclusions': PRIMARY_MODEL_EXCLUSIONS,\n"
    "    'spirometry_quality_gating': {'allowed_grades': "
    "list(PRIMARY_ALLOWED_GRADES),\n"
    "                                 'ruling': 'KM 2026-08-27; A/B/C is the "
    "pre-declared sensitivity arm'},\n"
    "    'protected_features': PROTECTED_FEATURES,",
    "nb04 cell 6: record gating in preprocessed-data config",
)
patch_cell(
    NB04, 6,
    "        'primary_model_exclusions': PRIMARY_MODEL_EXCLUSIONS,\n"
    "        'protected_features': PROTECTED_FEATURES,",
    "        'primary_model_exclusions': PRIMARY_MODEL_EXCLUSIONS,\n"
    "        'spirometry_quality_gating': {'allowed_grades': "
    "list(PRIMARY_ALLOWED_GRADES),\n"
    "                                     'ruling': 'KM 2026-08-27'},\n"
    "        'protected_features': PROTECTED_FEATURES,",
    "nb04 cell 6: record gating in results config",
)

print()
if PROBLEMS:
    print("PROBLEMS — fix before re-running:")
    for p in PROBLEMS:
        print(f"  - {p}")
    sys.exit(1)
print(f"Done: {len(CHANGES_APPLIED)} change(s) applied, "
      f"{5 - len(CHANGES_APPLIED)} already in place.")
print("Next: re-run notebook 03, then notebook 04 (production settings).")
