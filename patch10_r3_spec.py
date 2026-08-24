"""
patch10_r3_spec.py — apply the locked R3 specification to notebooks 04 and 05.

Guarded: every splice asserts the expected content is at the expected place
first; on any mismatch the script aborts WITHOUT writing. Writes are atomic
(temp file + validate + os.replace) with timestamped backups in archive/.

Changes to 04_model.ipynb
  cell 3  -> archived (converted to raw; exploratory leaderboard, not the
             reproducible path)
  cell 6  -> imports shared components from asthma_pipeline.py;
             local class definitions removed;
             ENQ020 / SPDBRONC / HUQ050 excluded from the primary model;
             correlation pruning executed on training data;
             imputer + scaler moved INSIDE each model pipeline (fold-safe);
             ProtectedSelectKBest keeps spirometry-availability indicators;
             mutual_info selection seeded;
             feature names persisted with every artifact;
             weighted validation metrics written to JSON.

Changes to 05_top10_sensitivity.ipynb
  adult-threshold BMI categories and one fillna(0) interaction commented out;
  stale-notice banner cell inserted (must be re-run against corrected 04
  artifacts).

Run:  python patch10_r3_spec.py
"""
import json
import os
import shutil
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "notebooks")
ARCHIVE = os.path.join(HERE, "archive")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

NB04 = os.path.join(NB_DIR, "04_model.ipynb")
NB05 = os.path.join(NB_DIR, "05_top10_sensitivity.ipynb")

failures = []


def guard(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"  GUARD FAILED: {msg}")
    return cond


def load_nb(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell_lines(nb, idx):
    return "".join(nb["cells"][idx]["source"]).split("\n")


def set_cell_source(nb, idx, lines):
    src = [l + "\n" for l in lines[:-1]]
    if lines:
        src.append(lines[-1])
    nb["cells"][idx]["source"] = src


def atomic_write(nb, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(tmp, encoding="utf-8") as f:
        reloaded = json.load(f)          # validate JSON round-trip
    assert len(reloaded["cells"]) == len(nb["cells"])
    os.replace(tmp, path)


# ===========================================================================
# Notebook 04
# ===========================================================================

nb4 = load_nb(NB04)
guard(len(nb4["cells"]) == 9, f"nb04 expected 9 cells, found {len(nb4['cells'])}")

# --- cell 3: archive (convert to raw) --------------------------------------
c3 = nb4["cells"][3]
c3_src = "".join(c3["source"])
guard("COMPREHENSIVE PEDIATRIC ASTHMA SCREENING PIPELINE" in c3_src,
      "cell 3 is not the leaderboard pipeline")
banner = (
    "# =========================================================================\n"
    "# ARCHIVED 2026-08-23 (R3 revision) — exploratory leaderboard pipeline.\n"
    "# NOT part of the reproducible analysis path. The production path is the\n"
    "# standalone tuning cell below, which imports shared components from\n"
    "# asthma_pipeline.py. Content preserved verbatim for the audit trail.\n"
    "# =========================================================================\n"
)
c3["cell_type"] = "raw"
c3.pop("outputs", None)
c3.pop("execution_count", None)
c3["metadata"] = {"tags": ["archived-r3"]}
c3["source"] = [banner] + c3["source"]

# --- cell 6: the production pipeline ---------------------------------------
L = cell_lines(nb4, 6)
guard(len(L) == 848, f"cell 6 expected 848 lines, found {len(L)}")
guard(L[0].startswith("# [R3 corrections]"), "cell6 line0")
guard(L[46].startswith("LEAKY_PROXIES"), "cell6 line46 LEAKY_PROXIES")
guard(L[55].startswith("N_TRIALS"), "cell6 line55 N_TRIALS")
guard(L[84].startswith("class NHANESCleaner"), "cell6 line84 NHANESCleaner")
guard(L[146].startswith("class ClinicalFeatureEngineer"), "cell6 line146 CFE")
guard(L[210].strip().startswith("return self.feature_names_"), "cell6 line210")
guard(L[262].startswith("# Remove leaky/restricted"), "cell6 line262")
guard(L[294].startswith("# Preprocessing pipeline"), "cell6 line294")
guard(L[334].strip().endswith("preprocessed_data.pkl')"), "cell6 line334")
guard(L[365].strip().startswith("pipeline = ImbPipeline"), "cell6 line365 cb-obj")
guard("X_train_final" in L[375], "cell6 line375 cb cv X")
guard(L[421].strip().startswith("final_pipeline_cb = ImbPipeline"), "cell6 line421")
guard("final_pipeline_cb.fit(X_train_final" in L[427], "cell6 line427")
guard(L[469].strip().startswith("pipeline = ImbPipeline"), "cell6 line469 mlp-obj")
guard("X_train_final" in L[480], "cell6 line480 mlp cv X")
guard(L[547].strip().startswith("final_pipeline_mlp = ImbPipeline"), "cell6 line547")
guard("final_pipeline_mlp.fit(X_train_final" in L[555], "cell6 line555")
guard(L[587].strip().startswith("pipeline = ImbPipeline"), "cell6 line587 rf-obj")
guard("X_train_final" in L[597], "cell6 line597 rf cv X")
guard(L[646].strip().startswith("final_pipeline_rf = ImbPipeline"), "cell6 line646")
guard("final_pipeline_rf.fit(X_train_final" in L[652], "cell6 line652")
guard(L[659].startswith("# ====="), "cell6 line659 viz header")

if failures:
    sys.exit(f"\nAborted before any write: {len(failures)} guard(s) failed.")

IND = "    "  # notebook uses 4-space indent inside functions


def obj_pipeline(selector_call, classifier_line):
    return [
        f"{IND}pipeline = ImbPipeline(preprocessing_steps() + [",
        f"{IND}    ('feature_selection', {selector_call}),",
        f"{IND}    ('smote_enn', SMOTEENN(random_state=RANDOM_STATE)),",
        classifier_line,
        f"{IND}])",
    ]


def final_pipeline(varname, selector_call, classifier_line, extra=None):
    body = [
        f"{IND}{varname} = ImbPipeline(preprocessing_steps() + [",
        f"{IND}    ('feature_selection', {selector_call}),",
        f"{IND}    ('smote_enn', SMOTEENN(random_state=RANDOM_STATE)),",
    ]
    if extra:
        body.append(extra)
    body += [classifier_line, f"{IND}])"]
    return body


SEL_F = ("ProtectedSelectKBest(f_classif, k=20, feature_names=FEATURE_NAMES, "
         "protect=PROTECTED_FEATURES)")
SEL_MI = ("ProtectedSelectKBest(mutual_info_seeded, k=20, feature_names=FEATURE_NAMES, "
          "protect=PROTECTED_FEATURES)")

# ---- splices, bottom-up so indices stay valid -----------------------------

# (13) weighted validation metrics + artifact bundles, inserted before viz
weighted_block = [
    "# ===========================================================================",
    "# [R3] WEIGHTED VALIDATION METRICS + FEATURE-NAME BUNDLES",
    "# Fitting stays unweighted pending the coauthor decision on survey weights;",
    "# weighted evaluation is reported so the decision can be made with numbers.",
    "# Validation set only — the test set is not touched here.",
    "# ===========================================================================",
    "",
    "final_models = {}",
    "if not CATBOOST_FAILED:",
    "    final_models['catboost'] = final_pipeline_cb",
    "if not MLP_FAILED:",
    "    final_models['mlp'] = final_pipeline_mlp",
    "if not BALANCEDRF_FAILED:",
    "    final_models['balancedrf'] = final_pipeline_rf",
    "",
    "val_metrics = {}",
    "for name, mdl in final_models.items():",
    "    sel = mdl.named_steps['feature_selection']",
    "    bundle = {",
    "        'model': name,",
    "        'feature_names_in': FEATURE_NAMES,",
    "        'selected_features': sel.get_selected_names(),",
    "        'pruned_features': PRUNED_FEATURES,",
    "        'primary_model_exclusions': PRIMARY_MODEL_EXCLUSIONS,",
    "        'protected_features': PROTECTED_FEATURES,",
    "        'random_state': RANDOM_STATE,",
    "        'run_dir': OUTPUT_DIR,",
    "    }",
    "    with open(f'{OUTPUT_DIR}/{name}_feature_bundle.json', 'w') as fh:",
    "        json.dump(bundle, fh, indent=2)",
    "",
    "    prob_val = mdl.predict_proba(X_val_feat)[:, 1]",
    "    val_metrics[name] = {",
    "        'unweighted': weighted_binary_metrics(y_val, prob_val, None),",
    "        'survey_weighted': (weighted_binary_metrics(y_val, prob_val, sw_val)",
    "                            if sw_val is not None else None),",
    "    }",
    "    uw = val_metrics[name]['unweighted']; sw = val_metrics[name]['survey_weighted']",
    "    print(f\"{name}: val AUC unweighted {uw['auc']:.3f}\" +",
    "          (f\" | survey-weighted {sw['auc']:.3f}\" if sw else ''))",
    "",
    "with open(f'{OUTPUT_DIR}/weighted_validation_metrics.json', 'w') as fh:",
    "    json.dump(val_metrics, fh, indent=2)",
    "print(f\"\\n[R3] Bundles + weighted validation metrics written to {OUTPUT_DIR}/\")",
    "",
]
L[659:659] = weighted_block

# (12) BalancedRF final: pipeline block 646-650, fit line 652
L[652] = L[652].replace("X_train_final", "X_train_feat")
L[646:651] = final_pipeline(
    "final_pipeline_rf", SEL_F,
    f"{IND}    ('classifier', BalancedRandomForestClassifier(**best_params_rf))")

# (11) BalancedRF objective: cv X line 597, pipeline 587-591
L[597] = L[597].replace("X_train_final", "X_train_feat")
L[587:592] = obj_pipeline(
    SEL_F, f"{IND}    ('classifier', BalancedRandomForestClassifier(**params))")

# (10) MLP final: fit 555, pipeline 547-552 (keeps its extra StandardScaler)
L[555] = L[555].replace("X_train_final", "X_train_feat")
L[547:553] = final_pipeline(
    "final_pipeline_mlp", SEL_MI,
    f"{IND}    ('classifier', MLPClassifier(**final_params_mlp))",
    extra=f"{IND}    ('mlp_scaler', StandardScaler()),")

# (9) MLP objective: cv X 480, pipeline 469-474 (6 lines incl. its scaler)
L[480] = L[480].replace("X_train_final", "X_train_feat")
L[469:475] = [
    f"{IND}pipeline = ImbPipeline(preprocessing_steps() + [",
    f"{IND}    ('feature_selection', {SEL_MI}),",
    f"{IND}    ('smote_enn', SMOTEENN(random_state=RANDOM_STATE)),",
    f"{IND}    ('mlp_scaler', StandardScaler()),",
    f"{IND}    ('classifier', MLPClassifier(**params))",
    f"{IND}])",
]

# (8) CatBoost final: fit 427, pipeline 421-425
L[427] = L[427].replace("X_train_final", "X_train_feat")
L[421:426] = final_pipeline(
    "final_pipeline_cb", SEL_F,
    f"{IND}    ('classifier', CatBoostClassifier(**best_params_cb))")

# (7) CatBoost objective: cv X 375, pipeline 365-369
L[375] = L[375].replace("X_train_final", "X_train_feat")
L[365:370] = obj_pipeline(
    SEL_F, f"{IND}    ('classifier', CatBoostClassifier(**params))")

# (6) preprocessing block 294-334 -> R3 version
new_preprocessing = [
    "# Preprocessing — R3: imputation and scaling now live INSIDE each model",
    "# pipeline (fold-safe). Here we only clean, engineer, and prune features.",
    "print(\"\\nApplying preprocessing (cleaning → feature engineering → pruning)...\")",
    "cleaner = NHANESCleaner()",
    "feat_eng = ClinicalFeatureEngineer()",
    "",
    "X_train_feat = feat_eng.fit_transform(cleaner.fit_transform(X_train))",
    "X_val_feat   = feat_eng.transform(cleaner.transform(X_val))",
    "X_test_feat  = feat_eng.transform(cleaner.transform(X_test))",
    "",
    "# [R3] correlation pruning (|r| > 0.90), fit on TRAINING data only",
    "X_train_feat, X_val_feat, X_test_feat, PRUNED_FEATURES = apply_correlation_pruning(",
    "    X_train_feat, X_val_feat, X_test_feat)",
    "print(f\"Correlation pruning dropped {len(PRUNED_FEATURES)}: {PRUNED_FEATURES}\")",
    "",
    "FEATURE_NAMES = X_train_feat.columns.tolist()",
    "print(f\"✓ Preprocessing complete! Features entering selection: {len(FEATURE_NAMES)}\")",
    "",
    "# Reference imputer/scaler fitted on the full training set — for artifact",
    "# compatibility and downstream notebooks. NOT used inside cross-validation.",
    "ref_imputer = SimpleImputer(strategy='median').fit(X_train_feat)",
    "ref_scaler = RobustScaler().fit(ref_imputer.transform(X_train_feat))",
    "",
    "joblib.dump({",
    "    'cleaner': cleaner,",
    "    'feat_eng': feat_eng,",
    "    'ref_imputer': ref_imputer,",
    "    'ref_scaler': ref_scaler,",
    "    'X_train_feat': X_train_feat,",
    "    'X_val_feat': X_val_feat,",
    "    'X_test_feat': X_test_feat,",
    "    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,",
    "    'sw_train': sw_train, 'sw_val': sw_val, 'sw_test': sw_test,",
    "    'feature_names': FEATURE_NAMES,",
    "    'pruned_features': PRUNED_FEATURES,",
    "    'primary_model_exclusions': PRIMARY_MODEL_EXCLUSIONS,",
    "    'protected_features': PROTECTED_FEATURES,",
    "    'note': 'PFQ020 retained per KM 2026-08-14; fitting unweighted pending '",
    "            'coauthor decision on survey weights.',",
    "}, f'{OUTPUT_DIR}/preprocessed_data.pkl')",
]
L[294:335] = new_preprocessing

# (5) drop block 262-269
L[262:270] = [
    "# Remove leaky/restricted columns + [R3] primary-model exclusions",
    "# (ENQ020, SPDBRONC: NHANES routing/eligibility; HUQ050: diagnostic-",
    "# opportunity proxy — returns only in the declared sensitivity analysis).",
    "# PFQ020 retained per Khamron 2026-08-14 (Yaseen: no response; logged).",
    "cols_to_drop = [c for c in X_clean.columns",
    "                if c in LEAKY_PROXIES + AGE_RESTRICTED_VARS + IDENTIFIERS",
    "                + PRIMARY_MODEL_EXCLUSIONS]",
    "print(f\"Dropping {len(cols_to_drop)} excluded columns: {sorted(cols_to_drop)}\")",
    "if cols_to_drop:",
    "    X_clean = X_clean.drop(columns=cols_to_drop)",
]

# (4) remove local class definitions 80-210 (headers + both classes)
L[80:211] = [
    "# ===========================================================================",
    "# CLEANING + FEATURE ENGINEERING — [R3] imported from asthma_pipeline.py",
    "# (single source of truth shared with notebook 05; local copies removed).",
    "# ===========================================================================",
]

# (3) stale comment on line 55
L[55] = "N_TRIALS = 100  # production setting"

# (2) exclusion lists 45-52 (comment + three lists) -> module note
L[45:53] = [
    "# [R3] Exclusion lists now live in asthma_pipeline.py (imported above):",
    "# LEAKY_PROXIES, AGE_RESTRICTED_VARS, IDENTIFIERS, PRIMARY_MODEL_EXCLUSIONS.",
]

# (1) header import
L[0:2] = [
    "# [R3 corrections] shared components — single source of truth",
    "from asthma_pipeline import (",
    "    NHANESCleaner, ClinicalFeatureEngineer, ProtectedSelectKBest,",
    "    preprocessing_steps, apply_correlation_pruning, weighted_binary_metrics,",
    "    mutual_info_seeded, LEAKY_PROXIES, AGE_RESTRICTED_VARS, IDENTIFIERS,",
    "    PRIMARY_MODEL_EXCLUSIONS, PROTECTED_FEATURES,",
    ")",
    "import json",
]

set_cell_source(nb4, 6, L)

# post-splice sanity: no live references to the old globals remain
joined = "\n".join(L)
for bad in ("X_train_final", "X_val_final", "X_test_final",
            "class NHANESCleaner", "class ClinicalFeatureEngineer",
            "('feature_selection', SelectKBest(",
            "IDENTIFIERS = ["):
    guard(bad not in joined, f"stale reference survived: {bad}")

# ===========================================================================
# Notebook 05 — minimal consistency fixes + stale banner
# ===========================================================================

nb5 = load_nb(NB05)
L5 = cell_lines(nb5, 2)
guard("X_df['underweight'] = (X_df['BMXBMI'] < 18.5)" in L5[213], "nb05 l213")
guard("fillna(0) * X_df['BMXBMI'].fillna(0)" in L5[276], "nb05 l276")

if failures:
    sys.exit(f"\nAborted before any write: {len(failures)} guard(s) failed.")

for i in range(213, 218):   # five adult-threshold category lines
    lead = L5[i][: len(L5[i]) - len(L5[i].lstrip())]
    L5[i] = f"{lead}# [R3 correction — removed, adult thresholds] {L5[i].strip()}"
lead = L5[276][: len(L5[276]) - len(L5[276].lstrip())]
L5[276] = (f"{lead}# [R3 correction — zero-fill removed] {L5[276].strip()}")
set_cell_source(nb5, 2, L5)

stale_banner = {
    "cell_type": "markdown",
    "metadata": {"tags": ["stale-r3"]},
    "source": [
        "> **STALE — do not re-run yet (2026-08-23).** This notebook was built "
        "against the pre-R3 artifacts. The predictor set has changed "
        "(HUQ050 and the 0.80 obstruction indicator are out of the primary "
        "model), so the top-10 list, SHAP figures, and reduced model must be "
        "rebuilt from the corrected `04_model.ipynb` production run. "
        "Re-work this notebook only after the new `tuning_results_*` "
        "directory exists.\n"
    ],
}
nb5["cells"].insert(1, stale_banner)

# ===========================================================================
# Write (backups first)
# ===========================================================================

os.makedirs(ARCHIVE, exist_ok=True)
for p in (NB04, NB05):
    shutil.copy2(p, os.path.join(ARCHIVE, f"{os.path.basename(p)}.bak-{STAMP}"))

atomic_write(nb4, NB04)
atomic_write(nb5, NB05)

print("OK: patched 04_model.ipynb and 05_top10_sensitivity.ipynb")
print(f"Backups: archive/*.bak-{STAMP}")
