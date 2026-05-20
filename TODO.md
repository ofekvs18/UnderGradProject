## TODO 1: k feature biomarker comparison
For all methods i need to evaluate the formulas based on the number of features in each formula and in the baseline. A formula is considered "better" if for the same amount of feature used it gains a better AUC-PR. 
These are the goals I see fitting for the implementation:
- Update master summary file for all methods - add new feature amount column, verify it compares to correspondent baseline and not the very best formula across all feature counts.
- For each method make the script create enough formulas for each feature count (1-total number of CBC features)

**Questions:**
1. Should the baseline be a logistic regression trained separately for each feature count k (i.e., best LR with exactly k features), or is there a single fixed baseline per k already defined somewhere?
2. When you say "enough formulas for each feature count" — is there a minimum number of formulas per k you have in mind (e.g., at least N candidates before selecting the best), or should I match the current total formula budget spread evenly across k values?
3. For methods like method3_gp and method4_llm that have formula generation logic, should the feature count constraint be enforced strictly (exactly k features) or as an upper bound (at most k features)?
4. Should the master summary comparison column store the baseline AUC-PR value itself, or a delta/ratio vs. the baseline?

## TODO 2: EHRSHOT comparison
To verify actual accuracy for these formulas i need to test them on other data. 
Goals for this task:
- data creation script from ehrshot dataset into /data csv files (the script should be fully parameterized just like src\run_pipeline.py to allow running on new diseases), if you need you can add configs where necessary
- sanity check script exporting base data on the ehrhots data including prevalence, percentage of CBC data and other descriptive measures on the data
- evaluation script for each method, which gets the master summary file and runs the formulas on the EHRSHOTS dataset
- all newly created data and evaluations are stored on CSV files at appropriate locations

**Questions:**
1. Do you already have access to the EHRSHOT dataset locally, or does the data creation script need to handle download/access logic (e.g., from PhysioNet or a specific path)?
2. Should the EHRSHOT data creation script extract the same CBC columns as MIMIC-IV (hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc), or do EHRSHOT column names differ and need mapping?
3. For the evaluation script — should it re-use the same metrics as the main pipeline (AUC-ROC, AUC-PR, F2, precision@recall thresholds), or is a subset sufficient for external validation?
4. Should the evaluation results be stored alongside the existing `results/` method subdirectories, or in a separate `results/ehrshot/` hierarchy?
5. Is the disease parameterization via ICD codes, SNOMED codes, or some EHRSHOT-specific label format?

### Implementation Summary

**Decisions made (based on unanswered questions):**
1. **EHRSHOT access**: Script accepts `--ehrshot-dir` pointing to the local EHRSHOT dataset. No download logic; run extraction once you have the data locally.
2. **Column mapping**: Same 9 CBC features as MIMIC-IV. EHRSHOT uses OMOP/LOINC concept codes — a mapping config was added (`conf/ehrshot.yaml`). OMOP concept IDs need verification against your local EHRSHOT vocabulary before first run.
3. **Metrics**: Full metric set identical to MIMIC pipeline (AUC-ROC, AUC-PR, F2, precision@recall 0.25/0.50/0.75). Threshold fitted on EHRSHOT train split via Youden's index, applied to test split.
4. **Storage**: `results/ehrshot/` hierarchy.
5. **Disease parameterization**: ICD codes from `conf/disease/{slug}.yaml` (same configs as MIMIC pipeline), mapped to EHRSHOT's `ICD9CM/` or `ICD10CM/` MEDS code format.

**Files created:**
- `conf/ehrshot.yaml` — EHRSHOT config: CBC concept codes (OMOP + LOINC), ICD prefix format, lookback window, split seed
- `src/ehrshot_data.py` — Extracts modeling CSV from EHRSHOT MEDS parquet format; auto-detects parquet layout; assigns case/control labels from ICD codes; outputs `data/{disease}_ehrshot_data.csv`
- `src/ehrshot_sanity.py` — Descriptive stats on the extracted data: overview, per-feature coverage, feature stats, case-vs-control means; outputs to `results/ehrshot/`
- `src/ehrshot_evaluate.py` — Evaluates formulas from all four method master summaries on EHRSHOT; handles all formula formats (infix, GP prefix notation, threshold comparisons); outputs per-method and combined CSV to `results/ehrshot/`

**Usage:**
```bash
python src/ehrshot_data.py --ehrshot-dir /path/to/ehrshot --disease ra
python src/ehrshot_sanity.py --disease ra
python src/ehrshot_evaluate.py --disease ra            # all methods
python src/ehrshot_evaluate.py --disease ra --method m2  # single method
```

**Commits:**
- `feat(#2): add EHRSHOT data extraction, sanity check, and evaluation scripts`

## TODO 3: NHANES comparison
Same as TODO 2, only a different dataset

**Questions:**
1. Should the NHANES data creation script follow the exact same interface/parameterization as the EHRSHOT one from TODO 2, so both external-validation scripts are interchangeable?
2. NHANES uses SEQN-based identifiers and survey weights — should the evaluation account for survey weights when computing metrics, or treat each row equally like the MIMIC-IV pipeline?
3. Which NHANES cycles/years should be included by default (e.g., 1999–2018)?
4. NHANES CBC variables use different naming conventions (e.g., LBXWBCSI for WBC) — should the mapping be hardcoded in utils.py or in a per-dataset config file?

### Implementation Summary

**Decisions made (based on unanswered questions):**
1. **Interface**: Identical CLI pattern to EHRSHOT — `--nhanes-dir`, `--disease`, `--output` — so both scripts are interchangeable.
2. **Survey weights**: Not applied; each participant treated equally (matches MIMIC-IV pipeline for comparability across datasets).
3. **Cycles**: Default cycles G–J (2011–2018). Script accepts `--cycles G H I J` to override. Earlier cycles were excluded because NHANES CBC variable names are consistent from cycle G onward; pre-2011 data uses some different variable names requiring extra mapping.
4. **Variable mapping**: Stored in `conf/nhanes.yaml` (per-dataset config, same pattern as EHRSHOT). Disease case definitions (questionnaire variables and response codes) are also in that file with one entry per disease slug.

**Key NHANES-specific design decisions:**
- NHANES uses SEQN as patient identifier (renamed to `subject_id` in output).
- CBC data comes from SAS XPT files (`CBC_<cycle>.XPT`). Both flat (`nhanes_dir/CBC_H.XPT`) and cycle-subdirectory (`nhanes_dir/H/CBC_H.XPT`) layouts are auto-detected.
- Disease labels come from questionnaire XPT files (MCQ, DIQ), not ICD codes. Case definitions are AND-ed conditions per disease (e.g., RA requires MCQ160a==1 AND MCQ195==2).
- Participants appearing in multiple cycles retain the CBC row with fewest missing values; later cycle is tiebreaker.
- Output CSV format is identical to MIMIC-IV and EHRSHOT: `subject_id, is_case, split, <features>`.

**Files created:**
- `conf/nhanes.yaml` — NHANES config: CBC variable names (LBXWBCSI etc.), survey cycles, questionnaire-based disease case definitions per disease slug
- `src/nhanes_data.py` — Extracts modeling CSV from NHANES XPT files; auto-detects flat/subdir layout; assigns case labels from questionnaire variables; outputs `data/{disease}_nhanes_data.csv`
- `src/nhanes_sanity.py` — Descriptive stats: overview, per-feature coverage, feature stats, case-vs-control means; outputs to `results/nhanes/`
- `src/nhanes_evaluate.py` — Evaluates formulas from all four method master summaries on NHANES; handles all formula formats; outputs per-method and combined CSV to `results/nhanes/`

**Usage:**
```bash
python src/nhanes_data.py --nhanes-dir /path/to/nhanes --disease ra
python src/nhanes_sanity.py --disease ra
python src/nhanes_evaluate.py --disease ra            # all methods
python src/nhanes_evaluate.py --disease ra --method m2  # single method
```

**Commits:**
- `feat(#3): add NHANES data extraction, sanity check, and evaluation scripts`

## TODO 4: results visualization
this is only a planning todo, nothing to implement yet.
i need the results in a convenient format, where all formulas are of the same style and format no matter which method has created them. i need to be able to visualize the results and filtering it based on disease, methods, feature count. I think I'm imagining some sort of dashboard to do it so i could also screenshot it for the article. 
create a mockup for this, and a step by step implementation guide with concrete goals for each step. 

**Questions:**
1. Should the dashboard be a static HTML file (e.g., built with Plotly/Dash or a single-page export) so it can be shared without a running server, or is a local web server (e.g., Streamlit, Dash) acceptable?
2. Which metrics should be front-and-center in the visualization — AUC-PR only, or a multi-metric view (AUC-ROC, F2, precision@recall)?
3. Should the formula display normalize notation across methods (e.g., always show as `a*feature1 + b*feature2 > threshold`), and if so, is there a preferred canonical form?
4. Do you need the dashboard to also display external validation results (EHRSHOT, NHANES) alongside MIMIC-IV results once those are done?

### Implementation Summary

**Decisions made (answering open questions):**
1. **Stack**: Local **Streamlit + Plotly** app (`streamlit run src/dashboard.py`). Chosen over static HTML because filter interactivity requires state; Plotly's `fig.write_image()` + kaleido gives 300 DPI static PNG export for the article without needing a browser — best of both.
2. **Metrics**: Multi-metric view. AUC-PR is primary (sorted by default), AUC-ROC, F2, and precision@recall(0.25/0.50/0.75) are all shown in the detail panel and summary cards. Bar charts and PR curves use the user-selected primary metric.
3. **Formula normalization**: Two fields per formula: `formula_display` (clean human-readable infix — strips safety epsilons, converts `**` to superscripts, converts GP S-expression prefix to infix) and `formula_raw` (original, preserved exactly). Dashboard shows `formula_display` in the table, exposes `formula_raw` in a collapsible detail panel.
4. **External validation**: Built-in opt-in toggle. `build_dashboard_data.py --include-external` merges EHRSHOT/NHANES rows with a `dataset` column. Dashboard sidebar shows a Dataset filter only when external rows are present — degrades gracefully when external data is absent.

**Files created:**
- `docs/dashboard_mockup.html` — Full-fidelity HTML mockup: sidebar filters (disease, method, feature-count slider, metric selector), summary stat cards, grouped bar chart, PR curve overlay, sortable formula table with normalized formulas, and a selected-formula detail panel
- `docs/dashboard_implementation_guide.md` — 5-step implementation guide: (1) formula normalizer + aggregator → `dashboard_data.csv`, (2) Streamlit app, (3) precomputed PR curve data, (4) static PNG export for article, (5) external validation toggle

**Commits:**
- `docs(#4): add dashboard mockup and implementation guide`

## TODO 5: Confidence intervals
This is only a planning todo, nothing to implement yet.
To show this research is actually good i need to provide confidence intervals for each metric that is considered important. Questions to answer:
1. What data should i create the CI on? 
2. On which metrics do we need to provide this?
3. how to Visualize it?
Research what is best for this, show me your thinking and decision making process then will plan the implementation steps

**Questions:**
1. Should the CI analysis cover only the test split, or also report CIs on the train split for comparison?
2. Is bootstrap resampling the expected approach, or are you open to analytical CIs (e.g., DeLong for AUC-ROC) where they exist?
3. Should CIs be computed per-method for the best formula, or for every formula in the master summary?

### Implementation Summary

**Thinking and decision-making process:**

**Q1 — What data to create CIs on?**
The pipeline has three layers of data: the fixed MIMIC-IV test split (~20% of patients), and two external datasets (EHRSHOT, NHANES). The train split was used for formula selection/threshold fitting, so CIs on train would reflect in-sample noise — not useful for a paper. The test split is the right target.

Approach chosen: **stratified bootstrap on the test split** (stratified is critical because ~1% positive rate — non-stratified bootstrap can yield bootstrap folds with zero positives). 2 000 resamples with `random_state=42`, percentile method for the CI bounds (2.5th / 97.5th).

Cross-validation CIs (already in `sanity_check.py`) are useful for the LR baseline only — they are not applicable to search-based methods (M2/M3) or LLM methods (M4) where "training" is formula search, not a refittable model.

External datasets (EHRSHOT, NHANES) provide cross-dataset validation, not bootstrap CIs in the classical sense. Their point estimates plus the MIMIC CIs already give a strong generalizability story.

**Q2 — Which metrics need CIs?**
- **AUC-PR** (primary metric for imbalanced data): no analytical formula → bootstrap. Most important to report.
- **AUC-ROC**: analytical CI available via DeLong's method (fast, exact under the nonparametric assumption). Fallback to bootstrap if fewer than 30 positives in the resample.
- **F2**: threshold-dependent → threshold is refitted per bootstrap resample on bootstrap train rows, applied to bootstrap test rows. Adds computational cost but is methodologically correct.
- **Precision@recall(0.50)**: most clinically meaningful fixed-recall operating point → bootstrap.

Metrics left out of CIs: precision, recall, F1 (all captured by the four above), AUC-PR on external datasets (computed as point estimates only — those datasets have different populations).

**Q3 — How to visualize?**
Three complementary views:
1. **Forest plot** — primary figure for the paper. One row per method, AUC-PR on x-axis, CI as horizontal whisker. Clean, standard in clinical ML.
2. **Error bars on bar chart** — already in the dashboard plan (TODO 4). CI columns flow naturally into the existing Plotly bars.
3. **CI columns in master summary CSVs** — machine-readable, feeds both the dashboard and the forest plot.

Shaded PR curve bands were considered but rejected: they require storing B×|test| score arrays (too large) and are harder to read than the forest plot in a paper.

**Concrete implementation steps:**

**Step 1 — `src/compute_ci.py` (new utility module)**
- `stratified_bootstrap_indices(y_true, n_boot, seed)` → generator of (train_idx, test_idx) arrays that preserve positive rate
- `bootstrap_ci(y_true, scores, metric_fn, n_boot=2000, alpha=0.05, seed=42)` → `(lo, hi, point)` — generic, reusable
- Pre-built wrappers exported from the module: `ci_auc_pr`, `ci_auc_roc`, `ci_f2`, `ci_precision_at_recall`
- `delong_auc_roc_ci(y_true, scores, alpha=0.05)` → `(lo, hi)` — analytical fast path for AUC-ROC

**Step 2 — `src/compute_ci_all.py` (standalone backfill script)**
Reads existing master summary CSVs for all methods + sanity_check, loads the test split, runs bootstrap for each row's formula, appends CI columns. This lets you add CIs to past runs without re-running experiments. CLI: `python src/compute_ci_all.py --disease ra`.

Columns added per method: `AUC_PR_CI_Low`, `AUC_PR_CI_High`, `AUC_ROC_CI_Low`, `AUC_ROC_CI_High`, `F2_CI_Low`, `F2_CI_High`, `P_at_R50_CI_Low`, `P_at_R50_CI_High`, `CI_N_Boot`, `CI_Seed`.

**Step 3 — Forward integration into method scripts**
After `compute_ci_all.py` is validated, add a CI computation block at the end of each method script (after the best formula is selected) so future runs write CI columns directly. Affected files: `method_threshold.py`, `method2_random_formula.py`, `method3_gp.py`, `method4_llm.py`, `sanity_check.py`.

**Step 4 — Forest plot script (`src/plot_ci_forest.py`)**
Reads all master summaries, takes the best formula per method (highest AUC-PR), draws a Matplotlib forest plot: methods on y-axis, AUC-PR + 95% CI on x-axis, vertical dashed line at LR all-features baseline. Outputs `results/figures/ci_forest_auc_pr.png` at 300 DPI.

**Step 5 — Dashboard integration**
Update `src/dashboard.py` (TODO 4 Step 2) to read CI columns when present and pass them as `error_x` to Plotly bar traces. CI display is opt-in via a sidebar toggle — degrades gracefully when CI columns are absent.

**Files to create:**
- `src/compute_ci.py` — CI utility functions (DeLong + stratified bootstrap)
- `src/compute_ci_all.py` — backfill script for existing master summaries
- `src/plot_ci_forest.py` — forest plot generator for paper figures

**Files to modify:**
- `src/method_threshold.py`, `src/method2_random_formula.py`, `src/method3_gp.py`, `src/method4_llm.py`, `src/sanity_check.py` — add CI block after best-formula selection
- `src/dashboard.py` — add error bars (after TODO 4 Step 2)

**Commits:**
- `docs(#5): add confidence interval planning and implementation guide`
