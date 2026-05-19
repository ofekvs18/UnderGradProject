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
