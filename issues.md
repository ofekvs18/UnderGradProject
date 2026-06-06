# Issues — Biomarker Pipeline

## Table of Contents

| # | Title | Depends On | Status |
|---|-------|------------|--------|
| 24 | Nested CV for M2, M3, M4 — winner selection on train only | — | closed |
| 25 | Cross-method score vector correlation analysis | 24 | closed |
| 26 | Store LLM seed files and define naming standard | — | closed |
| 27 | Add `--seed-file` flag to `method3_gp.py` for seeded-GP warm start | 26 | closed |
| 28 | Evaluate seeded GP against vanilla GP on Crohn's disease | 27 | closed |
| 29 | Generalize seeded-GP to all remaining diseases | 28 | closed (negative result — no seeded run beat vanilla; see experiment_log.md) |
| **30** | **Expand feature set to 14 CBC features + full data purge and rerun** | — | **open** |

---

## Issue 24 — Nested CV for M2, M3, M4 — winner selection on train only

**Depends on:** —

CV currently exists only for the LR baseline. M2, M3, and M4 each pick a winner by evaluating on the frozen test set, which inflates reported performance. Winner selection must happen exclusively on `train_df` via cross-validation; the frozen test set is evaluated exactly once for the final reported number.

**Files:** `src/method2_random_formula.py`, `src/method3_gp.py`, `src/method4_llm.py`

- In each method script, add a `cv_score(formula_str, train_df, n_folds=5)` helper (or import from `utils.py`) that computes mean AUC-PR across stratified k-folds of `train_df` only.
- M2: after generating the 10 000-formula pool, rank by `cv_auc_pr_mean` instead of test AUC-PR; evaluate the top-ranked formula on the frozen test once.
- M3 (`gplearn`): fitness function already runs on train; after evolution, take the hall-of-fame winner and compute its CV score on `train_df`. No test access during GP evolution.
- M4: candidate pool from inference is scored by `cv_score` on `train_df`; top candidate is then evaluated on frozen test.
- Each method's master summary CSV gains two columns: `CV_AUC_PR_Mean`, `CV_AUC_PR_Std`.
- `Frozen_Test_AUC_PR` column is written only after CV winner selection is complete.

**Pass criteria:**
```bash
python src/method2_random_formula.py --disease ra
python -c "
import pandas as pd
df = pd.read_csv('results/method2_random/ra/top_formulas_cv.csv')
assert 'CV_AUC_PR_Mean' in df.columns, 'CV column missing'
assert df['Frozen_Test_AUC_PR'].notna().any(), 'Frozen test never evaluated'
print('M2 CV OK — winner AUC-PR:', df['Frozen_Test_AUC_PR'].max())
"
```
Repeat for M3 (`method3_gp.py`) and M4 (`method4_llm.py evaluate`).

---

## Issue 25 — Cross-method score vector correlation analysis

**Depends on:** Issue 24

After all methods produce CV-selected predictions on the same test set, check whether methods are truly behaviorally different or just rescaled versions of each other. Pearson r between score vectors (continuous probabilities or formula outputs, not binary predictions) is the primary equivalence metric.

**Files:** `src/compare_methods.py`, new `src/cross_method_correlation.py`

- Create `src/cross_method_correlation.py` that:
  - Loads the `predictions.csv` from each method's results directory for a given disease.
  - Merges on `subject_id` to get aligned score vectors.
  - Computes pairwise Pearson r between all method-pairs and between each method and the LR baseline.
  - Saves a correlation matrix CSV to `results/cross_method/<disease>_score_correlation.csv`.
  - Prints a human-readable table to stdout.
- Rule of thumb to document in `experiment_log.md`: r > 0.95 between two methods means they are functionally equivalent for this disease; lower r is evidence of distinct behavior worth reporting.
- Run for RA first; generalize across all 6 diseases after Issue 28 is closed.

**Pass criteria:**
```python
import pandas as pd
df = pd.read_csv('results/cross_method/ra_score_correlation.csv', index_col=0)
assert df.shape == (5, 5), 'Expected 5x5 matrix (LR + 4 methods)'
assert (df.values.diagonal() == 1.0).all(), 'Diagonal must be 1.0'
print(df.round(3))
```

---

## Issue 26 — Store LLM seed files and define naming standard

**Depends on:** —

Three LLM agents (Gemini 2.5 Pro, GPT-4o Deep Research, SciSpace Agent) each produced formula candidate lists for Crohn's disease. These CSVs must be stored under a consistent path and naming convention before they can be consumed by the seeded-GP pipeline in Issue 27. This issue defines the standard and moves the files into place.

**Files:** new directory `data/llm_seeds/`, new file `docs/llm_seed_standard.md`

### File placement

Seed CSVs belong in:
```
data/llm_seeds/<disease>/<agent_slug>.csv
```

For the three Crohn's files uploaded now:

| Source file | Target path |
|---|---|
| `Gemini3Pro.csv` | `data/llm_seeds/crhn/gemini_25_pro.csv` |
| `GPTDeepResearch.csv` | `data/llm_seeds/crhn/gpt4o_deep_research.csv` |
| `SciSpaceAgent.csv` | `data/llm_seeds/crhn/scispace_agent.csv` |

`data/llm_seeds/` is **gitignored** (same as `data/`). Add the path to `.gitignore` if not already covered by `data/`.

### CSV format standard

Each seed file must contain exactly one column named `expression`. No index column. Each row is one candidate formula string. Formula variables must use the `lab_<FEATURE>_last` naming convention already used in these files.

Example valid row:
```
lab_NEUTpct_last / (lab_LYMpct_last)
```
| Variable in seed files | Maps to |
|---|---|
| `lab_NEUTpct_last` | Neutrophil % |
| `lab_LYMpct_last` | Lymphocyte % |
| `lab_MONOpct_last` | Monocyte % |
| `lab_EOS_pct_last` | Eosinophil % |
| `lab_BASO_pct_last` | Basophil % |
| `lab_WBC_last` | WBC |
| `lab_PLT_last` | PLT |
| `lab_RDW_last` | RDW |
| `lab_HB_last` | HGB (note: seed files use `HB`, pipeline uses `hgb`) |
| `lab_HCT_last` | HCT |
| `lab_RBC_last` | RBC |
| `lab_MCV_last` | MCV |
| `lab_MCH_last` | MCH |
| `lab_MCHC_last` | MCHC |

### Naming standard for agent slugs

| Agent | Slug |
|---|---|
| Gemini 2.5 Pro | `gemini_25_pro` |
| GPT-4o Deep Research | `gpt4o_deep_research` |
| SciSpace Agent | `scispace_agent` |
| Med-Gemma 4B (existing M4) | `medgemma_4b` |

Document this standard in `docs/llm_seed_standard.md` (create the file).

**Pass criteria:**
```bash
python -c "
import pandas as pd, os
for slug in ['gemini_25_pro', 'gpt4o_deep_research', 'scispace_agent']:
    path = f'data/llm_seeds/crhn/{slug}.csv'
    assert os.path.exists(path), f'Missing: {path}'
    df = pd.read_csv(path)
    assert 'expression' in df.columns, f'No expression column in {slug}'
    assert len(df) > 0, f'Empty file: {slug}'
    print(f'{slug}: {len(df)} expressions OK')
"
```

---

## Issue 27 — Add `--seed-file` flag to `method3_gp.py` for seeded-GP warm start

**Depends on:** Issue 26

Vanilla GP initializes its population entirely at random. Seeded GP pre-populates a fraction of the initial generation-0 pool with high-quality formulas from LLM candidates (seed files from Issue 26), giving evolution a head start. This issue adds the mechanism to `method3_gp.py` without breaking the existing vanilla-GP code path.

**Files:** `src/method3_gp.py`, `src/utils.py`

### Behavior

When `--seed-file <path>` is passed:

1. Load the CSV at `<path>`. Expect column `expression`.
2. Parse each expression into a `gplearn`-compatible program tree using `gplearn`'s `_Program` API or by wrapping expressions as strings that the custom fitness function can evaluate directly via `eval()`.
3. Inject parsed programs into the initial population before GP starts. If a seed expression fails to parse (unknown variable, syntax error, division structure gplearn cannot represent), skip it silently and log a warning — never crash.
4. Seed programs count toward `population_size`. If there are more seeds than `population_size * seed_fraction` (default 0.3), truncate to that fraction. If fewer, fill remaining slots with random programs as usual.
5. When `--seed-file` is not passed, behavior is identical to today — no regression.

### CLI interface (extend existing argparse block)

```python
parser.add_argument('--seed-file', type=str, default=None,
    help='Path to CSV with expression column (LLM seed formulas). '
         'Optional. If omitted, GP initializes fully at random.')
parser.add_argument('--seed-fraction', type=float, default=0.3,
    help='Fraction of initial population to fill with seed formulas (default 0.3).')
```

### Variable name translation

Seed files use `lab_<FEATURE>_last` convention. The GP fitness function uses plain lowercase feature names (`rdw`, `plt`, etc.). Add a translation step in `utils.py`:

```python
SEED_VAR_MAP = {
    'lab_HCT_last': 'hct', 'lab_HGB_last': 'hgb', 'lab_HB_last': 'hgb',
    'lab_MCH_last': 'mch', 'lab_MCHC_last': 'mchc', 'lab_MCV_last': 'mcv',
    'lab_PLT_last': 'plt', 'lab_RBC_last': 'rbc', 'lab_RDW_last': 'rdw',
    'lab_WBC_last': 'wbc',
    # CBC differential (Crohn's extended feature set)
    'lab_NEUTpct_last': 'neut_pct', 'lab_LYMpct_last': 'lym_pct',
    'lab_MONOpct_last': 'mono_pct', 'lab_EOS_pct_last': 'eos_pct',
    'lab_BASO_pct_last': 'baso_pct',
}

def translate_seed_expression(expr: str) -> str:
    """Translate lab_X_last variable names to pipeline feature names."""
    for seed_var, pipeline_var in SEED_VAR_MAP.items():
        expr = expr.replace(seed_var, pipeline_var)
    return expr
```

### Output additions

Add two columns to the method's master summary CSV:
- `Seed_File`: basename of the seed file used, or `"none"`.
- `Seed_Count_Used`: number of seed programs successfully injected into generation 0.

**Pass criteria:**
```bash
# Seeded run (Crohn's, small config for speed)
python src/method3_gp.py --disease crhn \
    --seed-file data/llm_seeds/crhn/gemini_25_pro.csv \
    --pop 100 --gen 5

# Validate output
python -c "
import pandas as pd
df = pd.read_csv('results/method3_gp/crhn/master_m3_summary.csv')
last = df.iloc[-1]
assert last['Seed_File'] == 'gemini_25_pro.csv', 'Seed file not recorded'
assert int(last['Seed_Count_Used']) > 0, 'No seeds were injected'
print(f'Seeded run OK. Seeds used: {last[\"Seed_Count_Used\"]}')
"

# Vanilla run (no --seed-file) must still work identically to before
python src/method3_gp.py --disease ra --pop 50 --gen 3
python -c "
import pandas as pd
df = pd.read_csv('results/method3_gp/ra/master_m3_summary.csv')
last = df.iloc[-1]
assert last['Seed_File'] == 'none', 'Vanilla run should record seed_file=none'
print('Vanilla run unaffected OK')
"
```

---

## Issue 28 — Evaluate seeded GP against vanilla GP on Crohn's disease

**Depends on:** Issue 27

Run a controlled comparison: same disease (Crohn's), same GP configuration (large: pop=500, gen=100), same random seed — once with `--seed-file` for each of the three LLM agents, once without. Report AUC-ROC, AUC-PR, and generation-of-best-formula for each run. This answers whether LLM seeding provides a meaningful head start or is noise.

**Files:** `src/method3_gp.py` (run only, no code changes), `results/experiment_log.md`

### Runs to execute (on cluster, CPU queue)

```bash
# Vanilla baseline
python src/method3_gp.py --disease crhn --pop 500 --gen 100

# Seeded — Gemini 2.5 Pro
python src/method3_gp.py --disease crhn --pop 500 --gen 100 \
    --seed-file data/llm_seeds/crhn/gemini_25_pro.csv

# Seeded — GPT-4o Deep Research
python src/method3_gp.py --disease crhn --pop 500 --gen 100 \
    --seed-file data/llm_seeds/crhn/gpt4o_deep_research.csv

# Seeded — SciSpace Agent
python src/method3_gp.py --disease crhn --pop 500 --gen 100 \
    --seed-file data/llm_seeds/crhn/scispace_agent.csv
```

All four runs use `RANDOM_SEED = 42`.

### Comparison table to fill in `experiment_log.md`

| Run | Seed file | AUC-ROC | AUC-PR | Best formula gen | Beats vanilla? |
|-----|-----------|---------|--------|-----------------|----------------|
| Vanilla | none | — | — | — | baseline |
| Gemini 2.5 Pro | gemini_25_pro.csv | — | — | — | ? |
| GPT-4o Deep Research | gpt4o_deep_research.csv | — | — | — | ? |
| SciSpace Agent | scispace_agent.csv | — | — | — | ? |

### Decision rule after results land

- If **any** seeded run beats vanilla on AUC-PR → seeded GP is adopted as the standard M3 configuration for all remaining diseases (Issue 29).
- If **no** seeded run beats vanilla → document as negative result in thesis; Issue 29 is closed without action.
- If results are mixed across agents → adopt the best-performing seed source only; document the others as negative.

**Pass criteria:**
```python
import pandas as pd
df = pd.read_csv('results/method3_gp/crhn/master_m3_summary.csv')
crhn_runs = df[df['Disease'] == 'crhn']
assert len(crhn_runs) >= 4, f'Expected 4 Crohn runs, got {len(crhn_runs)}'
print(crhn_runs[['Seed_File', 'AUC_ROC', 'AUC_PR', 'Best_Gen']].to_string())
```

Log results in `results/experiment_log.md` under a new entry:
```
## [DATE] Crohn's seeded-GP comparison (Issue #28)
Config: pop=500, gen=100, seed=42
[paste comparison table here]
Decision: [adopt / reject / partial]
```

---

## Issue 29 — Generalize seeded-GP to all remaining diseases

**Depends on:** Issue 28

If the decision from Issue 28 is to adopt seeded GP, apply it consistently across all diseases that have not yet been run with the winning seed source. Each disease needs its own seed file(s) from the same LLM agents; files follow the naming standard from Issue 26.

**Files:** `data/llm_seeds/<disease>/`, `src/method3_gp.py` (run only), `results/experiment_log.md`

### Per-disease seed file checklist

Before running any disease, confirm its seed directory exists and is populated:

```
data/llm_seeds/
├── crhn/          ← done (Issue 26)
│   ├── gemini_25_pro.csv
│   ├── gpt4o_deep_research.csv
│   └── scispace_agent.csv
├── ra/            ← needs files
├── t1d/           ← needs files
├── t2d/           ← needs files
├── psr/           ← needs files
└── lup/           ← needs files
```

For each missing disease: generate seed CSVs from the same LLM agents, save under `data/llm_seeds/<disease>/`, verify with the pass-criteria check from Issue 26 (swap disease slug).

### Run template (replace `<disease>` and `<agent_slug>`)

```bash
python src/method3_gp.py \
    --disease <disease> \
    --pop 500 --gen 100 \
    --seed-file data/llm_seeds/<disease>/<agent_slug>.csv
```

Use whichever agent slug won in Issue 28. If results were mixed, run only the winning agent.

### Validation after all runs complete

```python
import pandas as pd
df = pd.read_csv('results/method3_gp/master_m3_summary.csv')  # merged across diseases
diseases = ['ra', 't1d', 't2d', 'crhn', 'psr', 'lup']
for d in diseases:
    rows = df[(df['Disease'] == d) & (df['Seed_File'] != 'none')]
    assert len(rows) >= 1, f'No seeded run found for {d}'
    best = rows.sort_values('AUC_PR', ascending=False).iloc[0]
    print(f"{d}: seeded AUC-PR={best['AUC_PR']:.4f}  seed={best['Seed_File']}")
```

Update `results/experiment_log.md` with one entry per disease, same format as Issue 28.

**Note:** If Issue 28 decision is "reject", close this issue without action and document the negative result in the thesis as evidence that LLM seeding does not improve GP convergence for CBC-based biomarker discovery.

---

## Issue 30 — Expand feature set to 14 CBC features + full data purge and rerun

**Depends on:** —

The current pipeline uses 9 CBC features. Five differential count features have been confirmed as valid percentages (0–100 range, 276k patient coverage — essentially the full MIMIC-IV population) and are now registered in `ref_cbc_tests`. Adding them requires surgical changes across 7 locations, a complete purge of all generated data, and a full cluster rerun. All existing results are invalidated — nothing in `data/` or `results/` is compatible with the 14-feature schema.

**New features confirmed (BigQuery verified):**

| itemid | label | abbrev | avg_val | max_val |
|--------|-------|--------|---------|---------|
| 51256 | Neutrophils | `neut_pct` | 64.97 | 100.0 |
| 51244 | Lymphocytes | `lym_pct` | 23.45 | 100.0 |
| 51254 | Monocytes | `mono_pct` | 7.44 | 100.0 |
| 51200 | Eosinophils | `eos_pct` | 2.16 | 98.0 |
| 51146 | Basophils | `baso_pct` | 0.51 | 63.0 |

`ref_cbc_tests` already updated (done before this issue was written). The 7 remaining changes are below.

---

### Step 1 — `src/queries/cohort_pipeline.sql`

In Checkpoint 3, the final `SELECT` pivot hard-codes one `MAX(IF(...))` column per feature. Add 5 new lines:

```sql
-- existing 9:
  MAX(IF(f.test_abbrev = 'HCT',     f.valuenum, NULL)) AS hct,
  MAX(IF(f.test_abbrev = 'HGB',     f.valuenum, NULL)) AS hgb,
  MAX(IF(f.test_abbrev = 'MCH',     f.valuenum, NULL)) AS mch,
  MAX(IF(f.test_abbrev = 'MCHC',    f.valuenum, NULL)) AS mchc,
  MAX(IF(f.test_abbrev = 'MCV',     f.valuenum, NULL)) AS mcv,
  MAX(IF(f.test_abbrev = 'PLT',     f.valuenum, NULL)) AS plt,
  MAX(IF(f.test_abbrev = 'RBC',     f.valuenum, NULL)) AS rbc,
  MAX(IF(f.test_abbrev = 'RDW',     f.valuenum, NULL)) AS rdw,
  MAX(IF(f.test_abbrev = 'WBC',     f.valuenum, NULL)) AS wbc,
-- add these 5:
  MAX(IF(f.test_abbrev = 'NEUTpct', f.valuenum, NULL)) AS neut_pct,
  MAX(IF(f.test_abbrev = 'LYMpct',  f.valuenum, NULL)) AS lym_pct,
  MAX(IF(f.test_abbrev = 'MONOpct', f.valuenum, NULL)) AS mono_pct,
  MAX(IF(f.test_abbrev = 'EOS_pct', f.valuenum, NULL)) AS eos_pct,
  MAX(IF(f.test_abbrev = 'BASO_pct',f.valuenum, NULL)) AS baso_pct
```

Also update Checkpoint 4: the split filter currently checks `WHERE hgb IS NOT NULL`. This remains correct — HGB is the anchor completeness check and need not change.

Also update Checkpoint 5 export `SELECT`: append `, f.neut_pct, f.lym_pct, f.mono_pct, f.eos_pct, f.baso_pct` to the column list.

Also update the validation comment in Test 2: `-- expect 14 rows after adding differential features`.

---

### Step 2 — `src/utils.py`

Two constants need updating:

```python
# Change:
CBC_FEATURE_LIST = ["rdw", "hgb", "hct", "wbc", "plt", "mcv", "mch", "mchc", "rbc"]

# To:
CBC_FEATURE_LIST = [
    "hct", "hgb", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc",
    "neut_pct", "lym_pct", "mono_pct", "eos_pct", "baso_pct",
]
```

`CBC_FEATURE_LIST` is used by `method4_llm.py` for prompt construction and by `build_threshold_prompt()`. `load_data()` derives features dynamically from CSV columns so needs no change — it will automatically pick up the 5 new columns once the CSVs are regenerated.

Also update `SEED_VAR_MAP` (added in Issue 27) to confirm the 5 differential entries are present — they should already be there from Issue 27, but verify.

---

### Step 2b — `src/method4_llm.py` — prompt templates and seeded-prompt feature context

The `generate` subcommand builds prompt strings that tell Med-Gemma which features are available and what they mean. With 9 features the prompts listed them inline; with 14 features the prompt must be updated in three places.

**2b.1 — Blind prompt: feature list and medical descriptions**

The blind prompt currently enumerates all 9 CBC features with brief clinical descriptions. Extend it with the 5 differential features. The descriptions matter — Med-Gemma needs medical context to generate biologically plausible formulas for ratio/interaction terms:

```python
# In the blind prompt template (look for the block that describes available features)
# Change the feature description block from:
FEATURE_DESCRIPTIONS = """
Available CBC features (use exact variable names in your formulas):
  hct    - Hematocrit (%): volume fraction of red blood cells
  hgb    - Hemoglobin (g/dL): oxygen-carrying protein in RBCs
  mch    - Mean Corpuscular Hemoglobin (pg): average Hgb per RBC
  mchc   - Mean Corpuscular Hemoglobin Concentration (g/dL)
  mcv    - Mean Corpuscular Volume (fL): average RBC size
  plt    - Platelet count (10^9/L): thrombocytes
  rbc    - Red Blood Cell count (10^12/L)
  rdw    - Red Cell Distribution Width (%): RBC size variability
  wbc    - White Blood Cell count (10^9/L): total leukocytes
"""

# To:
FEATURE_DESCRIPTIONS = """
Available CBC features (use exact variable names in your formulas):
  hct      - Hematocrit (%): volume fraction of red blood cells
  hgb      - Hemoglobin (g/dL): oxygen-carrying protein in RBCs
  mch      - Mean Corpuscular Hemoglobin (pg): average Hgb per RBC
  mchc     - Mean Corpuscular Hemoglobin Concentration (g/dL)
  mcv      - Mean Corpuscular Volume (fL): average RBC size
  plt      - Platelet count (10^9/L): thrombocytes
  rbc      - Red Blood Cell count (10^12/L)
  rdw      - Red Cell Distribution Width (%): RBC size variability
  wbc      - White Blood Cell count (10^9/L): total leukocytes
  neut_pct - Neutrophil percentage (% of WBC): elevated in bacterial infection/inflammation
  lym_pct  - Lymphocyte percentage (% of WBC): key adaptive immune cell; altered in autoimmune disease
  mono_pct - Monocyte percentage (% of WBC): innate immunity; elevated in chronic inflammation
  eos_pct  - Eosinophil percentage (% of WBC): allergic/parasitic response; altered in IBD/autoimmune
  baso_pct - Basophil percentage (% of WBC): rare; elevated in hypersensitivity and some autoimmune conditions

Note: neut_pct + lym_pct + mono_pct + eos_pct + baso_pct ≈ 100 (they are parts of the same differential).
Ratios such as neut_pct / lym_pct (NLR) are clinically established inflammation markers.
"""
```

Also update the example formulas block in the blind prompt to include at least one example that uses a differential feature, so Med-Gemma understands those variables are valid:

```python
# Add to the examples block:
# np.log1p(neut_pct / lym_pct) * rdw          # NLR-based inflammation × RBC heterogeneity
# (neut_pct - lym_pct) / (neut_pct + lym_pct) # Systemic Inflammation Index variant
```

**2b.2 — Seeded prompt: update feature importance ranking**

The seeded prompt currently passes GP's feature importance ranking from the 9-feature RA run (RDW > PLT > MCHC > ...) as context to Med-Gemma. After Issue 30's rerun, GP will produce a new 14-feature importance ranking. The seeded prompt must be parameterized so it receives the ranking at runtime rather than having RA-specific values hardcoded:

```python
# Change: hardcoded string like
# "Top features by GP importance: RDW (0.41), PLT (0.28), MCHC (0.19), ..."

# To: runtime parameter built from GP results
def build_seeded_prompt(disease: str, feature_importances: dict, gp_best_formula: str) -> str:
    """
    feature_importances: dict mapping feature name -> importance score,
                         sorted descending. Comes from GP hall-of-fame analysis.
    gp_best_formula:     string representation of GP's best formula (motif only,
                         not exact tree — to avoid LLM just copying it).
    """
    ranked = sorted(feature_importances.items(), key=lambda x: -x[1])
    ranking_str = ", ".join(f"{feat} ({score:.2f})" for feat, score in ranked[:7])
    return SEEDED_PROMPT_TEMPLATE.format(
        disease=disease,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        top_features=ranking_str,
        gp_motif=gp_best_formula,
    )
```

The `feature_importances` dict should be loaded from `results/method3_gp/<disease>/feature_importance.json` (or equivalent output from `method3_gp.py`). If that file doesn't exist for the disease being run, fall back to the blind prompt and log a warning.

**2b.3 — `evaluate` subcommand: no changes needed, but verify**

The `evaluate` subcommand parses formula strings and runs them against the modeling CSV via `eval()`. Because `load_data()` in `utils.py` derives feature columns dynamically from the CSV, and the formula strings reference variable names directly (e.g. `neut_pct`), no code changes are needed in `evaluate`. However, confirm that:
- The formula namespace passed to `eval()` is built from `CBC_FEATURE_LIST` (updated in Step 2), not a hardcoded list.
- If any formula references `neut_pct` etc. and those columns are present in the CSV, evaluation succeeds without modification.

Add a one-line assertion in `evaluate`'s setup block:
```python
assert set(CBC_FEATURE_LIST).issubset(set(df.columns)), \
    f"CSV missing features: {set(CBC_FEATURE_LIST) - set(df.columns)}"
```

**Pass criteria for Step 2b:**
```python
# After generating with the new prompts on any disease post-rerun:
import json, subprocess, pandas as pd

# 1. Blind prompt contains all 14 feature variable names
result = subprocess.run(
    ['python', 'src/method4_llm.py', 'generate', '--disease', 'ra',
     '--strategy', 'blind', '--dry-run'],  # add --dry-run flag that prints prompt and exits
    capture_output=True, text=True
)
for feat in ['neut_pct', 'lym_pct', 'mono_pct', 'eos_pct', 'baso_pct']:
    assert feat in result.stdout, f'Blind prompt missing feature: {feat}'
print('Blind prompt: all 14 features present OK')

# 2. Seeded prompt references runtime ranking, not hardcoded RA values
# Run seeded generate for a non-RA disease — if it crashes with KeyError it means
# the prompt still has hardcoded RA feature importances
result2 = subprocess.run(
    ['python', 'src/method4_llm.py', 'generate', '--disease', 'crhn',
     '--strategy', 'seeded', '--dry-run'],
    capture_output=True, text=True
)
assert result2.returncode == 0, f'Seeded prompt crashed for crhn: {result2.stderr}'
print('Seeded prompt: disease-agnostic OK')

# 3. Evaluate namespace covers all 14 features
df = pd.read_csv('data/ra_modeling_data.csv')
from src.utils import CBC_FEATURE_LIST
assert set(CBC_FEATURE_LIST).issubset(set(df.columns)), 'Feature mismatch'
print('Evaluate namespace OK: all 14 features in CSV')
```

---

### Step 3 — `conf/ml/defaults.yaml`

The `baselines` block contains RA-specific hardcoded numbers that were computed on the 9-feature dataset. Reset them to placeholder zeros — they will be overwritten once `sanity_check.py` runs on the new data:

```yaml
baselines:
  lr_auc_roc: 0.0   # reset — recompute after Issue 30 rerun
  lr_auc_pr:  0.0   # reset — recompute after Issue 30 rerun
  m2_best_auc_pr:  0.0
  gp_best_auc_roc: 0.0
  gp_best_auc_pr:  0.0
```

Add a comment above the block: `# Values below are reset in Issue #30 and will be repopulated by sanity_check.py`.

---

### Step 4 — `conf/ehrshot.yaml`

Add 5 new entries to `cbc_codes`. OMOP and LOINC concept IDs for CBC differential percentages — these are standard OMOP codes but **must be verified** against the local EHRSHOT vocabulary before running `ehrshot_data.py`:

```yaml
cbc_codes:
  # existing 9 ...
  neut_pct:  ["OMOP/3013650", "LOINC/770-8"]   # Neutrophils/100 leukocytes in Blood
  lym_pct:   ["OMOP/3004327", "LOINC/736-9"]   # Lymphocytes/100 leukocytes in Blood
  mono_pct:  ["OMOP/3009744", "LOINC/5905-5"]  # Monocytes/100 leukocytes in Blood
  eos_pct:   ["OMOP/3013429", "LOINC/713-8"]   # Eosinophils/100 leukocytes in Blood
  baso_pct:  ["OMOP/3017732", "LOINC/706-2"]   # Basophils/100 leukocytes in Blood
```

Add a comment: `# IMPORTANT: neut_pct–baso_pct concept IDs added in Issue #30. Verify against local EHRSHOT vocabulary before running ehrshot_data.py`.

---

### Step 5 — `conf/nhanes.yaml`

NHANES CBC component files include differential counts. Add 5 new entries to `cbc_vars`. Variable names are consistent across cycles G–J:

```yaml
cbc_vars:
  # existing 9 ...
  neut_pct:  LBXNE    # Neutrophils percent [%]
  lym_pct:   LBXLYPCT # Lymphocyte percent [%]
  mono_pct:  LBXMOPCT # Monocyte percent [%]
  eos_pct:   LBXEOPCT # Eosinophils percent [%]
  baso_pct:  LBXBAPCT # Basophils percent [%]
```

Add a comment: `# neut_pct–baso_pct added in Issue #30. Verify variable names against CBC codebook for cycles G–J before running nhanes_data.py`.

**Note:** NHANES does not always include the full differential in every cycle. Run `nhanes_sanity.py` after extraction and check the coverage column for these 5 features. If coverage is below 40% for any active disease, remove the low-coverage feature(s) from that disease's run and document the decision.

---

### Step 6 — `STANDARDS.md`

Update the `CBC_FEATURES` constant reference:

```python
# Change:
CBC_FEATURES = ['hct', 'hgb', 'mch', 'mchc', 'mcv', 'plt', 'rbc', 'rdw', 'wbc']

# To:
CBC_FEATURES = [
    'hct', 'hgb', 'mch', 'mchc', 'mcv', 'plt', 'rbc', 'rdw', 'wbc',
    'neut_pct', 'lym_pct', 'mono_pct', 'eos_pct', 'baso_pct',
]
```

Update the data section description: `Columns: subject_id, is_case, split, + 14 CBC features (9 standard + 5 differential)`.

---

### Step 7 — `CLAUDE.md`

Update the data section:

```
# Change:
- Columns: subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc

# To:
- Columns: subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc,
           neut_pct, lym_pct, mono_pct, eos_pct, baso_pct
- Feature count: 14 (9 standard CBC + 5 differential: neutrophil/lymphocyte/monocyte/eosinophil/basophil %)
```

---

### Step 8 — Full data purge and rerun

Once all 7 code changes above are made and committed:

**Purge and resubmit** (preserves `data/nhanes/` XPT files — slow to re-download):
```bash
bash run_all.sh --purge
```
Deletes `data/*` (except `data/nhanes/`), `results/`, `cluster_results/`, then submits all jobs. Prompts for `YES` before deleting.

Optional: sanity-check locally before cluster submission:
```bash
for disease in ra t1d t2d crhn psr lup; do
    python src/run_pipeline.py disease=$disease
    python src/sanity_check.py --disease $disease
done
```
Pass criterion: best single-feature AUC-PR < 0.85 for all diseases.

If BQ credentials are not set up via ADC:
```bash
EHRSHOT_KEY_FILE=.secrets/bq_sa.json bash run_all.sh --purge
```

---

### Pass criteria (run after cluster results land)

```python
import pandas as pd

# 1. All modeling CSVs have 14 feature columns
for disease in ['ra', 't1d', 't2d', 'crhn', 'psr', 'lup']:
    df = pd.read_csv(f'data/{disease}_modeling_data.csv')
    feat_cols = [c for c in df.columns if c not in {'subject_id', 'is_case', 'split'}]
    assert len(feat_cols) == 14, f'{disease}: expected 14 features, got {len(feat_cols)}: {feat_cols}'
    new_feats = {'neut_pct', 'lym_pct', 'mono_pct', 'eos_pct', 'baso_pct'}
    assert new_feats.issubset(set(feat_cols)), f'{disease}: missing differential features'
    print(f'{disease}: 14 features OK, prevalence={df["is_case"].mean():.3%}')

# 2. New features are in valid percentage range (0–100)
df = pd.read_csv('data/ra_modeling_data.csv')
for feat in ['neut_pct', 'lym_pct', 'mono_pct', 'eos_pct', 'baso_pct']:
    vals = df[feat].dropna()
    assert vals.between(0, 100).all(), f'{feat} has out-of-range values'
    print(f'{feat}: min={vals.min():.1f}  max={vals.max():.1f}  mean={vals.mean():.1f}  OK')

# 3. Differential sum constraint (neut + lym + mono + eos + baso ≈ 100)
df_clean = df[['neut_pct','lym_pct','mono_pct','eos_pct','baso_pct']].dropna()
row_sums = df_clean.sum(axis=1)
# Allow 80–105 (some rows include bands/metamyelocytes in the differential)
assert row_sums.between(80, 105).mean() > 0.90, \
    f'Only {row_sums.between(80,105).mean():.1%} of rows sum to 80–105 — check itemids'
print(f'Differential sum check: {row_sums.between(80,105).mean():.1%} of rows in 80–105 range OK')
```
