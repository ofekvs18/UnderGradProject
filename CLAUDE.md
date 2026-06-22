# Biomarker Pipeline — Project Guide for Claude Code

## What this project does
Biomarker discovery for 6 autoimmune/metabolic diseases (RA, Crohn's, T1D, T2D, Psoriasis, Lupus) from CBC (Complete Blood Count) data.
Uses MIMIC-IV dataset. Compares multiple methods for generating biomarker formulas across all diseases.

## How to run
All scripts run from the project root (`biomarker-pipeline/`): `python src/<script_name>.py`
Data files: `data/<disease>_modeling_data.csv` (not tracked in git, one per disease slug).
Python: use `.venv` from the **parent** directory — `../.venv/Scripts/python.exe` (the venv lives at `C:/Users/ofek/Downloads/gitRepos/Project/.venv`, not inside `biomarker-pipeline/`).

## Git commit standards
- **Always include the issue number** in the commit message, e.g. `fix(#3): add AUC-PR metric` or `feat(#2): implement Youden threshold`
- Use the format: `<type>(#<issue>): <description>`

## Coding standards
**See [STANDARDS.md](STANDARDS.md) for complete coding standards** — file organization, style guide, data rules, metrics, experiment tracking, and git conventions.

Quick reference:
- **One file per method** in `src/`, named `method<N>_<name>.py`
- **Shared code goes in `src/utils.py`** — never duplicate data loading or metric computation
- **No classes, no frameworks** — keep it flat, functions only
- **Use the pre-computed split column** — NEVER re-split the data
- **All results go to `results/`** — each method gets its own subdirectory
- **Report these metrics for every method**: AUC-ROC, AUC-PR, precision, recall, F1, F2, precision@recall(0.25, 0.50, 0.75)
- **Print progress** to stdout as scripts run
- **Use `src/utils.py` constants** for paths, baseline AUC, feature lists

## Data
- `data/<disease>_modeling_data.csv` — one file per disease slug, exported from BigQuery, gitignored
- Columns: subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc,
           neut_pct, lym_pct, mono_pct, eos_pct, baso_pct
- Feature count: 14 (9 standard CBC + 5 differential: neutrophil/lymphocyte/monocyte/eosinophil/basophil %)
- Split is immutable: train (80%) / test (20%), patient-level, deterministic
- ~1% positive rate — always use imbalanced-aware metrics
- Disease slugs: `ra`, `crhn`, `t1d`, `t2d`, `psr`, `lup`

## EHRSHOT validation cohort (BigQuery)
External-validation cohort extracted from the `EHRSHOTS_DATA` BigQuery dataset (OMOP CDM: `condition_occurrence`, `measurement`, `visit_occurrence`).
- Run: `python src/ehrshot_bq_data.py --disease <slug>` → writes `data/<slug>_ehrshot_data.csv` in the standard 17-column format. Auth via ADC or `--key-file`.
- Config: `conf/ehrshot_bq.yaml` holds the dataset name, OMOP `measurement_concept_id` per CBC feature, and **EHRSHOT-specific** `disease_icd_patterns`.
- **Gotcha:** EHRSHOT stores **dotted** ICD-9 (`714.0`, `250.11`) and is mostly **ICD-10** (`E11.9`, `M05.9`) — unlike MIMIC's dotless ICD-9. The `conf/disease/<slug>.yaml` patterns DO NOT work here; always use `disease_icd_patterns` in `conf/ehrshot_bq.yaml` (covers both vocabularies per disease).
- Cohorts extracted for all 6 diseases. Prevalence runs high (3.7%–57.5%) because controls get a random index date and are dropped if they lack a CBC in the lookback window; AUC-ROC is unaffected, but AUC-PR / precision@recall are prevalence-sensitive.
- Distinct from `src/ehrshot_data.py` + `conf/ehrshot.yaml`, which read the local MEDS-parquet timeline.

## Cluster pipeline
`run_all.sh` submits 3 SLURM jobs per disease in an `afterok` chain:
1. **GPU** (`medgemma_generate.sbatch`) — MedGemma LLM generation
2. **CPU** (`pipeline_cpu.sbatch`) — MIMIC data extraction + M1–M4 + cross-method correlation
3. **POST** (`post_compute.sbatch`) — calls `src/post_compute.py` for 8 steps:
   - Step 1: MIMIC bootstrap CIs
   - Steps 2–4: NHANES data prep → evaluate → CIs (skipped for t1d/t2d)
   - Steps 5–6: EHRSHOT BQ extraction → evaluate (all 6 diseases)
   - Steps 7–8: dashboard data + forest plot

Key env vars (set before calling `run_all.sh`):
```bash
NHANES_DIR=data/nhanes          # path to downloaded NHANES XPT files (default)
EHRSHOT_KEY_FILE=               # GCP service-account JSON; unset = use ADC
SPLIT_SALT=                     # optional reproducibility salt
```

**Purge and rerun** — preserves `data/nhanes/` (XPT files are slow to re-download):
```bash
bash run_all.sh --purge         # interactive confirmation required
bash run_all.sh --purge --dry-run  # preview what would be deleted
```

## GitHub
- Repo: `ofekvs18/UnderGradProject`
- Git root: `biomarker-pipeline/` folder

## File structure
```
biomarker-pipeline/
├── CLAUDE.md              # this file
├── README.md              # project overview
├── STANDARDS.md           # project standards
├── PROJECT_STATUS.md      # running status of all TODOs and experiments
├── requirements.txt
├── run_all.sh             # submit all SLURM jobs (GPU + CPU + POST per disease, afterok chain)
├── conf/
│   ├── ml/defaults.yaml          # method hyperparameters (seed, baselines, GP tiers)
│   ├── disease/<slug>.yaml       # per-disease ICD patterns + feature_relevance for LLM prompts
│   ├── nhanes.yaml               # NHANES CBC variable names (validation cohort)
│   ├── ehrshot.yaml              # EHRSHOT MEDS-parquet concept codes (validation cohort)
│   └── ehrshot_bq.yaml           # EHRSHOT BigQuery OMOP config: dataset, concept IDs, per-disease ICD patterns
├── data/                  # gitignored
│   ├── nhanes/                   # downloaded NHANES XPT files (preserved by --purge)
│   └── llm_seeds/<disease>/      # LLM-generated seed formula CSVs (gitignored)
├── docs/                  # standards and design docs
│   ├── llm_seed_standard.md      # naming convention for LLM seed files (Issue #26)
│   ├── Can-a-Routine-CBC-Predict-Chronic-Disease_v2.pptx  # updated presentation (10 slides)
│   └── figures/                  # presentation PNGs generated by src/presentation_figures.py
│       ├── figA_{disease}.png    # per-disease AUC-PR with 95% CI bracket (one per disease)
│       ├── figA_per_disease_aupr.png  # same as figA but 2×3 grid
│       ├── figB_all_diseases_aupr.png # all-disease grouped bar (MIMIC AUC-PR)
│       ├── figC_complexity.png   # heatmap: N_Features (color) + AUC-PR (text)
│       ├── figD_ehrshot_generalization.png  # MIMIC vs EHRSHOT AUC-ROC (2×3 grid)
│       ├── figE_nhanes_generalization.png   # MIMIC vs NHANES AUC-ROC (RA + PSR)
│       ├── figF_mimic_lift.png   # AUC-PR lift on MIMIC (formula / prevalence)
│       └── figG_ehrshot_lift.png # AUC-PR lift on EHRSHOT
├── results/               # generated outputs, mostly gitignored
│   ├── experiment_log.md         # tracked — manual experiment tracking
│   ├── methods_comparison.csv    # generated by compare_methods.py
│   └── matched_lr_baseline.csv   # matched-feature LR baselines per disease/method (by matched_lr_baseline.py)
└── src/
    ├── utils.py                  # shared utilities (data loading, metrics, paths, SEED_VAR_MAP)
    ├── prompts.json              # LLM prompt templates and CBC feature metadata
    ├── queries/cohort_pipeline.sql  # BigQuery cohort extraction (parameterized)
    ├── run_pipeline.py           # runs BigQuery pipeline for one disease
    ├── run_full_pipeline.py      # end-to-end chain (8 steps) for one disease
    ├── sanity_check.py           # LR baseline + per-k best-subset analysis
    ├── method_threshold.py       # Method 1: threshold optimization
    ├── method2_random_formula.py # Method 2: random formula search (CV-selected)
    ├── method3_gp.py             # Method 3: genetic programming (CV-selected, seeded-GP via --seed-file)
    ├── method4_llm.py            # Method 4: LLM-generated formulas (Med-Gemma 4B)
    ├── compare_methods.py        # aggregate master summaries; M3=vanilla GP, M5=seeded GP
    ├── cross_method_correlation.py  # pairwise Pearson r between method score vectors (Issue #25)
    ├── matched_lr_baseline.py    # matched-feature LR baselines for M2/M3/M4/M5: formula + LR AUC + bootstrap CIs
    ├── presentation_figures.py   # generate all presentation PNGs (figA–figG) from CI/results CSVs
    ├── update_pptx.py            # one-shot script to regenerate docs/..._v2.pptx from results
    ├── build_dashboard_data.py   # prep data for Streamlit dashboard
    ├── dashboard.py              # Streamlit + Plotly interactive dashboard
    ├── mimic_compute_ci.py       # bootstrap CIs for MIMIC results
    ├── nhanes_compute_ci.py      # bootstrap CIs for NHANES results
    ├── plot_ci_forest.py         # forest plot of CIs across methods/diseases
    ├── nhanes_data.py            # download + extract NHANES CBC cycles G–J
    ├── nhanes_sanity.py          # NHANES sanity checks
    ├── nhanes_evaluate.py        # evaluate pipeline formulas on NHANES cohort
    ├── ehrshot_data.py           # extract EHRSHOT CBC features (MEDS parquet timeline)
    ├── ehrshot_bq_data.py        # extract EHRSHOT cohort from BigQuery OMOP tables (EHRSHOTS_DATA)
    ├── ehrshot_sanity.py         # EHRSHOT sanity checks
    └── ehrshot_evaluate.py       # evaluate pipeline formulas on EHRSHOT cohort
```

## Master summaries
Each method writes a master summary CSV that is **append-only** (runs are never overwritten).
Every row includes `Timestamp` and `Split_Salt` so you can trace which run produced which result.
Run `python src/compare_methods.py` to merge all master summaries into `results/methods_comparison.csv`.

Output columns: `Disease`, `Split_Salt`, `Best_Method`, `Best_Formula`, `Best_AUC_PR`, then per-method pairs `M1_Best_Formula`/`M1_Best_AUC_PR` … `M5_Best_Formula`/`M5_Best_AUC_PR`.
- **M3** = vanilla GP only (`Seed_File == "none"`); **M5** = seeded GP (`Seed_File != "none"`). T2D has no M5.

Method 3 writes two masters:
- Global: `results/method3_gp/master_gp_summary.csv`
- Per-disease: `results/method3_gp/<disease>/master_m3_summary.csv`

Both include `Seed_File` (basename or `"none"`) and `Seed_Count_Used` columns.

## LLM seed files
Seed formula CSVs live in `data/llm_seeds/<disease>/` (gitignored). Three agents per disease:
- `gemini_25_pro.csv`, `gpt4o_deep_research.csv`, `scispace_agent.csv`

Populated for: `ra`, `crhn`, `psr`, `lup`, `t1d`. No seeds for `t2d` — vanilla GP only.
Pass `--seed-file data/llm_seeds/<disease>/gemini_25_pro.csv` to `method3_gp.py` to warm-start GP.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
