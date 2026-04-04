# Biomarker Pipeline — Project Guide for Claude Code

## What this project does
Biomarker discovery for Rheumatoid Arthritis from CBC (Complete Blood Count) data.
Uses MIMIC-IV dataset. Compares multiple methods for generating biomarker formulas.

## How to run
All scripts run from the project root: `python src/<script_name>.py`
Data must be in `data/ra_modeling_data.csv` (not tracked in git).

## Coding standards
- **One file per method** in `src/`, named `method<N>_<name>.py`
- **Shared code goes in `src/utils.py`** — never duplicate data loading or metric computation
- **No classes, no frameworks** — keep it flat, functions only
- **Use the pre-computed split column** — NEVER re-split the data
- **All results go to `results/`** — each method gets its own subdirectory
- **Report these metrics for every method**: AUC-ROC, AUC-PR, precision, recall, F1, F2, precision@recall(0.25, 0.50, 0.75)
- **Print progress** to stdout as scripts run
- **Use `src/utils.py` constants** for paths, baseline AUC, feature lists

## Data
- `data/ra_modeling_data.csv` — exported from BigQuery, gitignored
- Columns: subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc
- Split is immutable: train (80%) / test (20%), patient-level, deterministic
- ~1% positive rate (RA cases) — always use imbalanced-aware metrics

## Key numbers
- Train: 1,273 cases / 115,955 controls
- Test: 300 cases / 28,976 controls
- Baseline AUC-ROC: 0.658 (all-features logistic regression)
- Baseline AUC-PR: 0.017 (all-features logistic regression)

## File structure
```
biomarker-pipeline/
├── CLAUDE.md              # this file
├── README.md              # project overview
├── requirements.txt
├── data/                  # gitignored
├── results/               # generated outputs, mostly gitignored
│   └── experiment_log.md  # tracked — manual experiment tracking
└── src/
    ├── utils.py           # shared utilities (data loading, metrics, paths)
    ├── checkpoint5_sanity.py
    ├── method1_threshold.py
    ├── add_pr_metrics.py
    └── (future method scripts)
```
