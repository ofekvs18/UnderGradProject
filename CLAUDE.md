# Biomarker Pipeline — Project Guide for Claude Code

## What this project does
Biomarker discovery for Rheumatoid Arthritis from CBC (Complete Blood Count) data.
Uses MIMIC-IV dataset. Compares multiple methods for generating biomarker formulas.

## How to run
All scripts run from the project root (`biomarker-pipeline/`): `python src/<script_name>.py`
Data file: `data/ra_modeling_data.csv` (not tracked in git).
Python: use `.venv` from the **parent** directory — `../.venv/Scripts/python.exe` (the venv lives at `C:/Users/ofek/Downloads/gitRepos/Project/.venv`, not inside `biomarker-pipeline/`).

## Git commit standards
- **Always include the issue number** in the commit message, e.g. `fix(#3): add AUC-PR metric` or `feat(#2): implement Youden threshold`
- Use the format: `<type>(#<issue>): <description>`

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
- `data/modeling_data.csv` — exported from BigQuery, gitignored
- Columns: subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc
- Split is immutable: train (80%) / test (20%), patient-level, deterministic
- ~1% positive rate (RA cases) — always use imbalanced-aware metrics

## Key numbers
- Train: 1,273 cases / 115,955 controls
- Test: 300 cases / 28,976 controls
- Baseline AUC-ROC: 0.658 (all-features logistic regression)
- Baseline AUC-PR: 0.017 (all-features logistic regression)

## GitHub
- Repo: `ofekvs18/UnderGradProject`
- Git root: `biomarker-pipeline/` folder

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
    ├── sanity_check.py
    ├── method_threshold.py
    ├── eval_pr_metrics.py
    └── (future method scripts)
```
