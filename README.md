# RA Biomarker Discovery Pipeline

Can routine CBC (Complete Blood Count) values predict Rheumatoid Arthritis?

This project systematically compares methods for deriving single-formula biomarkers from CBC data using the MIMIC-IV clinical database.

## Research question

> Can a simple formula derived from routine CBC values (RDW, HGB, HCT, WBC, PLT, RBC, MCH, MCHC, MCV) predict whether a patient has Rheumatoid Arthritis better than an all-features logistic regression (AUC=0.658)?

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data file (see Data section below)
#    data/ra_modeling_data.csv

# 3. Run from project root
python src/checkpoint5_sanity.py     # verify pipeline is leak-free
python src/method1_threshold.py      # single-feature threshold optimization
python src/add_pr_metrics.py         # add AUC-PR and imbalance-aware metrics
```

## Data

Source: **MIMIC-IV** (PhysioNet), queried via BigQuery.

- File: `data/ra_modeling_data.csv` (gitignored — provide your own export)
- Columns: `subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc`
- Cases: ICD-9 code 714.x (Rheumatoid Arthritis)
- Lookback window: 90 days + first 24h of index admission
- Split: pre-computed, patient-level, 80/20 train/test — **do not re-split**

| Split | Cases | Controls | Total |
|-------|-------|----------|-------|
| Train | 1,273 | 115,955 | 117,228 |
| Test  | 300   | 28,976  | 29,276 |

## Methods

| # | Method | Status | Best AUC-ROC | Best AUC-PR |
|---|--------|--------|-------------|------------|
| — | All-features logistic regression (baseline) | Done | 0.658 | 0.017 |
| 1 | Threshold optimization (literature + Youden) | Done | 0.617 | 0.0141 |
| 2 | Random formula generation | Planned | — | — |
| 3 | Genetic programming | Planned | — | — |
| 4 | LLM-generated formulas | Planned | — | — |

## Results

See [`results/experiment_log.md`](results/experiment_log.md) for full results tables.

Key finding so far: single-feature thresholds do not beat the all-features baseline.
AUC-PR rankings differ from AUC-ROC rankings under severe class imbalance (~1% positive rate).

## Project structure

```
biomarker-pipeline/
├── CLAUDE.md              # coding standards for Claude Code
├── README.md              # this file
├── requirements.txt
├── data/                  # gitignored — place ra_modeling_data.csv here
├── results/
│   ├── experiment_log.md  # tracked experiment tracking doc
│   └── method1_threshold/ # generated outputs per method
└── src/
    ├── utils.py           # shared data loading, metrics, path constants
    ├── checkpoint5_sanity.py
    ├── method1_threshold.py
    └── add_pr_metrics.py
```

## Author

Ofek VS — Undergraduate research project.
