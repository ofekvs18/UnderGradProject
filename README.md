# RA Biomarker Discovery Pipeline

Can routine CBC values predict Rheumatoid Arthritis?

Compares methods for deriving single-formula biomarkers from CBC data using MIMIC-IV.

**Goal:** Beat all-features logistic regression baseline (AUC-ROC: 0.658, AUC-PR: 0.017)

## Quick start

```bash
pip install -r requirements.txt
# Place data/modeling_data.csv (see Data section)
python src/run_pipeline.py
```

## Data

- File: `data/modeling_data.csv` (gitignored)
- Source: MIMIC-IV (BigQuery export)
- Features: `hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc`
- Target: ICD-9 714.x (RA), ~1% positive rate
- Split: pre-computed train/test (80/20) — **do not re-split**

## Methods

| # | Method | Status | AUC-ROC | AUC-PR |
|---|--------|--------|---------|--------|
| — | All-features logistic regression (baseline) | ✓ | 0.658 | 0.017 |
| 1 | Threshold optimization | ✓ | 0.617 | 0.0141 |
| 2 | Random formula generation | Planned | — | — |
| 3 | Genetic programming | Planned | — | — |
| 4 | LLM-generated formulas | Planned | — | — |

Full results: [`results/experiment_log.md`](results/experiment_log.md)


read all on server 
```bash
squeue --me -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
```