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

| # | Script | Method | Status |
|---|--------|--------|--------|
| — | `sanity_check.py` | All-features logistic regression (baseline) | ✓ |
| 1 | `method_threshold.py` | Threshold optimization (literature + data-driven) | ✓ |
| 2 | `method2_random_formula.py` | Random formula generation | ✓ |
| 3 | `method3_gp.py` | Genetic programming (gplearn) | ✓ |
| 4 | `method4_llm.py` | LLM-generated formulas (Med-Gemma 4B) | ✓ |

Full results: [`results/experiment_log.md`](results/experiment_log.md)

To compare all methods across diseases and split variants:
```bash
python src/compare_methods.py
# → results/methods_comparison.csv
```


read all on server 
```bash
squeue --me -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
```