# Biomarker Discovery Pipeline

Can routine CBC values predict autoimmune and metabolic diseases?

Compares four methods for deriving single-formula biomarkers from CBC data using MIMIC-IV, evaluated across 6 diseases.

## Diseases

| Slug | Disease | ICD-9 |
|------|---------|-------|
| `ra` | Rheumatoid Arthritis | 714.x |
| `crhn` | Crohn's Disease | 555.x |
| `t1d` | Type 1 Diabetes | 250.x1, 250.x3 |
| `t2d` | Type 2 Diabetes | 250.x0, 250.x2 |
| `psr` | Psoriasis | 696.1x |
| `lup` | Lupus (DLE/SLE) | 695.4x, 710.0x |

## Methods

| # | Script | Method |
|---|--------|--------|
| — | `sanity_check.py` | All-features LR baseline + per-k best-subset analysis |
| 1 | `method_threshold.py` | Threshold optimization (literature + data-driven) |
| 2 | `method2_random_formula.py` | Random formula search (10 000 candidates, CV-selected) |
| 3 | `method3_gp.py` | Genetic programming (gplearn, CV-selected, optional LLM seeding) |
| 4 | `method4_llm.py` | LLM-generated formulas (Med-Gemma 4B IT) |
| — | `cross_method_correlation.py` | Pairwise Pearson r between method score vectors |
| — | `nhanes_evaluate.py` | External validation on NHANES cycles G–J |
| — | `ehrshot_evaluate.py` | External validation on EHRSHOT (blocked on data access) |
| — | `dashboard.py` | Streamlit + Plotly interactive results dashboard |

Full results: [`results/experiment_log.md`](results/experiment_log.md)

## Quick start

```bash
pip install -r requirements.txt
# Place data/<disease>_modeling_data.csv (see Data section)
python src/sanity_check.py --disease ra
python src/method2_random_formula.py --disease ra
python src/method3_gp.py --disease ra
python src/method4_llm.py all --disease ra
python src/compare_methods.py              # → results/methods_comparison.csv
python src/cross_method_correlation.py --disease ra  # → results/cross_method/
```

## Data

- Files: `data/<disease>_modeling_data.csv` (gitignored, exported from BigQuery)
- Source: MIMIC-IV v3.1 via `src/queries/cohort_pipeline.sql` + `src/run_pipeline.py`
- Features: 14 CBC features — 9 standard + 5 differential counts
  ```
  hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc,
  neut_pct, lym_pct, mono_pct, eos_pct, baso_pct
  ```
- Target: ~1% positive rate — use imbalanced-aware metrics (AUC-PR primary)
- Split: pre-computed train/test (80/20, patient-level) — **do not re-split**

## Configuration

| File | Purpose |
|------|---------|
| `conf/disease/<slug>.yaml` | ICD patterns, full name, per-feature disease relevance for LLM prompts |
| `conf/ml/defaults.yaml` | Hyperparameters for all method scripts |
| `conf/nhanes.yaml` | NHANES CBC variable names (validation cohort) |
| `conf/ehrshot.yaml` | EHRSHOT concept IDs (validation cohort) |

## Cluster

```bash
bash run_all.sh                  # submit all jobs
squeue --me -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"  # monitor
```
