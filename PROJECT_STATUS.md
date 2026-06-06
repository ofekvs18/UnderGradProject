# Project Status
_Last updated: 2026-06-06 (synced to issues.md)_

---

## What is complete

### Feature implementation (TODOs 1–5)
- **Per-k biomarker comparison (TODO 1)** — exhaustive best-subset LR baseline per k=1..9 in `sanity_check.py`, cached to `per_k_baselines.csv`. All method scripts track best formula per k and compare against baseline. Per-k CSVs in `results/<method>/<disease>/per_k_best.csv`.
- **EHRSHOT scripts (TODO 2)** — `src/ehrshot_data.py`, `src/ehrshot_sanity.py`, `src/ehrshot_evaluate.py` + `conf/ehrshot.yaml`. Blocked on data access; scripts are ready.
- **NHANES scripts (TODO 3)** — `src/nhanes_data.py`, `src/nhanes_sanity.py`, `src/nhanes_evaluate.py` + `conf/nhanes.yaml`. Cycles G–J, survey weights not applied.
- **Dashboard (TODO 4)** — `src/build_dashboard_data.py` + `src/dashboard.py` (Streamlit + Plotly). Formula normalization: GP S-expressions converted to infix. External validation toggle via `--include-external`.
- **Confidence intervals (TODO 5)** — `src/compute_ci.py` (stratified bootstrap, DeLong AUC-ROC), `src/compute_ci_all.py` (backfill script), `src/plot_ci_forest.py` (forest plot). CI columns added to all master summaries.

### Infrastructure & scripts
- SLURM split into GPU-only (`medgemma_generate.sbatch`) and CPU-only (`pipeline_cpu.sbatch`)
- `run_all.sh` rewritten: GPU+CPU+POST jobs per disease with `afterok` chain; `--dry-run`, `--purge` flags; `--purge` now preserves `data/nhanes/` (XPT files)
- `post_compute.sbatch` / `src/post_compute.py`: 8-step post pipeline — MIMIC CIs, NHANES data+eval+CIs, EHRSHOT BQ extraction+eval, dashboard, forest plot; controlled by `SKIP_NHANES`, `SKIP_EHRSHOT`, `NHANES_DIR`, `EHRSHOT_KEY_FILE` env vars
- `run_full_pipeline.py` end-to-end chain fixed (8 steps, M4 `all` subcommand, cluster warnings)

### MIMIC-IV experiments
- All 6 diseases (ra, t1d, t2d, crhn, psr, lup) have M1–M4 results in `results/` and `cluster_results/`
- Sanity check master summary with LR baselines for all 6 diseases
- `results/sanity_check/per_k_baselines.csv` — 9 rows for RA, Formula column populated
- Cross-method correlation done for all 6 diseases (`results/cross_method/`) — Issue #25
- M4 `top_cv` files copied into `results/method4_llm/*/method4_top_cv.csv` for all 6 diseases
- Literature threshold JSONs for all 6 diseases
- **Nested CV for M2, M3, M4 (Issue #24)** — CV winner selection on `train_df` only; frozen test evaluated once. `CV_AUC_PR_Mean`/`CV_AUC_PR_Std` columns in all method master summaries.
- **LLM seed files (Issue #26)** — `data/llm_seeds/<disease>/` naming standard; seed CSVs for ra, crhn, psr, lup, t1d. No seeds for t2d (vanilla GP only).
- **Seeded-GP warm start (Issue #27)** — `--seed-file` / `--seed-fraction` flags in `method3_gp.py`; `SEED_VAR_MAP` + `translate_seed_expression()` in `utils.py`; `Seed_File` / `Seed_Count_Used` columns in M3 master.
- **Seeded-GP vs vanilla comparison on Crohn's (Issue #28)** — 4 runs (vanilla + 3 LLM agents), pop=500, gen=100, seed=42. Results logged in `results/experiment_log.md`.
- **Seeded-GP generalisation (Issue #29)** — closed as negative result; no seeded run beat vanilla across all diseases. Documented in `experiment_log.md`.
- **14-feature expansion code changes (Issue #30, Steps 1–7)** — SQL pivot, `CBC_FEATURE_LIST`, `SEED_VAR_MAP`, `method4_llm.py` prompts, `conf/ml/defaults.yaml` baselines reset, `conf/ehrshot_bq.yaml`, `conf/nhanes.yaml`, `STANDARDS.md`, `CLAUDE.md` all updated for 14 features (9 standard + 5 differential).

### External validation (NHANES)
- t1d/t2d removed from `conf/nhanes.yaml` — `nhanes_data.py --disease t1d` exits with "No case definition"
- MCQ195 confirmed present in all 4 NHANES cycles (G–J); RA keeps G–J
- NHANES data extracted and evaluated for ra, crhn, psr, lup

### External validation (EHRSHOT, BigQuery)
- EHRSHOT loaded into BigQuery dataset `EHRSHOTS_DATA` (OMOP CDM: condition_occurrence, measurement, visit_occurrence)
- New `src/ehrshot_bq_data.py` + `conf/ehrshot_bq.yaml` extract cohorts directly from BigQuery (distinct from the parquet-based `ehrshot_data.py`)
- OMOP `measurement_concept_id` verified against `measurement_source_value` — the initial config IDs were wrong (e.g. 3000963 = HGB not WBC); all 14 corrected
- **EHRSHOT uses dotted ICD-9 + ICD-10**, so MIMIC's dotless `conf/disease/*.yaml` patterns failed (psr/lup matched 0 cases; t1d leaked type-2). Added per-disease `disease_icd_patterns` covering both vocabularies
- Cohorts extracted for all 6 diseases: ra (1,156), crhn (1,022), t1d (1,241), t2d (1,665), psr (1,216), lup (1,115). Prevalence 3.7%–57.5%
- **Next:** run `ehrshot_sanity.py` / `ehrshot_evaluate.py` on these CSVs (may need path/format alignment with the BigQuery output)

### Outputs
- CI forest plot: `results/figures/ci_forest_auc_pr.png`
- Dashboard data and PNG for RA: `results/dashboard/`
- S01 sprint report written

### Design decisions locked
- T2D `huge` GP tier skipped — converged at AUC-PR 0.1934 by gen 40, no further improvement. Document in thesis.
- Survey weights not applied in NHANES — each participant treated equally for comparability with MIMIC-IV.
- Per-k constraint: upper bound (at most k features) for M3/M4, exactly k for M2.

---

## What needs to be done

### 1. Issue #30 Step 8 — Full data purge and cluster rerun  `[cluster]`

All 7 code changes for the 14-feature expansion are complete. The remaining work is to regenerate all data and results with the new schema.

**Purge and resubmit in one step** (preserves `data/nhanes/` XPT files):
```bash
bash run_all.sh --purge
```
This deletes `data/*` (except `data/nhanes/`), `results/`, and `cluster_results/`, then immediately submits all jobs. Prompts for confirmation before deleting.

If you need to rerun the MIMIC extraction locally first (sanity check before cluster submission):
```bash
for disease in ra t1d t2d crhn psr lup; do
    python src/run_pipeline.py disease=$disease
    python src/sanity_check.py --disease $disease
done
```
Pass: best single-feature AUC-PR < 0.85 for all diseases.

**Set BQ credentials if not using ADC:**
```bash
EHRSHOT_KEY_FILE=.secrets/bq_sa.json bash run_all.sh --purge
```

---

### 2. All-disease cluster run  `[cluster, after Issue #30 rerun]`

**Step 1 — Sanity check locally:**
```bash
for disease in ra t1d t2d crhn psr lup; do
    python src/sanity_check.py --disease $disease
done
```
Pass: best single-feature AUC-PR < 0.85 for all diseases.

**Step 2 — Submit overnight jobs:**
```bash
bash run_all.sh
```
Submits 18 jobs (1 GPU + 1 CPU + 1 POST per disease). POST jobs run NHANES eval (ra/crhn/psr/lup) and EHRSHOT eval (all 6 diseases) automatically.

**Step 3 — Pull and validate after cluster finishes:**
```bash
git pull origin main
python src/plot_ci_forest.py
python src/build_dashboard_data.py --disease ra
```
Spot-check: forest plot shows all 6 diseases; GP whisker for RA does not cross LR baseline (~0.017).

---

### 3. Output validation  `[local, after cluster results land]`

**M3 per-k formula tracking:**
```python
import pandas as pd, glob
for path in glob.glob("results/method3_gp/ra/*/per_k_best.csv"):
    df = pd.read_csv(path)
    k1 = df[df["k"] == 1].iloc[0]
    print(f"k=1 formula: {k1['formula']}")
    # Expected: single-feature expression like "rdw" or "log(plt)"
```

**M1 threshold outputs:**
```python
import pandas as pd
df_dd = pd.read_csv("results/method1_threshold/ra/datadriven_results.csv")
best = df_dd.loc[df_dd["auc_pr"].idxmax()]
print(f"Best DD: {best['feature']}  AUC-PR={best['auc_pr']:.4f}")
# For RA: expect RBC or RDW at top

master = pd.read_csv("results/method1_threshold/master_m1_summary.csv")
ra_row = master[master["Disease"] == "ra"].iloc[-1]
print(f"CV-selected: {ra_row['CV_Selected_Feature']}")
print(f"Frozen test AUC-PR: {ra_row['Frozen_Test_AUC_PR']:.4f}")
# Expected ~0.013–0.016
```

**M4 LLM outputs:**
```python
import json, pandas as pd
with open("results/method4_llm/ra/raw_outputs.json") as f:
    entries = json.load(f)
ok = [e for e in entries if e["status"] == "ok"]
assert len(ok) >= 20, "Too many inference errors"
results = pd.read_csv("results/method4_llm/ra/method4_results.csv")
assert len(results) >= 30, f"Only {len(results)} valid formulas — check parser"
best = results.sort_values("auc_pr", ascending=False).iloc[0]
print(f"Best: {best['formula']}  AUC-PR: {best['auc_pr']:.4f}")
```

**NHANES evaluation:**
```python
import pandas as pd
df = pd.read_csv("results/nhanes/ra_evaluation.csv")
assert df["method"].nunique() == 4
assert df["auc_pr"].between(0.001, 0.999).all()
```

**Dashboard GP formula normalization (infix, not S-expression):**
```python
import pandas as pd
df = pd.read_csv("results/dashboard/ra_dashboard_data.csv")
sample = df[df["method"] == "m3"].sort_values("auc_pr", ascending=False).iloc[0]["formula_display"]
assert "mul(" not in sample and "div(" not in sample, "S-expression not converted to infix"
print(sample)
```

---

### 4. Verify `ehrshot_sanity.py` reads the BigQuery cohorts  `[local]`

`ehrshot_bq_data.py` + `ehrshot_evaluate.py` now run automatically in the POST cluster job (step 5–6 of `post_compute.py`). However, `ehrshot_sanity.py` is not yet wired into the pipeline — run it manually once after the first cluster rerun to confirm the BigQuery CSVs are valid:
```bash
for disease in ra crhn t1d t2d psr lup; do
    python src/ehrshot_sanity.py --disease $disease
done
```
The sanity script was written for the parquet path; confirm it reads `data/<slug>_ehrshot_data.csv` (BigQuery output) and adjust if needed.

---

## Blocked — do not start

### EHRSHOT evaluation (now automated)
`ehrshot_bq_data.py` + `ehrshot_evaluate.py` are wired into the POST cluster job (steps 5–6 of `post_compute.py`) and run automatically for all 6 diseases after each cluster submission. No manual intervention needed.

`ehrshot_sanity.py` remains manual — see item 4 above.

Prevalence is intentionally high (3.7%–57.5%) due to the random control index-date design — do NOT treat high prevalence as a label-inversion bug.

### Other blocked items

| Item | Blocked by |
|------|-----------|
| **S08 cross-disease synthesis** | Sprint reports for S04–S07 not written |
| **S09 iterative LLM** | Needs GP convergence data from 2+ diseases |
| **S12 thesis** | Blocked on all experiments complete |

---

## Known limitations (document in thesis)

- NHANES diagnoses are self-reported, not ICD-code-based — weaker than MIMIC-IV or EHRSHOT.
- CBC-only signal is weak for low-prevalence diseases; GP achieves only 0.0179 AUC-PR for RA.
- GP plateaus at ~gen 65 for RA; further scaling has diminishing returns.
- T2D `huge` GP tier skipped — converged at gen 40 with no further improvement.
- EHRSHOT control prevalence is high (random index-date sampling drops controls without a CBC in the lookback window); report AUC-ROC alongside AUC-PR for the EHRSHOT cohort.
