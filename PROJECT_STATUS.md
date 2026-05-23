# Project Status
_Last updated: 2026-05-24_

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
- `run_all.sh` rewritten: paired GPU+CPU jobs per disease with `afterok` dependency; NHANES evaluate for ra/crhn/psr/lup only; `--dry-run` flag
- `run_full_pipeline.py` end-to-end chain fixed (8 steps, M4 `all` subcommand, cluster warnings)

### MIMIC-IV experiments
- All 6 diseases (ra, t1d, t2d, crhn, psr, lup) have M1–M4 results in `results/` and `cluster_results/`
- Sanity check master summary with LR baselines for all 6 diseases
- `results/sanity_check/per_k_baselines.csv` — 9 rows for RA, Formula column populated
- Cross-method correlation done for all 6 diseases (`results/cross_method/`)
- M4 `top_cv` files copied into `results/method4_llm/*/method4_top_cv.csv` for all 6 diseases
- Literature threshold JSONs for all 6 diseases

### External validation (NHANES)
- t1d/t2d removed from `conf/nhanes.yaml` — `nhanes_data.py --disease t1d` exits with "No case definition"
- MCQ195 confirmed present in all 4 NHANES cycles (G–J); RA keeps G–J
- NHANES data extracted and evaluated for ra, crhn, psr, lup

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

### 1. Nested CV for M2, M3, M4 — Issue #24  `[local, do first]`

CV currently exists only for the LR baseline. Add nested CV on `train_df` for winner selection. M2 and M4 pools scored on train only; frozen test evaluated once for the winner.

**Files:** `src/method2_random_formula.py`, `src/method3_gp.py`, `src/method4_llm.py`

Validate on RA before cluster submission:
```bash
python src/method2_random_formula.py --disease ra
python -c "
import pandas as pd
df = pd.read_csv('results/method2_random/ra/top_formulas_cv.csv')
assert 'cv_auc_pr_mean' in df.columns, 'CV column missing from M2 output'
assert df['frozen_test_auc_pr'].notna().any(), 'Frozen test never evaluated'
print('M2 CV OK — winner AUC-PR:', df['frozen_test_auc_pr'].max())
"
```
Repeat for M3 and M4.

---

### 2. All-disease cluster run  `[cluster, after #1]`

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
Submits 12 jobs (1 GPU + 1 CPU per disease); NHANES evaluate runs automatically for ra/crhn/psr/lup.

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

## Blocked — do not start

### EHRSHOT external validation
Scripts are fully implemented and ready. Blocked on dataset access only.

When EHRSHOT is available:

**1. Verify OMOP concept IDs** (do this before running anything — wrong IDs produce silent empty columns):
```python
import pandas as pd
cbc_omop = {
    "wbc": "OMOP/3000963", "rbc": "OMOP/3000905", "hgb": "OMOP/3024731",
    "hct": "OMOP/3009542", "mcv": "OMOP/3015182", "mch": "OMOP/3012030",
    "mchc": "OMOP/3010813", "plt": "OMOP/3007461", "rdw": "OMOP/3014111",
}
events = pd.read_parquet("path/to/ehrshot/data/meds_events.parquet")
for feat, code in cbc_omop.items():
    n = (events["code"] == code).sum()
    print(f"{feat}: {code} → {n:,} events")
# Any feature with 0 events needs the concept ID corrected in conf/ehrshot.yaml
```

**2. Verify ICD prefix format:**
```python
icd_events = events[events["code"].str.startswith("ICD")]
print(icd_events["code"].value_counts().head(20))
# Expected: "ICD9CM/714.0" — update conf/ehrshot.yaml if format differs
```

**3. Extract, sanity-check, and evaluate:**
```bash
python src/ehrshot_data.py --ehrshot-dir /path/to/ehrshot --disease ra
python src/ehrshot_sanity.py --disease ra
python src/ehrshot_evaluate.py --disease ra

for disease in crhn psr lup; do
    python src/ehrshot_data.py --ehrshot-dir /path/to/ehrshot --disease $disease
    python src/ehrshot_sanity.py --disease $disease
    python src/ehrshot_evaluate.py --disease $disease
done
```
Expected RA prevalence: ~0.5–2%. If 0%: ICD prefix wrong. If 50%: label logic inverted.

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
- EHRSHOT OMOP concept IDs unverified — external validation pending data access.
