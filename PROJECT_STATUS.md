# Project Status
_Last updated: 2026-05-24_

---

## What is complete

### Infrastructure & scripts
- SLURM batch scripts split into GPU (`medgemma_generate.sbatch`) and CPU-only (`pipeline_cpu.sbatch`)
- `run_all.sh` rewritten: submits paired GPU+CPU jobs per disease with `afterok` dependency; NHANES evaluate for ra/crhn/psr/lup only; `--dry-run` flag validated
- `run_full_pipeline.py` end-to-end chain fixed (8 steps, M4 `all` subcommand, cluster warnings)

### MIMIC-IV experiments
- All 6 diseases (ra, t1d, t2d, crhn, psr, lup) have M1–M4 results in both `results/` and `cluster_results/`
- Sanity check master summary with LR baselines for all 6 diseases
- `results/sanity_check/per_k_baselines.csv` — 9 rows for RA, Formula column populated
- Cross-method correlation done for all 6 diseases (`results/cross_method/`)
- M4 `top_cv` files copied from cluster into `results/method4_llm/*/method4_top_cv.csv` for all 6 diseases
- Literature threshold JSONs for all 6 diseases

### External validation (NHANES)
- t1d/t2d removed from `conf/nhanes.yaml` — `nhanes_data.py --disease t1d` exits with "No case definition"
- MCQ195 confirmed present in all 4 cycles (G–J); RA keeps G–J
- NHANES data extracted and evaluated for ra, crhn, psr, lup (t1d/t2d intentionally excluded)

### Outputs
- CI forest plot: `results/figures/ci_forest_auc_pr.png`
- Dashboard data and PNG for RA: `results/dashboard/`
- S01 sprint report written

### Design decisions
- T2D `huge` GP tier intentionally skipped — converged at AUC-PR 0.1934 by gen 40, no improvement gen 40→50. Document in thesis.

---

## What needs to be done

### 1. Nested CV for M2, M3, M4 — Issue #24  `[local, do first]`

CV currently exists only for the LR baseline. Add nested CV on `train_df` for winner selection in each method (M2 and M4 pools scored on train only, frozen test only evaluated once for the winner).

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
Repeat pattern for M3 and M4.

---

### 2. All-disease cluster run  `[cluster, after #1]`

**Step 1 — Sanity check locally:**
```bash
for disease in ra t1d t2d crhn psr lup; do
    python src/sanity_check.py --disease $disease
done
```
Pass condition: best single-feature AUC-PR < 0.85 for all diseases.

**Step 2 — Submit overnight jobs:**
```bash
bash run_all.sh
```
Submits 12 jobs (1 GPU + 1 CPU per disease) with CPU jobs dependent on their GPU counterpart. NHANES evaluate runs automatically for ra/crhn/psr/lup.

**Step 3 — Pull and validate after cluster finishes:**
```bash
git pull origin main
python src/plot_ci_forest.py
python src/build_dashboard_data.py --disease ra
```
Spot-check: `results/figures/ci_forest_auc_pr.png` shows all 6 diseases. GP whisker for RA should not cross the LR baseline dashed line (~0.017 AUC-PR).

---

### 3. Output validation (local, after cluster results land)

These checks verify outputs are not corrupted or mislabeled. Run once per disease after results are pulled.

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
print(f"Best DD feature: {best['feature']}  AUC-PR={best['auc_pr']:.4f}")
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
from pathlib import Path

with open("results/method4_llm/ra/raw_outputs.json") as f:
    entries = json.load(f)
ok = [e for e in entries if e["status"] == "ok"]
print(f"Total: {len(entries)}  OK: {len(ok)}")
assert len(ok) >= 20, "Too many inference errors"

results = pd.read_csv("results/method4_llm/ra/method4_results.csv")
assert len(results) >= 30, f"Only {len(results)} valid formulas — check parser"
best = results.sort_values("auc_pr", ascending=False).iloc[0]
print(f"Best formula: {best['formula']}  AUC-PR: {best['auc_pr']:.4f}")
```

**NHANES evaluation (ra, crhn, psr, lup):**
```python
import pandas as pd
df = pd.read_csv("results/nhanes/ra_evaluation.csv")
assert df["method"].nunique() == 4
assert df["auc_pr"].between(0.001, 0.999).all()
```

**Dashboard GP formula normalization (should be infix, not S-expression):**
```python
import pandas as pd
df = pd.read_csv("results/dashboard/ra_dashboard_data.csv")
gp_rows = df[df["method"] == "m3"]
sample = gp_rows.sort_values("auc_pr", ascending=False).iloc[0]["formula_display"]
assert "mul(" not in sample and "div(" not in sample, "S-expression not converted to infix"
print(sample)
```

---

## Blocked — do not start

| Item | Blocked by |
|------|-----------|
| **EHRSHOT external validation** | Dataset not downloaded. Scripts ready. When available: verify OMOP concept IDs in `conf/ehrshot.yaml` against the local vocabulary before running anything — wrong IDs produce silent empty columns. Then run `ehrshot_data.py → ehrshot_sanity.py → ehrshot_evaluate.py` per disease. |
| **S08 cross-disease synthesis** | Sprint reports for S04–S07 not written yet. |
| **S09 iterative LLM** | Needs GP convergence data from 2+ diseases. |
| **S12 thesis** | Blocked on all experiments complete. |

---

## Known limitations (document in thesis)

- NHANES diagnoses are self-reported, not ICD-code-based — weaker validation than MIMIC-IV or EHRSHOT.
- CBC-only signal is weak for low-prevalence diseases; GP achieves only 0.0179 AUC-PR for RA.
- GP plateaus at ~gen 65 for RA; further scaling has diminishing returns.
- T2D `huge` GP tier skipped — converged at gen 40 with no further improvement.
- EHRSHOT OMOP concept IDs unverified — external validation pending data access.
