# Project Status & Remaining Work
_Last updated: 2026-05-24 — items 1–4, 6–8 complete; item 5 remaining_

---

## WHAT IS DONE (verified from files on disk)

- [x] S01 — RA sprint complete. All 4 methods evaluated, sprint report written.
- [x] All 6 diseases have M1–M4 results in both `results/` and `cluster_results/`
- [x] Sanity check master summary with LR baselines for all 6 diseases
- [x] per_k_baselines.csv exists with Formula column for RA (`results/sanity_check/per_k_baselines.csv`)
- [x] Cross-method correlation done for all 6 diseases (`results/cross_method/`)
- [x] NHANES data extracted and evaluated for ra, crhn, psr, lup
      (t1d/t2d intentionally excluded — NHANES cannot distinguish them)
- [x] CI forest plot generated (`results/figures/ci_forest_auc_pr.png`)
- [x] Dashboard data and PNG exist for RA (`results/dashboard/`)
- [x] Literature threshold JSONs for all 6 diseases
- [x] SLURM batch scripts, configs, and disease YAMLs all present

---

## PART 1 — CODE CHANGES (do these first, locally)

- [x] **1. Formula column in per_k_baselines.csv** — validated. Code was already correct;
      9 rows present for RA with Formula column populated. No change needed.

- [x] **2. Remove t1d/t2d from NHANES config** — done in commit c27d5fa.
      Validated: `nhanes_data.py --disease t1d` exits with "No case definition" as expected.

- [x] **3. NHANES RA cycle G / MCQ195 check** — no change needed.
      MCQ195 is present in all 4 cycles (G–J all print True). RA keeps cycles G–J as-is.

- [x] **4. Copy M4 top_cv files from cluster_results into results** — done.
      All 6 files (crhn, lup, psr, ra, t1d, t2d) copied and verified present in
      `results/method4_llm/*/method4_top_cv.csv`.

- [ ] **5. Implement nested CV for M2, M3, M4 (Issue #24)**
      Files: `src/method2_random_formula.py`, `src/method3_gp.py`, `src/method4_llm.py`
      CV exists only for the LR baseline. Add nested CV on train_df for winner selection
      in each method. M2 and M4 pools scored on train_df only (not frozen test).
      Implement and validate on RA before submitting cluster jobs:
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
      Repeat pattern for M3 and M4 before overnight submission.

- [x] **6. Decide on T2D huge GP tier** — option (b): intentionally skipped.
      T2D large tier converged and plateaued at AUC-PR 0.1934 by gen 40 (no improvement
      gen 40→50). Higher prevalence (~12%) means GP already finds strong signal at large.
      `huge` (pop=20,000, gen=500) not worth the compute. Document in thesis.

- [x] **7. Fix run_full_pipeline.py end-to-end chain (S03)** — done.
      Two gaps fixed: (1) added `compare_methods.py` as step 8 (was missing entirely);
      (2) fixed M4 invocation to pass the required `all` subcommand; (3) added inline
      warnings before M3 and M4 steps noting they need cluster in production.
      Verified with `--dry-run`: all 8 steps print correctly.

- [x] **8. Split pipeline into GPU + CPU jobs and rewrite run_all.sh**
      Current `pipeline.sbatch` requests a GPU (`--gpus=rtx_3090:1`) for the entire
      run even though only M4-generate needs one. This wastes GPU allocation for the
      8h CPU work and increases queue wait time.

      **Files to create/modify:**
      - `pipeline_cpu.sbatch` — new CPU-only sbatch (no `--gpus` line). Runs:
        sanity → M1 → M2 → M3 → M4-evaluate → cross_method_correlation →
        compare_methods → compute_ci_all. Accepts `DISEASE` and `SPLIT_SALT` env vars.
      - `medgemma_generate.sbatch` — add `DISEASE` env var support; currently the
        disease is hardcoded. Change the generate call to pass `--disease "$DISEASE"`.
      - `run_all.sh` — rewrite to loop over `conf/disease/*.yaml` slugs and for each:
        1. Submit GPU job (`medgemma_generate.sbatch`), capture job ID with `--parsable`
        2. Submit CPU job (`pipeline_cpu.sbatch`) with `--dependency=afterok:<gpu_id>`
        3. If `data/<disease>_nhanes_data.csv` exists, also submit (or append to CPU job)
           `nhanes_evaluate.py --disease <disease>` — only ra, crhn, psr, lup will match

      **Validate before overnight submission:**
      ```bash
      bash run_all.sh --dry-run
      ```
      Expected: 6 GPU + 6 CPU sbatch lines, with dependency IDs shown, plus nhanes
      evaluate lines for ra, crhn, psr, lup only (not t1d, t2d).

---

## PART 2 — CLUSTER / ALL-DISEASE RUNS (do after Part 1 items 5 and 8 are done)

### Step 1 — Sanity check (local, fast)
```bash
for disease in ra t1d t2d crhn psr lup; do
    python src/sanity_check.py --disease $disease
done
```
Populates `results/sanity_check/{disease}/` (currently empty) and adds
non-RA rows to `per_k_baselines.csv`.
Pass: best single-feature AUC-PR < 0.85 for all diseases.

### Step 2 — Submit overnight cluster jobs
```bash
bash run_all.sh
```
This submits 12 jobs (1 GPU + 1 CPU per disease), with CPU jobs dependent on
their GPU counterpart. NHANES evaluate runs automatically for applicable diseases.

### Step 3 — Pull results and validate (local, after cluster finishes)
```bash
git pull origin main
python src/plot_ci_forest.py
python src/build_dashboard_data.py --disease ra
```
Spot-check: open `results/figures/ci_forest_auc_pr.png` and confirm all 6
diseases appear with CIs. Check that GP whisker does not cross the LR baseline
dashed line for RA (AUC-PR baseline ~0.017).

---

## BLOCKED — Do Not Start Yet

- **EHRSHOT** — no data downloaded. Scripts ready, blocked on dataset access.
  When available: verify OMOP concept IDs in `conf/ehrshot.yaml` first (see TODO2).

- **S08 cross-disease synthesis** — blocked on sprint reports for S04–S07.

- **S09 iterative LLM** — blocked on GP convergence pattern across 2+ diseases.

- **S12 thesis** — blocked on all experiments complete.

---

## KNOWN LIMITATIONS (document in thesis)

- NHANES diagnoses are self-reported, not ICD-code-based — weaker validation than MIMIC-IV or EHRSHOT.
- CBC-only signal is weak for low-prevalence diseases; GP achieves only 0.0179 AUC-PR for RA.
- GP plateaus at ~gen 65 for RA; further scaling has diminishing returns.
- EHRSHOT OMOP concept IDs unverified — external validation pending data access.
