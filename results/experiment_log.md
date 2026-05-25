# Experiment Log — Biomarker Pipeline

## Cross-Method Equivalence Rule (Issue #25)

**r > 0.95** between two methods' continuous score vectors on the test set means the methods
are **functionally equivalent** for this disease — they are ranking patients nearly identically
despite different formula structures.

**r ≤ 0.95** is evidence of genuinely distinct predictive behavior worth reporting.

Scores compared are the continuous probability/formula outputs before thresholding, not binary
predictions. Pearson r is computed on the frozen test set after each method's CV winner is
selected on train only (Issue #24).

---

## 2026-05-25 — Crohn's seeded-GP comparison (Issue #28)

Config: pop=500, gen=100 (early-stop patience per tier: small=5, medium=10, large=20), seed=42

| Run | Seed file | AUC-ROC | AUC-PR (train best) | Frozen Test AUC-PR | CV AUC-PR Mean | Best formula gen | Seeds injected | Beats vanilla? |
|-----|-----------|---------|---------------------|--------------------|----------------|-----------------|----------------|----------------|
| Vanilla | none | 0.5994 | 0.0244 | **0.0157** | 0.0360 | 1 | — | baseline |
| Gemini 2.5 Pro | gemini_25_pro.csv | 0.5943 | 0.0232 | 0.0120 | 0.0354 | 1 | 150 | No |
| GPT-4o Deep Research | gpt4o_deep_research.csv | 0.5942 | 0.0232 | 0.0139 | 0.0350 | 2 | 150 | No |
| SciSpace Agent | scispace_agent.csv | 0.5994 | 0.0244 | 0.0118 | 0.0377 | 1 | 150 | No |

**Decision: Reject — no seeded run beats vanilla on frozen test AUC-PR.**

Notes:
- All 3 seeded runs injected 150 seed programs into generation 0 successfully.
- GP converges rapidly (best formula found by gen 1–2 in all runs) and plateaus; early-stop fires at gen 6/11/21 for small/medium/large tiers.
- Vanilla AUC-PR (0.0157) and CV mean (0.0360) are slightly better than all seeded variants.
- SciSpace CV mean (0.0377) was marginally higher than vanilla (0.0360) but frozen test was lower (0.0118 vs 0.0157), indicating variance rather than a real signal.
- Negative result: LLM seeding does not improve GP convergence for CBC-based biomarker discovery on Crohn's disease. Issue #29 closed without action per the decision rule.

---
