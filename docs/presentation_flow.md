# Presentation Flow — Can a Routine CBC Predict Chronic Disease?

All figures live in `docs/figures/`. The slide order below follows a narrative arc:
**problem → insight → solution → proof of generalization**.

---

## Slide 0 — Title

**Title:** Can a Routine CBC Predict Chronic Disease?

**Content:**
- Ofek Vilkerstone
- Advisors: [Advisor 1 Name] · [Advisor 2 Name]
- [Institution] · [Year]

---

## Slide 1 — Hook / Motivation

**Title:** Can a routine blood count predict 6 chronic diseases?

**Content (bullet points):**
- Complete Blood Count (CBC) is the most ordered lab test — cheap, universal, already collected
- 14 features: 9 standard CBC (hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc) + 5 differentials (neut %, lym %, mono %, eos %, baso %)
- 6 diseases: Rheumatoid Arthritis (RA), Crohn's, Lupus, Psoriasis, T1D, T2D
- Dataset: MIMIC-IV (ICU/hospital EHR), ~1% positive rate, pre-computed patient-level split

**Goal:** Discover interpretable CBC formulas that flag disease risk — small enough to be a clinical index, good enough to generalize across datasets.

---

## Slide 2 — The Framework: Plug-and-Play Architecture

**Figure:** (image in `docs/figures/` — add as `figZ_framework.png` or embed from slide)

**What it shows:** Left-to-right pipeline: Dataset → Pre-processing → M1/M2/M3/M4 methods box → Evaluation panel.

**Talking points:**
- Dataset: MIMIC-IV, ~223,000 patients, ~1% prevalence, 90-day lookback
- Pre-processing: 90-day LB, first 24h index, last-value aggregation, imputation, normalization, 80/20 train/test split
- Four methods in the plug-and-play box: M1 (threshold/Youden), M2 (10k random formulas), M3 (genetic programming, gplearn), M4 (LLM / Med-Gemma 4B seeded prompts)
- Baseline: Logistic Regression on all 14 features
- Evaluation: AUC-PR lift (primary), AUC-ROC, bootstrap CI, vs. baseline, external validation (NHANES, EHRSHOT)

**Key message:** Every method uses the same data pipeline and evaluation harness — results are directly comparable.

---

## Slide 3 — Method Landscape + The GP Problem

**Figure:** `docs/figures/figC_complexity.png`

**What it shows:** Heatmap across 6 diseases × 5 methods. Color = number of features used. Cell text = AUC-PR on MIMIC test set.


**Talking points:**
- M1 (threshold): 1 feature, low AUC-PR — too simple
- M2 (random formula search): 10–14 features, slightly better — too complex
- **M3 (vanilla GP): 4–5 features, moderate AUC-PR — but formulas are biologically arbitrary.** GP explores mathematical space without any clinical compass; it can discover locally optimal expressions that are numerically coincidental
- M4 (LLM): 2–4 features, modest AUC-PR — *but the features make biological sense*
- M5 (seeded GP): 3–5 features, competitive AUC-PR — best complexity/performance ratio

**Key message:** Feature count alone doesn't tell the full story. M3 and M5 may use similar numbers of features, but M5's features are LLM-guided and biologically grounded.

---

## Slide 4 — What the LLM Contributes: RA Case Study

**Figure:** None — text/table slide

**Disease background (opening bullets, ~3 lines):**
- Rheumatoid Arthritis (RA): chronic autoimmune disease, ~1% population prevalence
- Diagnosis is often delayed — symptoms can be subtle for months/years before joint damage is visible on imaging
- A simple CBC flag from a routine blood draw could prompt earlier rheumatology referral at zero additional cost

**Why CBC should signal RA:**
- Chronic inflammation → anemia of chronic disease (↓ hemoglobin, ↓ MCV, ↑ RDW)
- Immune activation → elevated/shifted WBC, altered platelet counts
- These patterns are detectable in a standard blood count ordered at any GP visit

**Formula table (seed2 results):**

| Method | Formula | Features | AUC-PR |
|--------|---------|----------|--------|
| M3 (vanilla GP) | `mono_pct/neut_pct × log(hct − mchc) × …` (complex, 4 terms) | 4 | 0.022 |
| M4 (LLM) | `(mcv − 80) × (rdw + 10) / (hgb + 0.01)` | 3 | 0.028 |
| M5 (seeded GP) | builds on M4 seed → adds hct, mchc interactions | 4–6 | 0.031 |

**Lab name annotations on M4 formula** (show on slide with arrows pointing to each term):
- **MCV** (Mean Corpuscular Volume) → *red cell size — shrinks in anemia of inflammation*
- **RDW** (Red cell Distribution Width) → *size variability — rises when chronic inflammation disrupts erythropoiesis*
- **HGB** (Hemoglobin) → *oxygen carrier — normalizes for overall anaemia severity*

**Interpretation:**
- M3 found a valid formula, but it's a black box — no clinician would recognize it
- M4 identified the **MCV × RDW / HGB** pattern: three anemia-of-inflammation markers combined into a pseudo-index any clinician would read naturally
- M5 took M4's biological seed and GP-refined the math → higher AUC-PR, features still interpretable

**Secondary disease examples (brief, one row each):**
- Crohn's — M4: `(mch + 0.01) / (mcv + 0.01) × (plt + 0.01)` → platelet-MCH ratio (platelet elevation in IBD, MCH in mucosal anaemia)
- T2D — M4: `(mchc − 27) / (mch + 0.1) × (rdw / 20)` → red cell hemoglobin concentration shift with insulin resistance

---

## Slide 5 — M5: Seeded GP Performance

**Figure:** `docs/figures/figA_per_disease_aupr.png`

**What it shows:** 2×3 grid. One panel per disease. Grouped bars: **M3, M4, M5 only** (+ LR baseline). Vertical lines = 95% bootstrap CI brackets.

**Talking points:**
- M5 ≥ M3 on every disease — seeding never hurts
- Largest absolute gains: **T1D (+34–46%)**, **RA (+42–46%)**, **Lupus (+74% on seed3)**
- CI brackets reflect genuine uncertainty at ~1% prevalence; overlapping CIs are expected — what matters is that M5 is never below M3
- M4 (LLM) often has lower point AUC-PR than M3/M5, but M4's formulas are what seed M5
- **Matched-LR benchmark** (see `figJ_matched_lr.png`): LR fit on the same N features as each formula performs within CI for most diseases — the formula's features carry the signal; the formula's value is that it expresses the same relationship as a single arithmetic line any clinician can calculate without software

**Key message:** The LLM provides the biological coordinates; GP provides the optimization power. Together they dominate M3 alone.

*ponytail: drop M1/M2 bars from this figure — they distract from the M3→M4→M5 narrative arc*

---

## Slide 6 — Lift on MIMIC (Training Distribution)

**Figure:** `docs/figures/figF_mimic_lift.png`

**What it shows:** Bar chart, 6 diseases. Y-axis = AUC-PR / MIMIC test-set prevalence (capped at 8×). Values are **mean across all seed runs** (base + seed2/3/4). **Show M3, M4, M5 only** (3 bars per disease).

**Talking points:**
- Lift = how much better than random guessing (lift = 1 means the formula does no better than always predicting positive)
- T2D is the standout: >4× lift — the clearest CBC signal
- Most diseases show 2–5× lift despite 1% prevalence (small signal, hard problem)
- This is the within-distribution baseline; the interesting question is whether it holds externally

---

## Slide 7 — External Validation: EHRSHOT (Stanford EHR)

**Figure:** `docs/figures/figG_ehrshot_lift.png`

**What it shows:** Same lift chart structure (M3/M4/M5 only, 3 bars per disease), but evaluated on the EHRSHOT cohort from Stanford. EHRSHOT prevalences are 3.7–57.5% — **much higher than MIMIC's ~1%** because controls need a CBC in the lookback window.

**Talking points:**
- Despite the cohort shift (different hospital system, very different prevalence), lift is maintained above 1× for most methods/diseases
- AUC-PR / prevalence normalizes out the prevalence difference — if a formula is just memorizing MIMIC's label distribution it would collapse here
- M5 and M4 hold up comparably to M3, confirming the LLM-seeded features are not MIMIC-overfit

---

## Slide 8 — External Validation: NHANES (US General Population)

**Figure:** `docs/figures/figH_nhanes_lift.png`

**What it shows:** NHANES lift for RA and Psoriasis only (the two diseases with available NHANES data). NHANES is a survey of the general US population — very different from hospitalized patients.

**Talking points:**
- RA NHANES prevalence ≈ 7.7% (self-reported, CBC-filtered cohort); PSR ≈ 1.9%
- Formulas derived from ICU/hospital records (MIMIC) still predict in a survey cohort — the CBC signal is not an artifact of hospitalization
- RA lift is lower than MIMIC (expected: higher prevalence = lower maximum lift) but consistently above 1×
- This is the hardest generalization test: different population, different ascertainment method, different prevalence regime

---

## Slide 9 — Conclusion / Takeaway

**Title:** Three findings

1. **LLM as a biological compass:** M4 (LLM) generates compact, clinically interpretable formulas that identify the right feature relationships. GP alone (M3) cannot.

2. **M5 = best of both worlds:** Seeded GP matches or beats vanilla GP on every disease while using fewer, more meaningful features. The seed never hurts; it usually helps (+34–74% on best seeds).

3. **The signal travels:** AUC-PR lift > 1× is maintained across MIMIC → EHRSHOT → NHANES despite prevalence varying from ~1% to ~57%. CBC-derived formulas are not dataset artifacts.

---

## Figure Reference Index

| Figure | File | Used in Slide | Description |
|--------|------|---------------|-------------|
| figA per-disease | `figA_{ra,crhn,lup,psr,t1d,t2d}.png` | 5 (use grid) | Per-disease AUC-PR + 95% CI brackets |
| figA grid | `figA_per_disease_aupr.png` | **5** | 2×3 combined grid |
| figB | `figB_all_diseases_aupr.png` | — (backup for slide 5) | Grouped bars, all diseases together |
| figC | `figC_complexity.png` | **3** | N_Features heatmap + AUC-PR text |
| figD | `figD_ehrshot_generalization.png` | — (backup for slide 7) | MIMIC vs EHRSHOT AUC-ROC side-by-side |
| figE | `figE_nhanes_generalization.png` | — (backup for slide 8) | MIMIC vs NHANES AUC-ROC (RA + PSR) |
| figF | `figF_mimic_lift.png` | **6** | MIMIC lift, mean across seeds |
| figG | `figG_ehrshot_lift.png` | **7** | EHRSHOT lift |
| figH | `figH_nhanes_lift.png` | **8** | NHANES lift (RA + PSR) — new |
| figI formula | `figI_ra_formula.png` | **4** | Annotated RA M4 formula with MCV/RDW/HGB arrows |
| figI travels | `figI_signal_travels.png` | **7+8 merged** | EHRSHOT + NHANES lift side-by-side ("The Signal Travels") |
| figJ | `figJ_matched_lr.png` | — (backup for slide 5) | Formula vs same-feature LR scatter — near-diagonal = competitive |

**Bold** = primary figures for the narrative. figD and figE are AUC-ROC alternatives if the lift story needs backup (AUC-ROC is prevalence-independent so it can be shown on one axis without normalization).

---

## Notes / Open Items

- [x] Slide 4 uses RA as primary disease example (most interpretable M4 formula; best narrative for the "why this matters" framing)
- [x] Slides 6–8 simplified to M3/M4/M5 only (3-bar layout leads into M5 creation story)
- [x] Framework diagram added as Slide 2
- [x] Slide 4: create the actual annotated formula graphic with arrows on MCV/RDW/HGB → `figI_ra_formula.png`
- [x] Slides 6–8: figures regenerated with M3/M4/M5 only (drop M1/M2 bars)
- [x] Option: merge Slides 7+8 → `figI_signal_travels.png` (use instead of figG+figH if slide count is tight)

### Merged Slide 7+8 Option: "The Signal Travels"

```
Title: The Signal Travels — External Validation

┌──────────────────────────────┬──────────────────────────┐
│  EHRSHOT (Stanford EHR)      │  NHANES (US General Pop) │
│  6 diseases                  │  RA + Psoriasis only     │
│  Prevalence: 3.7–57.5%       │  Prevalence: RA≈7.7%,    │
│                              │  PSR≈1.9%                │
│  [bar chart: M3/M4/M5 lift]  │  [bar chart: M3/M4/M5]  │
└──────────────────────────────┴──────────────────────────┘

Key line: "Formulas derived from ICU records (MIMIC) maintain
lift > 1× across a Stanford hospital cohort and a US
survey population — the CBC signal is not a MIMIC artifact."
```

Use this if total slide count is tight (saves one slide). Keep them separate if NHANES needs more explanation (different ascertainment method).
