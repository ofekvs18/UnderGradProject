# Experiment Log — RA Biomarker Discovery

## Pipeline Info (frozen)
- **Disease**: Rheumatoid Arthritis (ICD-9: 714.x)
- **Lookback**: 90 days + first 24h of index admission
- **Train**: 1,273 cases / 115,955 controls
- **Test**: 300 cases / 28,976 controls
- **Features**: HCT, HGB, MCH, MCHC, MCV, PLT, RBC, RDW, WBC

---

## Baseline (Checkpoint 5)

_Single-feature logistic regression (class_weight="balanced", Youden threshold), sorted by AUC-PR_

| feature | AUC-ROC | AUC-PR | P@R25 | P@R50 | P@R75 | F1 | F2 | precision | recall |
|---------|---------|--------|-------|-------|-------|----|----|-----------|--------|
| RBC | 0.5861 | 0.0141 | 0.0123 | 0.0135 | 0.0127 | 0.027 | 0.063 | 0.014 | 0.613 |
| PLT | 0.5723 | 0.0133 | 0.0141 | 0.0137 | 0.0113 | 0.027 | 0.062 | 0.014 | 0.587 |
| RDW | 0.6169 | 0.0132 | 0.0135 | 0.0145 | 0.0141 | 0.028 | 0.066 | 0.014 | 0.797 |
| HCT | 0.5786 | 0.0126 | 0.0122 | 0.0134 | 0.0120 | 0.026 | 0.062 | 0.014 | 0.657 |
| HGB | 0.5690 | 0.0123 | 0.0125 | 0.0122 | 0.0123 | 0.025 | 0.059 | 0.013 | 0.767 |
| MCH | 0.5373 | 0.0122 | 0.0119 | 0.0115 | 0.0108 | 0.023 | 0.053 | 0.012 | 0.507 |
| WBC | 0.5213 | 0.0110 | 0.0109 | 0.0116 | 0.0108 | 0.023 | 0.054 | 0.012 | 0.590 |
| MCHC | 0.5328 | 0.0113 | 0.0111 | 0.0113 | 0.0110 | 0.023 | 0.054 | 0.011 | 0.720 |
| MCV | 0.4824 | 0.0096 | 0.0083 | 0.0101 | 0.0102 | 0.021 | 0.049 | 0.010 | 0.717 |
| **All 9 (LR)** | **0.6580** | **0.0170** | **0.0210** | **0.0180** | **0.0144** | **0.031** | **0.073** | **0.016** | **0.710** |

**Baseline to beat: AUC-ROC=0.658, AUC-PR=0.0170**

---

## Method 1: Threshold Optimization

_Date: 2026-04-04_

_Sorted by AUC-PR. AUC is threshold-independent; precision/recall/F scores are at the specified threshold._

### 1A: Literature-based threshold
| feature | threshold | direction | AUC-ROC | AUC-PR | P@R25 | P@R50 | P@R75 | precision | recall | F1 | F2 |
|---------|-----------|-----------|---------|--------|-------|-------|-------|-----------|--------|----|-----|
| RBC | 4.0 | below | 0.5861 | 0.0141 | 0.0123 | 0.0135 | 0.0127 | 0.013 | 0.703 | 0.025 | 0.060 |
| PLT | 400.0 | above | 0.5723 | 0.0133 | 0.0141 | 0.0137 | 0.0113 | 0.016 | 0.080 | 0.027 | 0.045 |
| RDW | 14.5 | above | 0.6169 | 0.0132 | 0.0135 | 0.0145 | 0.0141 | 0.015 | 0.487 | 0.029 | 0.066 |
| HCT | 36.0 | below | 0.5786 | 0.0126 | 0.0122 | 0.0134 | 0.0120 | 0.012 | 0.690 | 0.024 | 0.058 |
| HGB | 12.0 | below | 0.5690 | 0.0123 | 0.0125 | 0.0122 | 0.0123 | 0.013 | 0.703 | 0.025 | 0.059 |
| WBC | 11.0 | above | 0.4787 | 0.0099 | 0.0089 | 0.0096 | 0.0101 | 0.009 | 0.233 | 0.017 | 0.038 |
| MCV | 80.0 | below | 0.4824 | 0.0096 | 0.0083 | 0.0101 | 0.0102 | 0.008 | 0.037 | 0.013 | 0.021 |
| MCH | 27.0 | below | 0.4627 | 0.0092 | 0.0082 | 0.0094 | 0.0097 | 0.008 | 0.087 | 0.015 | 0.029 |
| MCHC | 32.0 | below | 0.4672 | 0.0092 | 0.0080 | 0.0092 | 0.0098 | 0.009 | 0.193 | 0.017 | 0.038 |

### 1B: Data-driven threshold (Youden's index on train set)
| feature | threshold | direction | AUC-ROC | AUC-PR | P@R25 | P@R50 | P@R75 | precision | recall | F1 | F2 |
|---------|-----------|-----------|---------|--------|-------|-------|-------|-----------|--------|----|-----|
| RBC | 3.99 | below | 0.5861 | 0.0141 | 0.0123 | 0.0135 | 0.0127 | 0.013 | 0.703 | 0.025 | 0.060 |
| PLT | 260.0 | above | 0.5723 | 0.0133 | 0.0141 | 0.0137 | 0.0113 | 0.014 | 0.390 | 0.027 | 0.060 |
| RDW | 13.4 | above | 0.6169 | 0.0132 | 0.0135 | 0.0145 | 0.0141 | 0.014 | 0.843 | 0.027 | 0.063 |
| HCT | 35.3 | below | 0.5786 | 0.0126 | 0.0122 | 0.0134 | 0.0120 | 0.013 | 0.670 | 0.026 | 0.060 |
| HGB | 11.9 | below | 0.5690 | 0.0123 | 0.0125 | 0.0122 | 0.0123 | 0.013 | 0.703 | 0.025 | 0.059 |
| WBC | 20.0 | above | 0.4787 | 0.0099 | 0.0089 | 0.0096 | 0.0101 | 0.010 | 0.037 | 0.016 | 0.024 |
| MCV | 83.0 | below | 0.4824 | 0.0096 | 0.0083 | 0.0101 | 0.0102 | 0.008 | 0.093 | 0.014 | 0.029 |
| MCH | 27.3 | below | 0.4627 | 0.0092 | 0.0082 | 0.0094 | 0.0097 | 0.008 | 0.103 | 0.015 | 0.030 |
| MCHC | 34.7 | below | 0.4672 | 0.0092 | 0.0080 | 0.0092 | 0.0098 | 0.010 | 0.863 | 0.020 | 0.048 |

### Notes
- **Best by AUC-ROC**: RDW (0.617) — **best by AUC-PR**: RBC (0.0141); rankings diverge
- AUC-PR is the primary metric going forward (~1% positive rate makes AUC-ROC misleading)
- Data-driven RDW threshold (13.4) boosts recall from 0.487 → 0.843 vs literature, at cost of slightly lower precision
- Precision at all recall levels is ~1–2%, reflecting severe class imbalance; even the best model flags ~99% false positives
- No single-feature threshold beats the all-features LR baseline (AUC-ROC=0.658, AUC-PR=0.017)

---

## Method 2: Random Formula Generation

_Date: 2026-04-05_

_10,000 unique random formulas; 2–4 features; ops: +, -, *, /, sqrt, log, square. Sorted by AUC-PR._

| formula | features_used | AUC-ROC | AUC-PR | P@R25 | P@R50 | P@R75 | F1 | F2 | precision | recall |
|---------|---------------|---------|--------|-------|-------|-------|----|----|-----------|--------|
| sqrt(hct) / rbc | hct, rbc | 0.5796 | 0.0174 | 0.0129 | 0.0130 | 0.0121 | 0.026 | 0.061 | 0.013 | 0.703 |
| mch * sqrt(hct) / hgb | mch, hct, hgb | 0.5792 | 0.0174 | 0.0133 | 0.0129 | 0.0120 | 0.026 | 0.061 | 0.013 | 0.713 |
| rbc / sqrt(hct) | rbc, hct | 0.5796 | 0.0174 | 0.0129 | 0.0130 | 0.0121 | 0.026 | 0.061 | 0.013 | 0.703 |
| (mchc - log(wbc)) / sqrt(rbc) * log(plt) | mchc, wbc, rbc, plt | 0.6256 | 0.0171 | 0.0209 | 0.0165 | 0.0128 | 0.028 | 0.065 | 0.014 | 0.713 |

### Best formula (by AUC-PR)
```
sqrt(abs(hct)) / (abs(rbc) + 1e-6)
```

### Best formula (by AUC-ROC)
```
((mchc - log(abs(wbc)+1)) / (sqrt(abs(rbc)) + 1e-6)) * log(abs(plt)+1)
```

### Notes
- 10,000 valid formulas evaluated; 0 invalid/skipped (all CBC values non-negative)
- Only 7 / 10,000 formulas beat baseline AUC-PR (0.0170) — the top few are marginally above
- Top formulas consistently involve HCT/RBC ratios — capturing mean corpuscular-like indices
- Best AUC-PR (0.0174) marginally beats baseline but AUC-ROC (0.580) is well below it
- Formula #7 achieves highest AUC-ROC (0.6256) among random formulas, close to baseline (0.658)
- Median AUC-PR across all formulas: 0.0125 — most random combos underperform baseline

---

## Method 3: Genetic Programming

_Date: ____

| generation | best_formula | AUC-ROC | AUC-PR | P@R50 | F1 | F2 | precision | recall |
|------------|-------------|---------|--------|-------|----|----|-----------|--------|
| | | | | | | | | |

### Best formula
```
(formula here)
```

### Notes

---

## Method 4: LLM-Generated Formulas

_Date: ____

| prompt_strategy | formula | AUC-ROC | AUC-PR | P@R50 | F1 | F2 | precision | recall |
|----------------|---------|---------|--------|-------|----|----|-----------|--------|
| | | | | | | | | |

### Best formula
```
(formula here)
```

### Notes

---

## Summary Comparison

_Fill this in as you complete each method_

| method | best AUC-ROC | best AUC-PR | best feature | beats baseline? |
|--------|-------------|------------|--------------|-----------------|
| Logistic regression (all features) | 0.658 | 0.017 | All 9 CBC | — (this IS the baseline) |
| Literature threshold (1A) | 0.617 | 0.0141 | RDW (ROC) / RBC (PR) | No |
| Data-driven threshold (1B) | 0.617 | 0.0141 | RDW (ROC) / RBC (PR) | No |
| Random formulas (10k) | 0.6256 | 0.0174 | sqrt(hct)/rbc (PR) / mchc-log(wbc)/sqrt(rbc)*log(plt) (ROC) | Marginally (PR only, +0.0004) |
| Genetic programming | | | | |
| LLM formulas | | | | |
