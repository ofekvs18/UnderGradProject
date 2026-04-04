# Experiment Log — RA Biomarker Discovery

## Pipeline Info (frozen)
- **Disease**: Rheumatoid Arthritis (ICD-9: 714.x)
- **Lookback**: 90 days + first 24h of index admission
- **Train**: 1,273 cases / 115,955 controls
- **Test**: 300 cases / 28,976 controls
- **Features**: HCT, HGB, MCH, MCHC, MCV, PLT, RBC, RDW, WBC

---

## Baseline (Checkpoint 5)

| method | features | AUC-ROC | AUC-PR | P@R50 | F1 | F2 | precision | recall | notes |
|--------|----------|---------|--------|-------|----|----|-----------|--------|-------|
| Logistic regression (single) | RDW | 0.617 | 0.0132 | 0.0145 | 0.028 | 0.066 | 0.014 | 0.797 | Best by AUC-ROC |
| Logistic regression (single) | RBC | 0.586 | 0.0141 | 0.0135 | 0.027 | 0.063 | 0.014 | 0.613 | Best by AUC-PR |
| Logistic regression (single) | HCT | 0.579 | 0.0126 | 0.0134 | 0.026 | 0.062 | 0.014 | 0.657 | |
| Logistic regression (all) | All 9 | 0.658 | 0.0170 | 0.0180 | 0.031 | 0.073 | 0.016 | 0.710 | **Baseline to beat** |

---

## Method 1: Threshold Optimization

_Date: 2026-04-04_

### 1A: Literature-based threshold
| feature | threshold | direction | AUC-ROC | AUC-PR | precision | recall | F1 | F2 | source |
|---------|-----------|-----------|---------|--------|-----------|--------|----|-----|--------|
| RDW | 14.5 | above | 0.617 | 0.0132 | 0.015 | 0.487 | 0.029 | 0.066 | RDW >14.5% associated with inflammatory conditions |
| RBC | 4.0 | below | 0.586 | 0.0141 | 0.013 | 0.703 | 0.025 | 0.060 | Low RBC in anemia of chronic disease |
| HCT | 36.0 | below | 0.579 | 0.0126 | 0.012 | 0.690 | 0.024 | 0.058 | Low hematocrit in anemia of chronic disease |
| HGB | 12.0 | below | 0.569 | 0.0123 | 0.013 | 0.703 | 0.025 | 0.059 | Anemia (Hgb <12 g/dL) common in chronic inflammation |
| PLT | 400.0 | above | 0.572 | 0.0133 | 0.016 | 0.080 | 0.027 | 0.045 | Thrombocytosis (reactive) in chronic inflammation |
| MCV | 80.0 | below | 0.482 | 0.0096 | 0.008 | 0.037 | 0.013 | 0.021 | Microcytosis in anemia of chronic disease |
| WBC | 11.0 | above | 0.479 | 0.0099 | 0.009 | 0.233 | 0.017 | 0.038 | Leukocytosis in active inflammation |
| MCHC | 32.0 | below | 0.467 | 0.0092 | 0.009 | 0.193 | 0.017 | 0.038 | Low MCHC in chronic disease |
| MCH | 27.0 | below | 0.463 | 0.0092 | 0.008 | 0.087 | 0.015 | 0.029 | Low MCH in iron-deficiency/chronic disease anemia |

### 1B: Data-driven threshold (Youden's index)
| feature | threshold | direction | AUC-ROC | AUC-PR | precision | recall | F1 | F2 |
|---------|-----------|-----------|---------|--------|-----------|--------|----|-----|
| RDW | 13.4 | above | 0.6169 | 0.0132 | 0.014 | 0.843 | 0.027 | 0.063 |
| RBC | 2.75 | below | 0.5861 | 0.0141 | 0.011 | 0.067 | 0.019 | 0.060 |
| HCT | 27.1 | below | 0.5786 | 0.0126 | 0.010 | 0.107 | 0.018 | 0.060 |
| HGB | 8.4 | below | 0.5690 | 0.0123 | 0.009 | 0.070 | 0.015 | 0.059 |
| PLT | 260.0 | above | 0.5723 | 0.0133 | 0.014 | 0.390 | 0.027 | 0.060 |
| MCH | 32.3 | above | 0.5373 | 0.0092 | 0.012 | 0.170 | 0.023 | 0.030 |
| MCHC | 32.9 | above | 0.5328 | 0.0092 | 0.011 | 0.627 | 0.022 | 0.048 |
| MCV | 93.0 | above | 0.5176 | 0.0096 | 0.011 | 0.370 | 0.021 | 0.029 |
| WBC | 20.0 | below | 0.5213 | 0.0099 | 0.010 | 0.963 | 0.020 | 0.024 |

### Notes
- **Best by AUC-ROC**: RDW (0.617) — but **best by AUC-PR**: RBC (0.0141); rankings diverge
- AUC-PR is the preferred metric under class imbalance (~1% positive rate); use it as the primary metric going forward
- Data-driven threshold for RDW (13.4) boosts recall from 0.487 → 0.843 vs literature, at cost of lower precision
- No single-feature threshold beats the all-features logistic regression baseline (AUC-ROC=0.658, AUC-PR=0.017)

---

## Method 2: Random Formula Generation

_Date: ____

| formula | features_used | AUC | precision | recall |
|---------|---------------|-----|-----------|--------|
| | | | | |

### Best formula
```
(formula here)
```

### Notes

---

## Method 3: Genetic Programming

_Date: ____

| generation | best_formula | AUC | precision | recall |
|------------|-------------|-----|-----------|--------|
| | | | | |

### Best formula
```
(formula here)
```

### Notes

---

## Method 4: LLM-Generated Formulas

_Date: ____

| prompt_strategy | formula | AUC | precision | recall |
|----------------|---------|-----|-----------|--------|
| | | | | |

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
| Random formulas | | | | |
| Genetic programming | | | | |
| LLM formulas | | | | |
