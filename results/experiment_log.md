# Experiment Log — RA Biomarker Discovery

## Pipeline Info (frozen)
- **Disease**: Rheumatoid Arthritis (ICD-9: 714.x)
- **Lookback**: 90 days + first 24h of index admission
- **Train**: 1,273 cases / 115,955 controls
- **Test**: 300 cases / 28,976 controls
- **Features**: HCT, HGB, MCH, MCHC, MCV, PLT, RBC, RDW, WBC

---

## Baseline (Checkpoint 5)

| method | features | AUC | precision | recall | notes |
|--------|----------|-----|-----------|--------|-------|
| Logistic regression (single) | RDW | 0.617 | 0.014 | 0.797 | Best single feature |
| Logistic regression (single) | RBC | 0.586 | 0.014 | 0.613 | |
| Logistic regression (single) | HCT | 0.579 | 0.014 | 0.657 | |
| Logistic regression (all) | All 9 | 0.658 | 0.016 | 0.710 | **Baseline to beat** |

---

## Method 1: Threshold Optimization

_Date: 2026-04-04_

### 1A: Literature-based threshold
| feature | threshold | direction | AUC | precision | recall | F1 | source |
|---------|-----------|-----------|-----|-----------|--------|----|--------|
| RDW | 14.5 | above | 0.617 | 0.015 | 0.487 | 0.029 | RDW >14.5% associated with inflammatory conditions |
| RBC | 4.0 | below | 0.586 | 0.013 | 0.703 | 0.025 | Low RBC in anemia of chronic disease |
| HCT | 36.0 | below | 0.579 | 0.012 | 0.690 | 0.024 | Low hematocrit in anemia of chronic disease |
| HGB | 12.0 | below | 0.569 | 0.013 | 0.703 | 0.025 | Anemia (Hgb <12 g/dL) common in chronic inflammation |
| PLT | 400.0 | above | 0.572 | 0.016 | 0.080 | 0.027 | Thrombocytosis (reactive) in chronic inflammation |
| MCV | 80.0 | below | 0.482 | 0.008 | 0.037 | 0.013 | Microcytosis in anemia of chronic disease |
| WBC | 11.0 | above | 0.479 | 0.009 | 0.233 | 0.017 | Leukocytosis in active inflammation |
| MCHC | 32.0 | below | 0.467 | 0.009 | 0.193 | 0.017 | Low MCHC in chronic disease |
| MCH | 27.0 | below | 0.463 | 0.008 | 0.087 | 0.015 | Low MCH in iron-deficiency/chronic disease anemia |

### 1B: Data-driven threshold (Youden's index)
| feature | threshold | direction | AUC | precision | recall |
|---------|-----------|-----------|-----|-----------|--------|
| | | | | | |

### Notes
_What did you learn? Did data-driven beat literature? By how much?_

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

| method | best AUC | best formula/feature | beats baseline? |
|--------|----------|---------------------|-----------------|
| Logistic regression (all features) | 0.658 | All 9 CBC | — (this IS the baseline) |
| Literature threshold (1A) | 0.617 | RDW >14.5 | No (vs 0.658) |
| Data-driven threshold | | | |
| Random formulas | | | |
| Genetic programming | | | |
| LLM formulas | | | |
