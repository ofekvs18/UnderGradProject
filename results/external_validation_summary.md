# External Validation Summary

## Cohort Details

| Cohort | Diseases | Prevalence |
|--------|----------|------------|
| MIMIC (internal test) | All 6 | RA 1.0%, Crohn 1.0%, T1D 1.1%, T2D 12.7%, Psoriasis 0.6%, Lupus 0.4% |
| NHANES (external) | RA, Psoriasis only | Unknown — data file not on disk |
| EHRSHOT (external) | All 6 | RA 11.1%, Crohn 3.7%, T1D 9.4%, T2D 57.5%, Psoriasis 6.5%, Lupus 14.5% |

> **EHRSHOT prevalences are elevated** by cohort construction: controls are assigned a random index date and dropped if they lack a CBC in the lookback window, which inflates case fraction relative to the true population rate. AUC-ROC is unaffected; AUC-PR and lift are prevalence-sensitive.

> **NHANES lift** cannot be computed — the processed `data/{disease}_nhanes_data.csv` files are not on disk. Raw AUC-PR values are reported instead.

---

## Results Table

### EHRSHOT — AUC-PR Lift (lift = AUC-PR / prevalence)

| Disease | Method | MIMIC lift | EHRSHOT lift | Delta |
|---------|--------|-----------|-------------|-------|
| RA | M1-Threshold | 1.38 | 0.79 | -0.59 |
| RA | M2-Random | 1.32 | 1.68 | +0.36 |
| RA | M3-GP | 1.59 | 2.46 | +0.86 |
| RA | M4-LLM | 1.45 | 0.99 | -0.46 |
| Crohn | M1-Threshold | 1.97 | 4.30 | +2.33 |
| Crohn | M2-Random | 1.41 | 3.42 | +2.01 |
| Crohn | M3-GP | 1.38 | 5.27 | +3.89 |
| Crohn | M4-LLM | 1.98 | 3.29 | +1.31 |
| T1D | M1-Threshold | 1.41 | 1.50 | +0.09 |
| T1D | M2-Random | 1.25 | 1.77 | +0.52 |
| T1D | M3-GP | 1.66 | 3.02 | +1.36 |
| T1D | M4-LLM | 1.22 | 2.64 | +1.43 |
| T2D | M1-Threshold | 1.19 | 1.10 | -0.09 |
| T2D | M2-Random | 1.18 | 1.03 | -0.15 |
| T2D | M3-GP | 1.26 | 1.13 | -0.13 |
| T2D | M4-LLM | 1.12 | 1.09 | -0.03 |
| Psoriasis | M1-Threshold | 2.42 | 1.33 | -1.09 |
| Psoriasis | M2-Random | 1.88 | 1.19 | -0.69 |
| Psoriasis | M3-GP | 1.54 | 1.33 | -0.21 |
| Psoriasis | M4-LLM | 2.48 | 1.48 | -1.00 |
| Lupus | M1-Threshold | 2.10 | 1.26 | -0.84 |
| Lupus | M2-Random | 1.78 | 1.71 | -0.08 |
| Lupus | M3-GP | 2.34 | 1.67 | -0.67 |
| Lupus | M4-LLM | 3.70 | 1.76 | -1.94 |

### NHANES — Raw AUC-PR (lift requires prevalence)

| Disease | Method | MIMIC AUC-PR | NHANES AUC-PR |
|---------|--------|-------------|--------------|
| RA | M1-Threshold | 0.0141 | 0.0816 |
| RA | M2-Random | 0.0135 | 0.0773 |
| RA | M3-GP | 0.0163 | 0.0790 |
| RA | M4-LLM | 0.0149 | 0.1016 |
| Psoriasis | M1-Threshold | 0.0134 | 0.0222 |
| Psoriasis | M2-Random | 0.0104 | 0.0334 |
| Psoriasis | M3-GP | 0.0085 | 0.0382 |
| Psoriasis | M4-LLM | 0.0137 | 0.0354 |

> NHANES AUC-PR values are substantially higher than MIMIC across all methods (e.g. RA M4-LLM: 0.1016 vs 0.0149). This likely reflects a higher NHANES cohort prevalence — self-reported RA and psoriasis via NHANES MCQ questionnaires typically runs 5–10% in the eligible adult population vs ~1% in MIMIC.

---

## Patterns and Observations

### EHRSHOT generalisation by disease

| Pattern | Diseases | Notes |
|---------|----------|-------|
| Strong generalisation (all methods gain lift) | Crohn (+1.3 to +3.9), T1D (+0.1 to +1.4) | Partly inflation from higher EHRSHOT prevalence |
| Roughly neutral | T2D (all within ±0.15) | Most stable cohort; T2D CBC signal is robust |
| Lift drops on external | Psoriasis (−0.2 to −1.1), Lupus (−0.1 to −1.9), RA M1/M4 (−0.5 to −0.6) | Suggests MIMIC-specific overfitting |

### Method ranking on EHRSHOT

- **M3-GP** is the best or tied-best on EHRSHOT for RA, Crohn, T1D, and T2D.
- **M4-LLM** shows the largest overfitting signal overall: Lupus drops 3.70× → 1.76× (−1.94), Psoriasis −1.00.
- **M1-Threshold** (single-feature rule) generalises poorly for Psoriasis and Lupus but is stable for T1D and T2D.
- **M2-Random** is consistently mid-table on both cohorts.

### Biggest overfitting signal

M4-LLM Lupus: MIMIC lift 3.70× → EHRSHOT lift 1.76× (delta −1.94). The formula `(mcv - mchc) / (mcv + mchc + 0.01) * rbc` achieves high lift on the MIMIC test set but does not generalise.

### Most robust formula

M3-GP RA: MIMIC 1.59× → EHRSHOT 2.46× (+0.86). The seeded GP formula generalises and actually improves on the independent EHRSHOT cohort.
