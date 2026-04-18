# Coding Standards

**Project:** Biomarker Pipeline Research | **Student:** Ofek

---

## File Organization

**ONE FILE PER METHOD** — Each method = one Python script (exception: shared `utils.py`)

```
src/
├── utils.py                  # Shared utilities only
├── method1_threshold.py      # One complete method
├── method2_random_formula.py # One complete method
└── method3_gp.py            # One complete method
```

---

## Code Style

### Imports
All imports at top, three blocks (standard lib, third-party, local):

```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# Local
from utils import load_data, compute_metrics
```

### Printing
**Minimize prints.** Allowed: progress indicators, final results, critical errors.

```python
# ❌ BAD: verbose iteration
for i, formula in enumerate(formulas):
    print(f"Formula {i}: {formula} → {result}")

# ✅ GOOD: summary only
print(f"Evaluated {len(formulas)} formulas. Best AUC-PR: {max_auc:.4f}")
```

### Docstrings
Every function needs one:

```python
def evaluate_formula(formula_str: str, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate mathematical formula on test data.
    
    Args:
        formula_str: Python-evaluable formula
        X: Feature dataframe
        y: Binary labels
    
    Returns:
        Dict with 'auc_roc' and 'auc_pr'
    """
```

### Naming
- `snake_case`: variables, functions
- `PascalCase`: classes
- `UPPERCASE`: constants
- Descriptive: `train_auc` not `ta`

### Random Seeds
Always set for reproducibility:

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
model = LogisticRegression(random_state=RANDOM_SEED)
```

---

## Data Rules

### IMMUTABLE DATASET
Once `ra_modeling_data.csv` exists with train/test split → **NEVER** regenerate or modify.  
Rationale: All methods must use identical data.

### Feature Names
Standard CBC features (lowercase):
```python
CBC_FEATURES = ['hct', 'hgb', 'mch', 'mchc', 'mcv', 'plt', 'rbc', 'rdw', 'wbc']
```

### Data Leakage Prevention
**NEVER** include CBC from:
- Beyond first 24h of index admission
- After diagnosis timestamp

---

## Metrics

**Primary:** AUC-PR (class imbalance ~1%)  
**Secondary:** AUC-ROC

**Baseline (RA):** All-features logistic regression (AUC-PR ~0.017, AUC-ROC ~0.658)

**Reporting:**
- Code: 4 decimals (`0.0179`)
- Tables: 3 decimals (`0.018`)

**Success criteria:** Beat baseline on **both** metrics, or have clear reason why not.

---

## Experiment Tracking

Every run gets logged in `results/experiment_log.md`:
- Date, method, config, dataset, results (AUC-ROC + AUC-PR), observations

Each method saves to `results/methodX_*/`:
- `predictions.csv`
- `best_formula.txt`
- `metrics.json`
- SLURM logs (cluster jobs)

---

## Git

**Commit format:** `<type>: <description>`  
Types: `feat`, `fix`, `docs`, `refactor`, `data`, `exp`

```
feat: implement GP method
exp: GP large config (AUC-PR 0.0179)
```

**Commit:** code, docs, small results (<1MB)  
**Don't commit:** datasets, temp files (`.pyc`, `__pycache__`)

---

## Checklist Before Push

- [ ] Imports at top, organized
- [ ] No debug prints
- [ ] Random seed set
- [ ] Docstrings present
- [ ] Results saved to `results/`
- [ ] Logged in `experiment_log.md`
- [ ] One-file-per-method rule followed
- [ ] Both AUC-ROC and AUC-PR reported

---

**Non-negotiable:** Immutable dataset, train/test integrity, temporal guard logic
