"""
Method 1 — Threshold Optimization (Literature + Data-Driven).

Part 1A: Apply known clinical thresholds to the test set and evaluate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

DATA_PATH = "data/ra_modeling_data.csv"
RESULTS_DIR = "results/method1_threshold"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows\n")

# ── Literature thresholds ─────────────────────────────────────────────────────
# Values from standard clinical references for inflammatory/autoimmune conditions.
literature_thresholds = {
    # feature: (threshold, direction, source)
    "rdw":  (14.5, "above", "RDW >14.5% associated with inflammatory conditions"),
    "hgb":  (12.0, "below", "Anemia (Hgb <12 g/dL) common in chronic inflammation"),
    "hct":  (36.0, "below", "Low hematocrit in anemia of chronic disease"),
    "wbc":  (11.0, "above", "Leukocytosis in active inflammation"),
    "plt":  (400.0, "above", "Thrombocytosis (reactive) in chronic inflammation"),
    "mcv":  (80.0, "below", "Microcytosis in anemia of chronic disease"),
    "mch":  (27.0, "below", "Low MCH in iron-deficiency/chronic disease anemia"),
    "mchc": (32.0, "below", "Low MCHC in chronic disease"),
    "rbc":  (4.0,  "below", "Low RBC in anemia of chronic disease"),
}

# ── Part 1A: Literature-based evaluation ─────────────────────────────────────
print("=== Part 1A: Literature-based thresholds ===")

lit_results = []

for feat, (threshold, direction, source) in literature_thresholds.items():
    te = test_df[[feat, "is_case"]].dropna()
    y_test = te["is_case"].values
    x_test = te[feat].values

    if y_test.sum() < 5:
        print(f"  {feat:6s}: skipped (too few positives)")
        continue

    # AUC uses raw feature value; flip sign for "below" so higher = more likely RA
    score = x_test if direction == "above" else -x_test
    auc = roc_auc_score(y_test, score)

    # Binary prediction at the literature threshold
    if direction == "above":
        preds = (x_test > threshold).astype(int)
    else:
        preds = (x_test < threshold).astype(int)

    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    lit_results.append({
        "feature":    feat,
        "threshold":  threshold,
        "direction":  direction,
        "auc":        round(auc, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "n_test":     len(te),
        "source":     source,
    })

    print(f"  {feat:6s} ({direction:5s} {threshold:>6}): "
          f"AUC={auc:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

# ── Save literature results ───────────────────────────────────────────────────
lit_df = pd.DataFrame(lit_results)
lit_df.to_csv(f"{RESULTS_DIR}/literature_results.csv", index=False)
print(f"\nSaved {RESULTS_DIR}/literature_results.csv")

best_lit = lit_df.loc[lit_df["auc"].idxmax()]
print(f"\nBest literature AUC: {best_lit['feature']} — AUC={best_lit['auc']:.4f}")
