"""
Shared utilities for the RA biomarker pipeline.

Import from here instead of duplicating data loading, metric computation,
or path constants across method scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
DATA_PATH   = DATA_DIR / "ra_modeling_data.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
META_COLS    = {"subject_id", "is_case", "split"}
BASELINE_AUC = 0.658   # all-features logistic regression (AUC-ROC)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """Load the modeling CSV and return (df, feature_names)."""
    df       = pd.read_csv(path)
    features = [c for c in df.columns if c not in META_COLS]
    return df, features


def get_splits(df):
    """Return (train_df, test_df) using the pre-computed split column."""
    return df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_binary_metrics(y_true, y_pred):
    """Compute precision, recall, F1, F2 from binary predictions."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) > 0 else 0.0)
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "f2":        f2,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


def find_youden_threshold(y_true, scores):
    """
    Find optimal threshold via Youden's index on the ROC curve.
    Returns (threshold, fpr_array, tpr_array).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Youden's J = sensitivity + specificity - 1 = TPR - FPR
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx]), fpr, tpr


# ── Output helpers ────────────────────────────────────────────────────────────
def ensure_dir(path):
    """Create directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
