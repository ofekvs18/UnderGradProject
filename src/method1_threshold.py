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

# ── Part 1B: Data-driven thresholds (Youden's index) ─────────────────────────
print("\n=== Part 1B: Data-driven thresholds (Youden's index) ===")

features = list(literature_thresholds.keys())
dd_results = []

for feat in features:
    # Use train set to find the optimal threshold
    tr = train_df[[feat, "is_case"]].dropna()
    te = test_df[[feat, "is_case"]].dropna()

    if tr["is_case"].sum() < 5 or te["is_case"].sum() < 5:
        print(f"  {feat:6s}: skipped (too few positives)")
        continue

    y_train = tr["is_case"].values
    x_train = tr[feat].values
    y_test  = te["is_case"].values
    x_test  = te[feat].values

    # Compute ROC on train to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_train, x_train)

    # Youden's index: J = sensitivity + specificity - 1 = TPR + (1 - FPR) - 1 = TPR - FPR
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[best_idx]

    # Determine direction from the ROC: if higher feature value -> higher TPR,
    # the ROC is computed as-is; we predict RA if feature >= threshold.
    # Check which direction the AUC favours on the train set.
    auc_train_above = roc_auc_score(y_train, x_train)
    if auc_train_above >= 0.5:
        direction = "above"
        preds = (x_test >= optimal_threshold).astype(int)
        score  = x_test
    else:
        direction = "below"
        preds = (x_test <= optimal_threshold).astype(int)
        score  = -x_test

    auc = roc_auc_score(y_test, score)

    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    dd_results.append({
        "feature":           feat,
        "optimal_threshold": round(float(optimal_threshold), 4),
        "direction":         direction,
        "auc":               round(auc, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1":                round(f1, 4),
        "n_train":           len(tr),
        "n_test":            len(te),
    })

    print(f"  {feat:6s} ({direction:5s} {optimal_threshold:>8.4f}): "
          f"AUC={auc:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

# ── Save data-driven results ──────────────────────────────────────────────────
dd_df = pd.DataFrame(dd_results)
dd_df.to_csv(f"{RESULTS_DIR}/datadriven_results.csv", index=False)
print(f"\nSaved {RESULTS_DIR}/datadriven_results.csv")

best_dd = dd_df.loc[dd_df["auc"].idxmax()]
print(f"\nBest data-driven AUC: {best_dd['feature']} — AUC={best_dd['auc']:.4f}  "
      f"threshold={best_dd['optimal_threshold']} ({best_dd['direction']})")

# ── Part 1C: Comparison table and visualizations ──────────────────────────────
print("\n=== Part 1C: Comparison ===")

# Merge literature and data-driven results on feature
comp = lit_df[["feature", "threshold", "direction", "auc", "precision", "recall"]].rename(columns={
    "threshold": "literature_threshold",
    "direction": "literature_direction",
    "auc":       "auc",
    "precision": "literature_precision",
    "recall":    "literature_recall",
})
comp = comp.merge(
    dd_df[["feature", "optimal_threshold", "direction", "precision", "recall"]].rename(columns={
        "optimal_threshold": "datadriven_threshold",
        "direction":         "datadriven_direction",
        "precision":         "datadriven_precision",
        "recall":            "datadriven_recall",
    }),
    on="feature",
)
comp = comp[["feature", "literature_threshold", "literature_precision", "literature_recall",
             "datadriven_threshold", "datadriven_precision", "datadriven_recall", "auc"]]
comp = comp.sort_values("auc", ascending=False).reset_index(drop=True)

comp.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False)
print(f"Saved {RESULTS_DIR}/comparison_table.csv")
print(comp.to_string(index=False))

# ── Bar chart: AUC by feature, grouped literature vs data-driven ──────────────
# AUC is threshold-independent, so both groups share the same AUC values.
# The chart shows AUC per feature with two bars to make that explicit.
features_ordered = comp["feature"].tolist()
x = np.arange(len(features_ordered))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars_lit = ax.bar(x - width / 2, comp["auc"], width, label="Literature threshold", color="#4C72B0")
bars_dd  = ax.bar(x + width / 2, comp["auc"], width, label="Data-driven threshold", color="#DD8452", alpha=0.8)

ax.axhline(0.658, color="red", linestyle="--", linewidth=1.2, label="Baseline (all-features LR, AUC=0.658)")
ax.set_xticks(x)
ax.set_xticklabels([f.upper() for f in features_ordered])
ax.set_ylabel("AUC (test set)")
ax.set_title("Method 1: AUC by Feature — Literature vs Data-Driven Threshold")
ax.set_ylim(0.4, 0.72)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/comparison_chart.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/comparison_chart.png")

# ── ROC curve for best feature with both thresholds marked ───────────────────
best_feat = comp.loc[0, "feature"]
lit_thresh  = float(comp.loc[0, "literature_threshold"])
lit_dir     = lit_df.loc[lit_df["feature"] == best_feat, "direction"].values[0]
dd_thresh   = float(comp.loc[0, "datadriven_threshold"])
dd_dir      = dd_df.loc[dd_df["feature"] == best_feat, "direction"].values[0]

te_best = test_df[[best_feat, "is_case"]].dropna()
y_best  = te_best["is_case"].values
x_best  = te_best[best_feat].values

# Score: flip sign for "below" features so higher score = more likely RA
score_best = x_best if lit_dir == "above" else -x_best
fpr_curve, tpr_curve, thresh_curve = roc_curve(y_best, score_best)

# Find ROC operating points for each threshold
def _find_roc_point(x_vals, y_vals, threshold, direction):
    """Return (fpr, tpr) closest to the given threshold on the ROC curve."""
    preds = (x_vals >= threshold).astype(int) if direction == "above" else (x_vals <= threshold).astype(int)
    tp = int(((preds == 1) & (y_vals == 1)).sum())
    fp = int(((preds == 1) & (y_vals == 0)).sum())
    fn = int(((preds == 0) & (y_vals == 1)).sum())
    tn = int(((preds == 0) & (y_vals == 0)).sum())
    tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr_pt, tpr_pt

fpr_lit, tpr_lit = _find_roc_point(x_best, y_best, lit_thresh, lit_dir)
fpr_dd,  tpr_dd  = _find_roc_point(x_best, y_best, dd_thresh,  dd_dir)

auc_best = roc_auc_score(y_best, score_best)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr_curve, tpr_curve, color="#4C72B0", lw=2, label=f"ROC (AUC={auc_best:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.scatter([fpr_lit], [tpr_lit], s=120, zorder=5, color="green",
           label=f"Literature threshold ({lit_thresh}, {lit_dir})\nTPR={tpr_lit:.3f}  FPR={fpr_lit:.3f}")
ax.scatter([fpr_dd], [tpr_dd], s=120, zorder=5, color="orange", marker="^",
           label=f"Data-driven threshold ({dd_thresh}, {dd_dir})\nTPR={tpr_dd:.3f}  FPR={fpr_dd:.3f}")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"Method 1: ROC curve — {best_feat.upper()} (best feature)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/best_feature_roc.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/best_feature_roc.png")

# ── Summary text ──────────────────────────────────────────────────────────────
best_lit_row = lit_df.loc[lit_df["feature"] == best_feat].iloc[0]
best_dd_row  = dd_df.loc[dd_df["feature"]  == best_feat].iloc[0]
auc_improvement = best_dd_row["auc"] - best_lit_row["auc"]
auc_vs_baseline = best_dd_row["auc"] - 0.658

insight = (
    "Data-driven threshold does not improve AUC over literature (AUC is threshold-independent). "
    "The key difference is recall: Youden-optimized threshold trades precision for higher sensitivity."
    if abs(auc_improvement) < 0.001
    else f"Data-driven threshold improves AUC by {auc_improvement:+.4f} over literature."
)

summary = f"""Method 1: Threshold Optimization Results
=========================================

Best single feature: {best_feat.upper()}
  Literature threshold: {best_lit_row['threshold']} ({best_lit_row['direction']}) -> AUC={best_lit_row['auc']:.3f}, Precision={best_lit_row['precision']:.3f}, Recall={best_lit_row['recall']:.3f}
  Data-driven threshold: {best_dd_row['optimal_threshold']} ({best_dd_row['direction']}) -> AUC={best_dd_row['auc']:.3f}, Precision={best_dd_row['precision']:.3f}, Recall={best_dd_row['recall']:.3f}
  Improvement from data-driven: {auc_improvement:+.4f} AUC

Comparison to baseline:
  All-features logistic regression AUC: 0.658
  Best threshold AUC: {best_dd_row['auc']:.3f}
  Difference: {auc_vs_baseline:+.3f}

Key insight: {insight}
"""

with open(f"{RESULTS_DIR}/method1_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print(f"Saved {RESULTS_DIR}/method1_summary.txt")
print("\n" + summary)
