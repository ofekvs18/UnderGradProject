"""
Issue #3 — Add imbalanced-data metrics to all evaluations.

Adds AUC-PR, precision@recall (0.25, 0.50, 0.75), F1, and F-beta(2) to:
  - Checkpoint 5 logistic regression results  -> results/sanity_check_results_v2.csv
  - Method 1 threshold results                -> results/method1_threshold/comparison_table_v2.csv
  - PR curves for all features                -> results/method1_threshold/pr_curves.png
  - Plain text comparison summary             -> results/evaluation_summary.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)

DATA_PATH   = "data/ra_modeling_data.csv"
RESULTS_DIR = "results"
M1_DIR      = "results/method1_threshold"

os.makedirs(M1_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df       = pd.read_csv(DATA_PATH)
META     = {"subject_id", "is_case", "split"}
features = [c for c in df.columns if c not in META]

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()
print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fbeta(precision, recall, beta):
    """F-beta score from scalar precision and recall."""
    b2 = beta ** 2
    if (b2 * precision + recall) == 0:
        return 0.0
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def precision_at_recall_levels(y_train_scores, y_train, y_test_scores, y_test,
                                levels=(0.25, 0.50, 0.75)):
    """
    For each target recall level, find the score threshold on TRAIN that achieves
    recall >= level, then apply it to TEST and return (precision, recall) pairs.

    Uses the PR curve on train to pick the threshold.
    """
    prec_tr, rec_tr, thresh_tr = precision_recall_curve(y_train, y_train_scores)
    # precision_recall_curve returns arrays in descending recall order;
    # thresh_tr has one fewer element than prec_tr/rec_tr.
    # We search over thresh_tr (indices 0..n-2), matching rec_tr[:-1].
    rec_search  = rec_tr[:-1]
    prec_search = prec_tr[:-1]
    thresholds  = thresh_tr

    results = {}
    for level in levels:
        # Find thresholds where train recall >= target level; pick highest threshold
        # (most conservative / highest precision among qualifying points).
        mask = rec_search >= level
        if not mask.any():
            results[level] = (0.0, 0.0)
            continue
        best_thresh = thresholds[mask].max()

        # Apply to test
        preds  = (y_test_scores >= best_thresh).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[level] = (round(p, 4), round(r, 4))
    return results


def threshold_metrics(y_true, preds):
    """Compute precision, recall, F1, F2 from binary predictions."""
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(p, 4), round(r, 4), round(fbeta(p, r, 1), 4), round(fbeta(p, r, 2), 4)


# ── Part A: Checkpoint 5 — logistic regression with new metrics ───────────────
print("=== Checkpoint 5: logistic regression + new metrics ===")

def fit_lr_full(X_tr, y_tr, X_te, y_te, label):
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    proba_tr = clf.predict_proba(X_tr)[:, 1]
    proba_te = clf.predict_proba(X_te)[:, 1]

    auc_roc = roc_auc_score(y_te, proba_te)
    auc_pr  = average_precision_score(y_te, proba_te)

    # Youden threshold for operating-point metrics
    fpr, tpr, thresholds = roc_curve(y_te, proba_te)
    best_idx  = int(np.argmax(tpr - fpr))
    threshold = thresholds[best_idx]
    preds     = (proba_te >= threshold).astype(int)
    p, r, f1, f2 = threshold_metrics(y_te, preds)

    par = precision_at_recall_levels(proba_tr, y_tr, proba_te, y_te)

    return {
        "model":                    label,
        "n_train":                  len(X_tr),
        "n_test":                   len(X_te),
        "auc_roc":                  round(auc_roc, 4),
        "auc_pr":                   round(auc_pr, 4),
        "threshold":                round(float(threshold), 4),
        "precision":                p,
        "recall":                   r,
        "f1":                       f1,
        "f2":                       f2,
        "precision_at_recall_25":   par[0.25][0],
        "precision_at_recall_50":   par[0.50][0],
        "precision_at_recall_75":   par[0.75][0],
    }

sanity_rows = []

for feat in features:
    tr = train_df[[feat, "is_case"]].dropna()
    te = test_df[[feat, "is_case"]].dropna()
    if tr["is_case"].sum() < 5 or te["is_case"].sum() < 5:
        continue
    row = fit_lr_full(tr[[feat]].values, tr["is_case"].values,
                      te[[feat]].values, te["is_case"].values, label=feat)
    sanity_rows.append(row)
    print(f"  {feat:8s}: AUC-ROC={row['auc_roc']:.4f}  AUC-PR={row['auc_pr']:.4f}  "
          f"P@R50={row['precision_at_recall_50']:.4f}  F1={row['f1']:.4f}  F2={row['f2']:.4f}")

# All-features
tr_all = train_df[features + ["is_case"]].dropna()
te_all = test_df[features + ["is_case"]].dropna()
row_all = fit_lr_full(tr_all[features].values, tr_all["is_case"].values,
                      te_all[features].values, te_all["is_case"].values, label="all_features")
sanity_rows.append(row_all)
print(f"  {'all_features':8s}: AUC-ROC={row_all['auc_roc']:.4f}  AUC-PR={row_all['auc_pr']:.4f}  "
      f"P@R50={row_all['precision_at_recall_50']:.4f}  F1={row_all['f1']:.4f}  F2={row_all['f2']:.4f}")

sanity_v2 = pd.DataFrame(sanity_rows)
sanity_v2.to_csv(f"{RESULTS_DIR}/sanity_check_results_v2.csv", index=False)
print(f"\nSaved {RESULTS_DIR}/sanity_check_results_v2.csv")

# ── Part B: Method 1 — re-apply thresholds with new metrics ──────────────────
print("\n=== Method 1: threshold results + new metrics ===")

literature_thresholds = {
    "rdw":  (14.5, "above"),
    "hgb":  (12.0, "below"),
    "hct":  (36.0, "below"),
    "wbc":  (11.0, "above"),
    "plt":  (400.0, "above"),
    "mcv":  (80.0, "below"),
    "mch":  (27.0, "below"),
    "mchc": (32.0, "below"),
    "rbc":  (4.0,  "below"),
}

m1_rows = []

for feat, (lit_thresh, lit_dir) in literature_thresholds.items():
    tr = train_df[[feat, "is_case"]].dropna()
    te = test_df[[feat, "is_case"]].dropna()
    if tr["is_case"].sum() < 5 or te["is_case"].sum() < 5:
        continue

    y_tr, x_tr = tr["is_case"].values, tr[feat].values
    y_te, x_te = te["is_case"].values, te[feat].values

    # Score: flip for "below" so higher = more likely RA
    score_tr = x_tr if lit_dir == "above" else -x_tr
    score_te = x_te if lit_dir == "above" else -x_te

    auc_roc = roc_auc_score(y_te, score_te)
    auc_pr  = average_precision_score(y_te, score_te)

    # Literature threshold operating point
    lit_preds = (x_te > lit_thresh).astype(int) if lit_dir == "above" else (x_te < lit_thresh).astype(int)
    lit_p, lit_r, lit_f1, lit_f2 = threshold_metrics(y_te, lit_preds)

    # Youden threshold on train
    fpr, tpr, thresholds = roc_curve(y_tr, score_tr)
    best_idx    = int(np.argmax(tpr - fpr))
    dd_thresh   = float(thresholds[best_idx])
    # Convert back to original feature space
    dd_thresh_orig = dd_thresh if lit_dir == "above" else -dd_thresh
    dd_dir         = lit_dir   # direction is the same; threshold value differs

    dd_preds = (x_te >= dd_thresh_orig).astype(int) if dd_dir == "above" else (x_te <= dd_thresh_orig).astype(int)
    dd_p, dd_r, dd_f1, dd_f2 = threshold_metrics(y_te, dd_preds)

    # Precision at fixed recall levels (using train scores)
    par = precision_at_recall_levels(score_tr, y_tr, score_te, y_te)

    m1_rows.append({
        "feature":                  feat,
        "auc_roc":                  round(auc_roc, 4),
        "auc_pr":                   round(auc_pr, 4),
        "literature_threshold":     lit_thresh,
        "literature_direction":     lit_dir,
        "literature_precision":     lit_p,
        "literature_recall":        lit_r,
        "literature_f1":            lit_f1,
        "literature_f2":            lit_f2,
        "datadriven_threshold":     round(dd_thresh_orig, 4),
        "datadriven_direction":     dd_dir,
        "datadriven_precision":     dd_p,
        "datadriven_recall":        dd_r,
        "datadriven_f1":            dd_f1,
        "datadriven_f2":            dd_f2,
        "precision_at_recall_25":   par[0.25][0],
        "precision_at_recall_50":   par[0.50][0],
        "precision_at_recall_75":   par[0.75][0],
    })

    print(f"  {feat:6s}: AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
          f"P@R50={par[0.50][0]:.4f}  lit_F2={lit_f2:.4f}  dd_F2={dd_f2:.4f}")

m1_v2 = pd.DataFrame(m1_rows).sort_values("auc_pr", ascending=False).reset_index(drop=True)
m1_v2.to_csv(f"{M1_DIR}/comparison_table_v2.csv", index=False)
print(f"\nSaved {M1_DIR}/comparison_table_v2.csv")

# ── Part C: PR curves for all features ───────────────────────────────────────
print("\nGenerating PR curves...")

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
axes = axes.flatten()
feat_list = [r["feature"] for r in m1_rows]

for i, feat in enumerate(feat_list):
    ax  = axes[i]
    row = next(r for r in m1_rows if r["feature"] == feat)

    te   = test_df[[feat, "is_case"]].dropna()
    y_te = te["is_case"].values
    x_te = te[feat].values

    direction  = row["literature_direction"]
    score_te   = x_te if direction == "above" else -x_te
    prevalence = y_te.mean()

    prec_c, rec_c, _ = precision_recall_curve(y_te, score_te)
    ap = row["auc_pr"]

    ax.plot(rec_c, prec_c, color="#4C72B0", lw=1.5, label=f"AUC-PR={ap:.4f}")
    ax.axhline(prevalence, color="gray", linestyle=":", lw=1, label=f"Baseline ({prevalence:.3f})")

    # Mark literature operating point
    ax.scatter([row["literature_recall"]], [row["literature_precision"]],
               s=80, color="green", zorder=5, label=f"Lit ({row['literature_threshold']})")
    # Mark data-driven operating point
    ax.scatter([row["datadriven_recall"]], [row["datadriven_precision"]],
               s=80, color="orange", marker="^", zorder=5, label=f"DD ({row['datadriven_threshold']})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(prec_c.max() * 1.15, prevalence * 3))
    ax.set_title(feat.upper(), fontsize=10)
    ax.set_xlabel("Recall", fontsize=8)
    ax.set_ylabel("Precision", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

fig.suptitle("Method 1: Precision-Recall Curves (all features)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{M1_DIR}/pr_curves.png", dpi=150)
plt.close()
print(f"Saved {M1_DIR}/pr_curves.png")

# ── Part D: Evaluation summary ────────────────────────────────────────────────
print("\nWriting evaluation summary...")

sanity_single = sanity_v2[sanity_v2["model"] != "all_features"].copy()
roc_ranking   = sanity_single.sort_values("auc_roc", ascending=False)["model"].tolist()
pr_ranking    = sanity_single.sort_values("auc_pr",  ascending=False)["model"].tolist()
rankings_match = roc_ranking == pr_ranking

m1_roc_rank = m1_v2.sort_values("auc_roc", ascending=False)["feature"].tolist()
m1_pr_rank  = m1_v2.sort_values("auc_pr",  ascending=False)["feature"].tolist()
m1_rankings_match = m1_roc_rank == m1_pr_rank

prevalence = test_df["is_case"].mean()

lines = [
    "Evaluation Summary — AUC-ROC vs AUC-PR",
    "=" * 50,
    f"",
    f"Class imbalance: {prevalence:.4f} positive rate in test set",
    f"  -> Random classifier AUC-PR baseline: ~{prevalence:.4f}",
    f"",
    "── Checkpoint 5: Logistic Regression ──",
    f"{'Model':<14} {'AUC-ROC':>8} {'AUC-PR':>8} {'P@R25':>7} {'P@R50':>7} {'P@R75':>7} {'F1':>6} {'F2':>6}",
    "-" * 70,
]
for _, row in sanity_v2.sort_values("auc_pr", ascending=False).iterrows():
    lines.append(
        f"{row['model']:<14} {row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
        f"{row['precision_at_recall_25']:>7.4f} {row['precision_at_recall_50']:>7.4f} "
        f"{row['precision_at_recall_75']:>7.4f} {row['f1']:>6.4f} {row['f2']:>6.4f}"
    )

lines += [
    "",
    f"AUC-ROC ranking: {' > '.join(roc_ranking)}",
    f"AUC-PR  ranking: {' > '.join(pr_ranking)}",
    f"Rankings identical: {rankings_match}",
    "",
    "── Method 1: Single-Feature Thresholds ──",
    f"{'Feature':<8} {'AUC-ROC':>8} {'AUC-PR':>8} {'P@R25':>7} {'P@R50':>7} {'P@R75':>7}",
    "-" * 50,
]
for _, row in m1_v2.sort_values("auc_pr", ascending=False).iterrows():
    lines.append(
        f"{row['feature']:<8} {row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
        f"{row['precision_at_recall_25']:>7.4f} {row['precision_at_recall_50']:>7.4f} "
        f"{row['precision_at_recall_75']:>7.4f}"
    )

lines += [
    "",
    f"AUC-ROC ranking: {' > '.join(m1_roc_rank)}",
    f"AUC-PR  ranking: {' > '.join(m1_pr_rank)}",
    f"Rankings identical: {m1_rankings_match}",
    "",
    "── Key Findings ──",
]

best_roc = sanity_single.loc[sanity_single["auc_roc"].idxmax(), "model"]
best_pr  = sanity_single.loc[sanity_single["auc_pr"].idxmax(),  "model"]
if best_roc == best_pr:
    lines.append(f"  * Best feature is {best_roc.upper()} by both AUC-ROC and AUC-PR.")
    lines.append(f"    Class imbalance does not change feature selection.")
else:
    lines.append(f"  * Best feature by AUC-ROC: {best_roc.upper()}")
    lines.append(f"  * Best feature by AUC-PR:  {best_pr.upper()}")
    lines.append(f"    Rankings diverge — AUC-PR should be preferred for model selection.")

if not rankings_match:
    lines.append(f"  * Full feature rankings differ between AUC-ROC and AUC-PR.")
    lines.append(f"    Use AUC-PR as primary metric for future methods.")
else:
    lines.append(f"  * Full feature rankings are stable under both metrics.")

all_row = sanity_v2[sanity_v2["model"] == "all_features"].iloc[0]
lines += [
    "",
    f"All-features LR: AUC-ROC={all_row['auc_roc']:.4f}  AUC-PR={all_row['auc_pr']:.4f}",
    f"  -> This is the baseline all future methods must beat on BOTH metrics.",
]

summary_text = "\n".join(lines) + "\n"
with open(f"{RESULTS_DIR}/evaluation_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)
print(f"Saved {RESULTS_DIR}/evaluation_summary.txt")
print("\n" + summary_text.encode("ascii", errors="replace").decode("ascii"))
