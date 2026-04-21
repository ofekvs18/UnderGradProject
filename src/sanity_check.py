"""
Sanity check — logistic regression (single-feature and all-features).

Purpose: verify the data pipeline is free of leakage.
Expected AUC range if clean: ~0.55–0.75 for single features.
AUC > 0.95 on a single feature strongly suggests label or temporal leakage.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from utils import (
    load_data_for, load_disease_config, get_splits, compute_binary_metrics, find_youden_threshold,
    precision_at_recall_levels, ensure_dir, RESULTS_DIR,
)

parser = argparse.ArgumentParser(description="Sanity check — logistic regression leakage test")
parser.add_argument("--disease", default="ra", help="Disease slug (e.g. ra, dm1)")
args = parser.parse_args()

disease = load_disease_config(args.disease)

ensure_dir(RESULTS_DIR)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df, features = load_data_for(disease.name)
train_df, test_df = get_splits(df)

print(f"Columns: {list(df.columns)}")
print(f"Features: {features}\n")

# ── Dataset summary ───────────────────────────────────────────────────────────
print("=== Dataset summary ===")
for split_name in ["train", "test"]:
    sub = df[df["split"] == split_name]
    n_cases    = sub["is_case"].sum()
    n_controls = len(sub) - n_cases
    print(f"  {split_name}: {len(sub):,} rows  |  cases={n_cases:,}  controls={n_controls:,}")

print("\nMissing value counts per feature:")
for feat in features:
    n_miss = df[feat].isna().sum()
    pct    = 100 * n_miss / len(df)
    print(f"  {feat:8s}: {n_miss:,} missing ({pct:.1f}%)")
print()

# ── Helper: fit + evaluate ────────────────────────────────────────────────────
def fit_evaluate(X_train, y_train, X_test, y_test, label):
    """
    Fit logistic regression and return a dict of metrics.
    Uses Youden's index to pick an operating threshold.
    No hyperparameter tuning — just default solver with balanced class weight
    to handle the severe class imbalance (~1:100).

    Returns:
        Dict with AUC-ROC, AUC-PR, threshold-based metrics (precision, recall, F1, F2),
        and precision@recall levels (0.25, 0.50, 0.75).
    """
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    proba_train = clf.predict_proba(X_train)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, proba_test)
    auc_pr = average_precision_score(y_test, proba_test)

    threshold, fpr, tpr = find_youden_threshold(y_test, proba_test)
    preds = (proba_test >= threshold).astype(int)
    m = compute_binary_metrics(y_test, preds)
    par = precision_at_recall_levels(proba_train, y_train, proba_test, y_test)

    return {
        "model":                    label,
        "n_train":                  len(X_train),
        "n_test":                   len(X_test),
        "auc_roc":                  round(auc_roc, 4),
        "auc_pr":                   round(auc_pr, 4),
        "threshold":                round(threshold, 4),
        "precision":                round(m["precision"], 4),
        "recall":                   round(m["recall"], 4),
        "f1":                       round(m["f1"], 4),
        "f2":                       round(m["f2"], 4),
        "precision_at_recall_25":   par[0.25][0],
        "precision_at_recall_50":   par[0.50][0],
        "precision_at_recall_75":   par[0.75][0],
        "fpr_arr":                  fpr,
        "tpr_arr":                  tpr,
    }

# ── Single-feature models ─────────────────────────────────────────────────────
print("=== Single-feature logistic regression ===")
results = []

for feat in features:
    tr = train_df[[feat, "is_case"]].dropna()
    te = test_df[[feat, "is_case"]].dropna()

    if tr["is_case"].sum() < 5 or te["is_case"].sum() < 5:
        print(f"  {feat:8s}: skipped (too few positive examples after dropping nulls)")
        continue

    res = fit_evaluate(tr[[feat]].values, tr["is_case"].values,
                       te[[feat]].values, te["is_case"].values, label=feat)
    results.append(res)
    print(f"  {feat:8s}: AUC-ROC={res['auc_roc']:.4f}  AUC-PR={res['auc_pr']:.4f}  "
          f"P@R50={res['precision_at_recall_50']:.4f}  F1={res['f1']:.4f}  F2={res['f2']:.4f}")

# ── All-features model ────────────────────────────────────────────────────────
print("\n=== All-features logistic regression ===")
tr_all = train_df[features + ["is_case"]].dropna()
te_all = test_df[features + ["is_case"]].dropna()

res_all = fit_evaluate(tr_all[features].values, tr_all["is_case"].values,
                       te_all[features].values, te_all["is_case"].values, label="all_features")
results.append(res_all)
print(f"  all_features: AUC-ROC={res_all['auc_roc']:.4f}  AUC-PR={res_all['auc_pr']:.4f}  "
      f"P@R50={res_all['precision_at_recall_50']:.4f}  F1={res_all['f1']:.4f}  F2={res_all['f2']:.4f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
single_aucs_roc  = [r["auc_roc"] for r in results if r["model"] != "all_features"]
single_aucs_pr   = [r["auc_pr"] for r in results if r["model"] != "all_features"]
best_single_roc  = max(single_aucs_roc) if single_aucs_roc else 0.0
best_single_pr   = max(single_aucs_pr) if single_aucs_pr else 0.0
best_feat_roc    = next(r["model"] for r in results if r["auc_roc"] == best_single_roc and r["model"] != "all_features")
best_feat_pr     = next(r["model"] for r in results if r["auc_pr"] == best_single_pr and r["model"] != "all_features")
all_feat_roc     = res_all["auc_roc"]
all_feat_pr      = res_all["auc_pr"]

if best_single_roc > 0.95:
    verdict = "FAIL — likely data leakage (single-feature AUC-ROC > 0.95)"
elif best_single_roc >= 0.85:
    verdict = "WARNING — single-feature AUC-ROC 0.85–0.95, investigate potential leakage"
elif all_feat_roc >= 0.95:
    verdict = "FAIL — likely data leakage (all-features AUC-ROC >= 0.95)"
else:
    verdict = "PASS — AUC in expected range, pipeline appears clean"

print(f"\n{'='*70}")
print(f"Best single-feature AUC-ROC : {best_single_roc:.4f}  ({best_feat_roc})")
print(f"Best single-feature AUC-PR  : {best_single_pr:.4f}  ({best_feat_pr})")
print(f"All-features AUC-ROC        : {all_feat_roc:.4f}")
print(f"All-features AUC-PR         : {all_feat_pr:.4f}")
print(f"Verdict                     : {verdict}")
print(f"{'='*70}\n")

# ── Save CSV results ──────────────────────────────────────────────────────────
csv_rows   = [{k: v for k, v in r.items() if k not in ("fpr_arr", "tpr_arr")} for r in results]
results_df = pd.DataFrame(csv_rows)
results_df.to_csv(RESULTS_DIR / "sanity_check_results.csv", index=False)
print(f"Saved {RESULTS_DIR}/sanity_check_results.csv")

# ── ROC curves plot ───────────────────────────────────────────────────────────
single_results = [r for r in results if r["model"] != "all_features"]
top3           = sorted(single_results, key=lambda r: r["auc_roc"], reverse=True)[:3]
plot_models    = top3 + [res_all]

fig, ax = plt.subplots(figsize=(7, 6))
for r in plot_models:
    ax.plot(r["fpr_arr"], r["tpr_arr"], label=f"{r['model']} (AUC={r['auc_roc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Sanity Check")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/roc_curves.png")

# ── Plain text summary ────────────────────────────────────────────────────────
prevalence = test_df["is_case"].mean()
rankings_match = best_feat_roc == best_feat_pr

summary_lines = [
    "Sanity Check Summary — AUC-ROC vs AUC-PR",
    "=" * 70,
    "",
    f"Class imbalance: {prevalence:.4f} positive rate in test set",
    f"  -> Random classifier AUC-PR baseline: ~{prevalence:.4f}",
    "",
    f"{'Model':<14} {'AUC-ROC':>8} {'AUC-PR':>8} {'P@R25':>7} {'P@R50':>7} {'P@R75':>7} {'F1':>6} {'F2':>6}",
    "-" * 70,
]
for r in sorted(single_results, key=lambda r: r["auc_pr"], reverse=True):
    summary_lines.append(
        f"{r['model']:<14} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} "
        f"{r['precision_at_recall_25']:>7.4f} {r['precision_at_recall_50']:>7.4f} "
        f"{r['precision_at_recall_75']:>7.4f} {r['f1']:>6.4f} {r['f2']:>6.4f}"
    )
summary_lines += [
    "",
    "All-features result:",
    f"  {'all_features':<12} AUC-ROC={all_feat_roc:.4f}  AUC-PR={all_feat_pr:.4f}  "
    f"P@R50={res_all['precision_at_recall_50']:.4f}  F1={res_all['f1']:.4f}  F2={res_all['f2']:.4f}",
    "",
    f"Best single-feature AUC-ROC : {best_single_roc:.4f}  ({best_feat_roc})",
    f"Best single-feature AUC-PR  : {best_single_pr:.4f}  ({best_feat_pr})",
    f"Rankings identical          : {rankings_match}",
    "",
    f"VERDICT: {verdict}",
    "",
    "Key Findings:",
]

if best_feat_roc == best_feat_pr:
    summary_lines.append(f"  * Best feature is {best_feat_roc.upper()} by both AUC-ROC and AUC-PR.")
    summary_lines.append(f"    Class imbalance does not change feature selection.")
else:
    summary_lines.append(f"  * Best feature by AUC-ROC: {best_feat_roc.upper()}")
    summary_lines.append(f"  * Best feature by AUC-PR:  {best_feat_pr.upper()}")
    summary_lines.append(f"    Rankings diverge — AUC-PR should be preferred for model selection.")

summary_lines += [
    "",
    f"All-features LR: AUC-ROC={all_feat_roc:.4f}  AUC-PR={all_feat_pr:.4f}",
    f"  -> This is the baseline all future methods must beat on BOTH metrics.",
]

with open(RESULTS_DIR / "sanity_summary.txt", "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"Saved {RESULTS_DIR}/sanity_summary.txt")
