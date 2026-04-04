"""
Sanity check — logistic regression (single-feature and all-features).

Purpose: verify the data pipeline is free of leakage.
Expected AUC range if clean: ~0.55–0.75 for single features.
AUC > 0.95 on a single feature strongly suggests label or temporal leakage.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve

from utils import load_data, get_splits, compute_binary_metrics, find_youden_threshold, ensure_dir, RESULTS_DIR

ensure_dir(RESULTS_DIR)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df, features = load_data()
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
    """
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, proba)

    threshold, fpr, tpr = find_youden_threshold(y_test, proba)
    preds = (proba >= threshold).astype(int)
    m = compute_binary_metrics(y_test, preds)

    return {
        "model":     label,
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "auc":       round(auc, 4),
        "threshold": round(threshold, 4),
        "precision": round(m["precision"], 4),
        "recall":    round(m["recall"], 4),
        "fpr_arr":   fpr,
        "tpr_arr":   tpr,
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
    print(f"  {feat:8s}: AUC={res['auc']:.4f}  precision={res['precision']:.4f}  recall={res['recall']:.4f}  (n_test={res['n_test']:,})")

# ── All-features model ────────────────────────────────────────────────────────
print("\n=== All-features logistic regression ===")
tr_all = train_df[features + ["is_case"]].dropna()
te_all = test_df[features + ["is_case"]].dropna()

res_all = fit_evaluate(tr_all[features].values, tr_all["is_case"].values,
                       te_all[features].values, te_all["is_case"].values, label="all_features")
results.append(res_all)
print(f"  all_features: AUC={res_all['auc']:.4f}  precision={res_all['precision']:.4f}  recall={res_all['recall']:.4f}  (n_test={res_all['n_test']:,})")

# ── Verdict ───────────────────────────────────────────────────────────────────
single_aucs      = [r["auc"] for r in results if r["model"] != "all_features"]
best_single_auc  = max(single_aucs) if single_aucs else 0.0
best_single_feat = next(r["model"] for r in results if r["auc"] == best_single_auc and r["model"] != "all_features")
all_feat_auc     = res_all["auc"]

if best_single_auc > 0.95:
    verdict = "FAIL — likely data leakage (single-feature AUC > 0.95)"
elif best_single_auc >= 0.85:
    verdict = "WARNING — single-feature AUC 0.85–0.95, investigate potential leakage"
elif all_feat_auc >= 0.95:
    verdict = "FAIL — likely data leakage (all-features AUC >= 0.95)"
else:
    verdict = "PASS — AUC in expected range, pipeline appears clean"

print(f"\n{'='*55}")
print(f"Best single-feature AUC : {best_single_auc:.4f}  ({best_single_feat})")
print(f"All-features AUC        : {all_feat_auc:.4f}")
print(f"Verdict                 : {verdict}")
print(f"{'='*55}\n")

# ── Save CSV results ──────────────────────────────────────────────────────────
csv_rows   = [{k: v for k, v in r.items() if k not in ("fpr_arr", "tpr_arr")} for r in results]
results_df = pd.DataFrame(csv_rows)
results_df.to_csv(RESULTS_DIR / "sanity_check_results.csv", index=False)
print(f"Saved {RESULTS_DIR}/sanity_check_results.csv")

# ── ROC curves plot ───────────────────────────────────────────────────────────
single_results = [r for r in results if r["model"] != "all_features"]
top3           = sorted(single_results, key=lambda r: r["auc"], reverse=True)[:3]
plot_models    = top3 + [res_all]

fig, ax = plt.subplots(figsize=(7, 6))
for r in plot_models:
    ax.plot(r["fpr_arr"], r["tpr_arr"], label=f"{r['model']} (AUC={r['auc']:.3f})")
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
summary_lines = [
    "Sanity Check Summary",
    "=" * 40,
    "",
    "Single-feature results:",
]
for r in sorted(single_results, key=lambda r: r["auc"], reverse=True):
    summary_lines.append(
        f"  {r['model']:8s}  AUC={r['auc']:.4f}  precision={r['precision']:.4f}  recall={r['recall']:.4f}"
    )
summary_lines += [
    "",
    "All-features result:",
    f"  all_features  AUC={all_feat_auc:.4f}  precision={res_all['precision']:.4f}  recall={res_all['recall']:.4f}",
    "",
    f"Best single-feature AUC : {best_single_auc:.4f}  ({best_single_feat})",
    f"All-features AUC        : {all_feat_auc:.4f}",
    "",
    f"VERDICT: {verdict}",
]

with open(RESULTS_DIR / "sanity_summary.txt", "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"Saved {RESULTS_DIR}/sanity_summary.txt")
