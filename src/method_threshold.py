import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

from utils import (
    load_data_for, load_disease_config, get_splits, compute_binary_metrics,
    find_youden_threshold, precision_at_recall_levels, ensure_dir, RESULTS_DIR,
    get_literature_thresholds, build_threshold_prompt, THRESHOLDS_CACHE_DIR,
)

# Baseline metrics from all-features logistic regression
BASELINE_AUC_ROC = 0.5  
BASELINE_AUC_PR = 0.0
BASE_M1_DIR = RESULTS_DIR / "method1_threshold"
SANITY_MASTER_PATH = RESULTS_DIR / "sanity_check" / "master_sanity_summary.csv"
M1_MASTER_PATH = BASE_M1_DIR / "master_m1_summary.csv"


def main():
    # ── CLI Arguments ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Method 1 — Threshold Optimization")
    parser.add_argument("--disease", default="ra", help="Disease slug (e.g. ra, dm1)")
    parser.add_argument("--split-salt", default="", help="Labeled split variant (e.g. _seed2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print LLM prompt without loading model")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-query LLM even if cache exists")
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    M1_DIR = RESULTS_DIR / "method1_threshold" / disease.name
    ensure_dir(M1_DIR)

    # ── Load Baseline from Sanity Check ───────────────────────────────────────
    if SANITY_MASTER_PATH.exists():
        sanity_df = pd.read_csv(SANITY_MASTER_PATH)
        baseline_row = sanity_df[sanity_df["Disease"] == disease.name]
        if not baseline_row.empty:
            BASELINE_AUC_PR = baseline_row["All_Feat_AUC_PR"].values[0]
            BASELINE_AUC_ROC = baseline_row["All_Feat_AUC_ROC"].values[0]
            print(f"Loaded Baseline for {disease.name}: AUC-PR={BASELINE_AUC_PR}, AUC-ROC={BASELINE_AUC_ROC}")
        else:
            print(f"Warning: No baseline found for {disease.name} in sanity master.")
    else:
        print("Warning: master_sanity_summary.csv not found. Using default 0.0 baselines.")
    
    # ── Dry-run mode ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("=" * 70)
        print("=== DRY RUN: LLM prompt for literature thresholds ===")
        print("=" * 70)
        print(build_threshold_prompt(disease.full_name))
        print("\n" + "=" * 70)
        slug = disease.full_name.lower().replace(" ", "_").replace("-", "_")
        print(f"Cache would be written to: {THRESHOLDS_CACHE_DIR / f'{slug}.json'}")
        print("=" * 70)
        sys.exit(0)

    # ── Load ──────────────────────────────────────────────────────────────────────
    print("Loading data...")
    df, _ = load_data_for(disease.name, args.split_salt)
    train_df, test_df = get_splits(df)
    print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows\n")

    # ── Literature thresholds (auto-retrieved via MedGemma) ───────────────────────
    print(f"Fetching literature thresholds for {disease.full_name}...")
    literature_thresholds = get_literature_thresholds(disease.full_name, force_refresh=args.force_refresh)
    print(f"  Retrieved thresholds for: {list(literature_thresholds.keys())}\n")

    # ── Part 1A: Literature-based evaluation ─────────────────────────────────────
    print("=== Part 1A: Literature-based thresholds ===")

    lit_results = []

    for feat, (threshold, direction, source) in literature_thresholds.items():
        te     = test_df[[feat, "is_case"]].dropna()
        y_test = te["is_case"].values
        x_test = te[feat].values

        if y_test.sum() < 5:
            print(f"  {feat:6s}: skipped (too few positives)")
            continue

        # AUC uses raw feature value; flip sign for "below" so higher = more likely RA
        score   = x_test if direction == "above" else -x_test
        auc_roc = roc_auc_score(y_test, score)
        auc_pr  = average_precision_score(y_test, score)

        preds = (x_test > threshold).astype(int) if direction == "above" else (x_test < threshold).astype(int)
        m     = compute_binary_metrics(y_test, preds)

        lit_results.append({
            "feature":   feat,
            "threshold": threshold,
            "direction": direction,
            "auc_roc":   round(auc_roc, 4),
            "auc_pr":    round(auc_pr, 4),
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"], 4),
            "f1":        round(m["f1"], 4),
            "f2":        round(m["f2"], 4),
            "n_test":    len(te),
            "source":    source,
        })

        print(f"  {feat:6s} ({direction:5s} {threshold:>6}): "
              f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    lit_df = pd.DataFrame(lit_results)
    lit_df.to_csv(M1_DIR / "literature_results.csv", index=False)
    print(f"\nSaved {M1_DIR}/literature_results.csv")

    best_lit = lit_df.loc[lit_df["auc_pr"].idxmax()]
    print(f"\nBest literature (by AUC-PR): {best_lit['feature']} — AUC-ROC={best_lit['auc_roc']:.4f}, AUC-PR={best_lit['auc_pr']:.4f}")

    # ── Part 1B: Data-driven thresholds (Youden's index) ─────────────────────────
    print("\n=== Part 1B: Data-driven thresholds (Youden's index) ===")

    dd_results = []

    for feat in list(literature_thresholds.keys()):
        tr = train_df[[feat, "is_case"]].dropna()
        te = test_df[[feat, "is_case"]].dropna()

        if tr["is_case"].sum() < 5 or te["is_case"].sum() < 5:
            print(f"  {feat:6s}: skipped (too few positives)")
            continue

        y_train, x_train = tr["is_case"].values, tr[feat].values
        y_test,  x_test  = te["is_case"].values, te[feat].values

        # Determine direction from train AUC; flip score for "below" features
        auc_above = roc_auc_score(y_train, x_train)
        if auc_above >= 0.5:
            direction  = "above"
            score_tr   = x_train
            score_te   = x_test
        else:
            direction  = "below"
            score_tr   = -x_train
            score_te   = -x_test

        # Youden's index: J = sensitivity + specificity - 1 = TPR - FPR
        optimal_threshold_score, _, _ = find_youden_threshold(y_train, score_tr)
        # Convert back to original feature space
        optimal_threshold = optimal_threshold_score if direction == "above" else -optimal_threshold_score

        auc_roc = roc_auc_score(y_test, score_te)
        auc_pr  = average_precision_score(y_test, score_te)
        preds   = (x_test >= optimal_threshold).astype(int) if direction == "above" else (x_test <= optimal_threshold).astype(int)
        m       = compute_binary_metrics(y_test, preds)
        par     = precision_at_recall_levels(score_tr, y_train, score_te, y_test)

        dd_results.append({
            "feature":                  feat,
            "optimal_threshold":        round(float(optimal_threshold), 4),
            "direction":                direction,
            "auc_roc":                  round(auc_roc, 4),
            "auc_pr":                   round(auc_pr, 4),
            "precision":                round(m["precision"], 4),
            "recall":                   round(m["recall"], 4),
            "f1":                       round(m["f1"], 4),
            "f2":                       round(m["f2"], 4),
            "precision_at_recall_25":   par[0.25][0],
            "precision_at_recall_50":   par[0.50][0],
            "precision_at_recall_75":   par[0.75][0],
            "n_train":                  len(tr),
            "n_test":                   len(te),
        })

        print(f"  {feat:6s} ({direction:5s} {optimal_threshold:>8.4f}): "
              f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    dd_df = pd.DataFrame(dd_results)
    dd_df.to_csv(M1_DIR / "datadriven_results.csv", index=False)
    print(f"\nSaved {M1_DIR}/datadriven_results.csv")

    best_dd = dd_df.loc[dd_df["auc_pr"].idxmax()]
    print(f"\nBest data-driven (by AUC-PR): {best_dd['feature']} — AUC-ROC={best_dd['auc_roc']:.4f}, AUC-PR={best_dd['auc_pr']:.4f}  "
          f"threshold={best_dd['optimal_threshold']} ({best_dd['direction']})")

    # ── Part 1C: Comparison table and visualizations ──────────────────────────────
    print("\n=== Part 1C: Comparison ===")

    comp = lit_df[["feature", "threshold", "direction", "auc_roc", "auc_pr", "precision", "recall"]].rename(columns={
        "threshold": "literature_threshold",
        "direction": "literature_direction",
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
    comp = comp[["feature", "literature_threshold", "literature_direction", "literature_precision", "literature_recall",
                 "datadriven_threshold", "datadriven_direction", "datadriven_precision", "datadriven_recall", "auc_roc", "auc_pr"]]
    comp = comp.sort_values("auc_pr", ascending=False).reset_index(drop=True)

    comp.to_csv(M1_DIR / "comparison_table.csv", index=False)
    print(f"Saved {M1_DIR}/comparison_table.csv")
    print(comp.to_string(index=False))

    # ── Part 1D: Master Summary Aggregation (Clean Version) ───────────────────
    best_lit = lit_df.loc[lit_df["auc_pr"].idxmax()]
    best_dd = dd_df.loc[dd_df["auc_pr"].idxmax()]

    def fmt_formula(row, thresh_key):
        op = ">" if row["direction"] == "above" else "<"
        return f"{row['feature']} {op} {row[thresh_key]}"

    new_m1_row = {
        "Disease": disease.name,
        "Best_Lit_Feature": best_lit["feature"],
        "Best_Lit_Formula": fmt_formula(best_lit, "threshold"),
        "Best_Lit_AUC_PR": best_lit["auc_pr"],
        "Best_Lit_AUC_ROC": best_lit["auc_roc"],
        "Best_DD_Feature": best_dd["feature"],
        "Best_DD_Formula": fmt_formula(best_dd, "optimal_threshold"),
        "Best_DD_AUC_PR": best_dd["auc_pr"],
        "Best_DD_AUC_ROC": best_dd["auc_roc"]
    }

    M1_MASTER_PATH = RESULTS_DIR / "method1_threshold" / "master_m1_summary.csv"
    ensure_dir(M1_MASTER_PATH.parent)

    if M1_MASTER_PATH.exists():
        m1_master = pd.read_csv(M1_MASTER_PATH)
        m1_master = m1_master[m1_master["Disease"] != disease.name] # Update existing
        m1_master = pd.concat([m1_master, pd.DataFrame([new_m1_row])], ignore_index=True)
    else:
        m1_master = pd.DataFrame([new_m1_row])

    m1_master.sort_values("Disease").to_csv(M1_MASTER_PATH, index=False)
    print(f"Updated master Method 1 summary at: {M1_MASTER_PATH}")
    
    # ── Bar chart: AUC-PR by feature ─────────────────────────────────────────────
    features_ordered = comp["feature"].tolist()
    x     = np.arange(len(features_ordered))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC-ROC
    ax1.bar(x, comp["auc_roc"], width, color="#4C72B0")
    ax1.axhline(BASELINE_AUC_ROC, color="red", linestyle="--", linewidth=1.2,
                label=f"Baseline (all-features LR, AUC-ROC={BASELINE_AUC_ROC})")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.upper() for f in features_ordered])
    ax1.set_ylabel("AUC-ROC (test set)")
    ax1.set_title("Method 1: AUC-ROC by Feature (Data-Driven Threshold)")
    ax1.set_ylim(0.4, 0.72)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # AUC-PR
    ax2.bar(x, comp["auc_pr"], width, color="#DD8452")
    ax2.axhline(BASELINE_AUC_PR, color="red", linestyle="--", linewidth=1.2,
                label=f"Baseline (all-features LR, AUC-PR={BASELINE_AUC_PR})")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.upper() for f in features_ordered])
    ax2.set_ylabel("AUC-PR (test set)")
    ax2.set_title("Method 1: AUC-PR by Feature (Data-Driven Threshold)")
    ax2.set_ylim(0.0, 0.04)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(M1_DIR / "comparison_chart.png", dpi=150)
    plt.close()
    print(f"Saved {M1_DIR}/comparison_chart.png")

    # ── ROC curve for best feature with both thresholds marked ───────────────────
    best_feat  = comp.loc[0, "feature"]
    lit_thresh = float(comp.loc[0, "literature_threshold"])
    lit_dir    = lit_df.loc[lit_df["feature"] == best_feat, "direction"].values[0]
    dd_thresh  = float(comp.loc[0, "datadriven_threshold"])
    dd_dir     = dd_df.loc[dd_df["feature"] == best_feat, "direction"].values[0]

    te_best    = test_df[[best_feat, "is_case"]].dropna()
    y_best     = te_best["is_case"].values
    x_best     = te_best[best_feat].values
    score_best = x_best if lit_dir == "above" else -x_best

    fpr_curve, tpr_curve, _ = roc_curve(y_best, score_best)
    auc_best                 = roc_auc_score(y_best, score_best)

    def _roc_point(x_vals, y_vals, threshold, direction):
        """
        Compute (FPR, TPR) point on ROC curve for a given threshold.

        Args:
            x_vals: Feature values
            y_vals: Binary labels
            threshold: Classification threshold
            direction: "above" or "below" (direction of positive prediction)

        Returns:
            Tuple of (fpr, tpr) for this threshold
        """
        preds  = (x_vals >= threshold).astype(int) if direction == "above" else (x_vals <= threshold).astype(int)
        m      = compute_binary_metrics(y_vals, preds)
        tn     = int(((preds == 0) & (y_vals == 0)).sum())
        fpr_pt = m["fp"] / (m["fp"] + tn) if (m["fp"] + tn) > 0 else 0.0
        return fpr_pt, m["recall"]

    fpr_lit, tpr_lit = _roc_point(x_best, y_best, lit_thresh, lit_dir)
    fpr_dd,  tpr_dd  = _roc_point(x_best, y_best, dd_thresh,  dd_dir)

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
    plt.savefig(M1_DIR / "best_feature_roc.png", dpi=150)
    plt.close()
    print(f"Saved {M1_DIR}/best_feature_roc.png")

    # ── PR curves for all features ────────────────────────────────────────────────
    print("\n=== Generating PR curves for all features ===")

    features_list = comp["feature"].tolist()
    n_features = len(features_list)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 11))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feat in enumerate(features_list):
        ax = axes[i]

        # Get data for this feature
        te = test_df[[feat, "is_case"]].dropna()
        y_te = te["is_case"].values
        x_te = te[feat].values

        # Get direction and threshold info from comp table
        lit_dir = comp.loc[comp["feature"] == feat, "literature_direction"].values[0]
        lit_thresh = comp.loc[comp["feature"] == feat, "literature_threshold"].values[0]
        dd_thresh = comp.loc[comp["feature"] == feat, "datadriven_threshold"].values[0]
        lit_prec = comp.loc[comp["feature"] == feat, "literature_precision"].values[0]
        lit_rec = comp.loc[comp["feature"] == feat, "literature_recall"].values[0]
        dd_prec = comp.loc[comp["feature"] == feat, "datadriven_precision"].values[0]
        dd_rec = comp.loc[comp["feature"] == feat, "datadriven_recall"].values[0]
        auc_pr_val = comp.loc[comp["feature"] == feat, "auc_pr"].values[0]

        # Compute score (flip for "below" direction)
        score_te = x_te if lit_dir == "above" else -x_te
        prevalence = y_te.mean()

        # PR curve
        prec_c, rec_c, _ = precision_recall_curve(y_te, score_te)

        ax.plot(rec_c, prec_c, color="#4C72B0", lw=1.5, label=f"AUC-PR={auc_pr_val:.4f}")
        ax.axhline(prevalence, color="gray", linestyle=":", lw=1, label=f"Baseline ({prevalence:.3f})")
        ax.scatter([lit_rec], [lit_prec], s=80, color="green", zorder=5,
                   label=f"Lit ({lit_thresh:.1f})")
        ax.scatter([dd_rec], [dd_prec], s=80, color="orange", marker="^", zorder=5,
                   label=f"DD ({dd_thresh:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(prec_c.max() * 1.15, prevalence * 3))
        ax.set_title(feat.upper(), fontsize=10)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Method 1: Precision-Recall Curves (all features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(M1_DIR / "pr_curves.png", dpi=150)
    plt.close()
    print(f"Saved {M1_DIR}/pr_curves.png")

    # ── Summary text ──────────────────────────────────────────────────────────────
    best_lit_row     = lit_df.loc[lit_df["feature"] == best_feat].iloc[0]
    best_dd_row      = dd_df.loc[dd_df["feature"]   == best_feat].iloc[0]
    auc_pr_vs_base   = best_dd_row["auc_pr"] - BASELINE_AUC_PR
    auc_roc_vs_base  = best_dd_row["auc_roc"] - BASELINE_AUC_ROC

    summary = f"""Method 1: Threshold Optimization Results
    =========================================

    Best single feature: {best_feat.upper()} (selected by AUC-PR)

    Literature threshold: {best_lit_row['threshold']} ({best_lit_row['direction']})
      AUC-ROC: {best_lit_row['auc_roc']:.4f}
      AUC-PR:  {best_lit_row['auc_pr']:.4f}
      Precision: {best_lit_row['precision']:.4f}
      Recall:    {best_lit_row['recall']:.4f}
      F1: {best_lit_row['f1']:.4f}
      F2: {best_lit_row['f2']:.4f}

    Data-driven threshold: {best_dd_row['optimal_threshold']:.4f} ({best_dd_row['direction']})
      AUC-ROC: {best_dd_row['auc_roc']:.4f}
      AUC-PR:  {best_dd_row['auc_pr']:.4f}
      Precision: {best_dd_row['precision']:.4f}
      Recall:    {best_dd_row['recall']:.4f}
      F1: {best_dd_row['f1']:.4f}
      F2: {best_dd_row['f2']:.4f}
      Precision@Recall(0.25): {best_dd_row['precision_at_recall_25']:.4f}
      Precision@Recall(0.50): {best_dd_row['precision_at_recall_50']:.4f}
      Precision@Recall(0.75): {best_dd_row['precision_at_recall_75']:.4f}

    Comparison to baseline (all-features logistic regression):
      Baseline AUC-ROC: {BASELINE_AUC_ROC:.3f}  |  Best feature AUC-ROC: {best_dd_row['auc_roc']:.3f}  |  Diff: {auc_roc_vs_base:+.3f}
      Baseline AUC-PR:  {BASELINE_AUC_PR:.3f}  |  Best feature AUC-PR:  {best_dd_row['auc_pr']:.3f}  |  Diff: {auc_pr_vs_base:+.3f}

    Key insight:
      Single-feature thresholds underperform the all-features baseline on both metrics.
      Data-driven (Youden) thresholds optimize the precision-recall tradeoff but cannot
      match the predictive power of multi-feature models.
    """

    with open(M1_DIR / "method1_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {M1_DIR}/method1_summary.txt")
    print("\n" + summary)


if __name__ == "__main__":
    main()
