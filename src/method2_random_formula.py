"""
Issue #10 / Method 2: Random formula generation.

Generates 10,000 random mathematical formulas combining 2-4 CBC features
and evaluates them on the frozen test set. Serves as a baseline for more
sophisticated search methods (genetic programming, LLM-guided).

Research question: can random CBC combinations beat single-feature thresholds
(AUC-PR=0.014) or all-features logistic regression (AUC-PR=0.017)?
"""

# Standard library
import argparse
from datetime import datetime
import sys
import warnings

# Third-party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Local
sys.path.insert(0, "src")
from utils import (
    load_data_for, load_disease_config, load_ml_config, get_splits, compute_binary_metrics,
    find_youden_threshold, precision_at_recall_levels, ensure_dir, RESULTS_DIR,
    eval_formula_scores, evaluate_formula_full, get_cv_folds, cv_summary,
    count_formula_features, load_per_k_baselines,
)


def main():
    parser = argparse.ArgumentParser(description="Method 2: Random formula generation")
    parser.add_argument("--disease", default="ra", help="Disease slug (e.g. ra, dm1)")
    parser.add_argument("--split-salt", default="", help="Labeled split variant (e.g. _seed2)")
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    ml = load_ml_config()
    N_FORMULAS      = ml.method2.n_formulas
    SEED            = ml.seed
    BAD_FRAC        = ml.method2.bad_frac
    BASELINE_AUC_PR = ml.baselines.lr_auc_pr

    OUT_DIR = RESULTS_DIR / "method2_random" / disease.name
    ensure_dir(OUT_DIR)

    # ── Part A: Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    df, features = load_data_for(disease.name, args.split_salt)
    train_df, test_df = get_splits(df)
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    print(f"Features: {features}\n")

    # ── Part B: Formula generation ────────────────────────────────────────────────

    def _random_term(feature, rng):
        """Apply a random unary transform to one feature."""
        op = rng.choice(["identity", "sqrt", "log", "square"])
        if op == "identity":
            return feature
        elif op == "sqrt":
            return f"sqrt(abs({feature}))"
        elif op == "log":
            return f"log(abs({feature})+1)"
        else:  # square
            return f"({feature}**2)"

    def random_formula_k(rng, feats, k):
        """Generate a random formula string with exactly k distinct features."""
        chosen = list(rng.choice(feats, size=k, replace=False))
        terms = [_random_term(f, rng) for f in chosen]
        if k == 1:
            return terms[0]
        result = terms[0]
        for term in terms[1:]:
            op = rng.choice(["+", "-", "*", "/"])
            if op == "/":
                result = f"({result}) / (abs({term})+1e-6)"
            else:
                result = f"({result}) {op} ({term})"
        return result

    def generate_formulas_per_k(n_per_k, seed, feats):
        """Generate n_per_k unique formulas for each k=1..len(feats).
        Returns list of (formula_str, k) tuples."""
        all_formulas = []
        for k in range(1, len(feats) + 1):
            rng = np.random.RandomState(seed + k * 1000)
            seen, formulas_k, attempts = set(), [], 0
            while len(formulas_k) < n_per_k:
                f = random_formula_k(rng, feats, k)
                attempts += 1
                if f not in seen:
                    seen.add(f)
                    formulas_k.append(f)
                if attempts > n_per_k * 20:
                    break
            print(f"  k={k}: {len(formulas_k):,} formulas ({attempts:,} attempts)")
            all_formulas.extend((f, k) for f in formulas_k)
        return all_formulas

    # ── Part C: Formula evaluation ────────────────────────────────────────────────
    # eval_formula_scores and evaluate_formula_full are imported from utils

    # ── Part D: Run experiment ────────────────────────────────────────────────────
    N_PER_K = N_FORMULAS // len(features)

    print(f"=== Generating formulas: {N_PER_K} per k, k=1..{len(features)} ===")
    formulas_with_k = generate_formulas_per_k(N_PER_K, SEED, features)
    print(f"Generated {len(formulas_with_k):,} total formulas\n")

    # Flat list of formula strings for CV section
    all_formula_strs = [f for f, _ in formulas_with_k]

    print("=== Evaluating formulas ===")
    results = []
    skipped = 0
    for i, (formula, k) in enumerate(formulas_with_k):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{len(formulas_with_k):,}  valid={len(results):,}  skipped={skipped}")
        row = evaluate_formula_full(formula, train_df, test_df, features)
        if row is None:
            skipped += 1
        else:
            row["num_features"] = k
            results.append(row)

    print(f"\nDone: {len(results):,} valid formulas, {skipped:,} skipped\n")

    results_df = pd.DataFrame(results).sort_values("auc_pr", ascending=False).reset_index(drop=True)

    # Save all results
    all_path = OUT_DIR / "all_formulas.csv"
    results_df.to_csv(all_path, index=False)
    print(f"Saved {all_path}")

    # Save top 10 by frozen AUC-PR (unchanged)
    top10 = results_df.head(10)
    top_path = OUT_DIR / "top_formulas.csv"
    top10.to_csv(top_path, index=False)
    print(f"Saved {top_path}\n")

    # Print top 10
    print("=== Top 10 formulas by AUC-PR ===")
    print(f"{'Rank':<5} {'AUC-ROC':>8} {'AUC-PR':>8} {'P@R25':>7} {'P@R50':>7} {'P@R75':>7} {'F1':>6} {'F2':>6}  formula")
    print("-" * 120)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{rank:<5} {row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
              f"{row['precision_at_recall_25']:>7.4f} {row['precision_at_recall_50']:>7.4f} "
              f"{row['precision_at_recall_75']:>7.4f} {row['f1']:>6.4f} {row['f2']:>6.4f}  {row['formula']}")

    # Compare best vs baseline
    best = results_df.iloc[0]
    print(f"\n=== Best formula vs baseline ===")
    print(f"Baseline (all-features LR): AUC-ROC=0.6580  AUC-PR={BASELINE_AUC_PR:.4f}")
    print(f"Best random formula       : AUC-ROC={best['auc_roc']:.4f}  AUC-PR={best['auc_pr']:.4f}")
    beats = best['auc_pr'] > BASELINE_AUC_PR
    print(f"Beats baseline AUC-PR: {'YES' if beats else 'NO'}")

    # Distribution stats
    print(f"\n=== AUC-PR distribution ({len(results_df):,} valid formulas) ===")
    print(f"  max:    {results_df['auc_pr'].max():.4f}")
    print(f"  99th:   {results_df['auc_pr'].quantile(0.99):.4f}")
    print(f"  95th:   {results_df['auc_pr'].quantile(0.95):.4f}")
    print(f"  median: {results_df['auc_pr'].median():.4f}")
    print(f"  min:    {results_df['auc_pr'].min():.4f}")
    print(f"  # beating baseline ({BASELINE_AUC_PR}): {(results_df['auc_pr'] > BASELINE_AUC_PR).sum()}")

    # ── Part E: Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # PR curves for top 5
    tr_clean = train_df[features + ["is_case"]].dropna()
    te_clean = test_df[features + ["is_case"]].dropna()
    y_tr = tr_clean["is_case"].values
    y_te = te_clean["is_case"].values
    prevalence = y_te.mean()

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for i, (_, row) in enumerate(results_df.head(5).iterrows()):
        ax = axes[i]
        local = {f: te_clean[f].values.astype(float) for f in features}
        local["sqrt"] = np.sqrt
        local["log"]  = np.log1p
        local["abs"]  = np.abs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = eval(row["formula"], {"__builtins__": {}}, local)  # noqa: S307
        score = np.asarray(score, dtype=float)
        if not np.isfinite(score).all():
            score[~np.isfinite(score)] = np.nanmedian(score[np.isfinite(score)])
        # Flip if needed
        if roc_auc_score(y_te, score) < 0.5:
            score = -score
        prec_c, rec_c, _ = precision_recall_curve(y_te, score)
        ax.plot(rec_c, prec_c, color="#4C72B0", lw=1.5, label=f"AUC-PR={row['auc_pr']:.4f}")
        ax.axhline(prevalence, color="gray", linestyle=":", lw=1, label=f"Baseline ({prevalence:.4f})")
        ax.set_xlim(0, 1)
        ax.set_title(f"#{i+1}", fontsize=10)
        ax.set_xlabel("Recall", fontsize=8)
        if i == 0:
            ax.set_ylabel("Precision", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    fig.suptitle("Method 2: PR Curves — Top 5 Random Formulas", fontsize=12)
    plt.tight_layout()
    pr_path = OUT_DIR / "top_pr_curves.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"Saved {pr_path}")

    # AUC-PR histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(results_df["auc_pr"], bins=80, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(BASELINE_AUC_PR, color="red", linestyle="--", lw=1.5, label=f"Baseline ({BASELINE_AUC_PR})")
    ax.axvline(best["auc_pr"], color="green", linestyle="--", lw=1.5, label=f"Best ({best['auc_pr']:.4f})")
    ax.set_xlabel("AUC-PR")
    ax.set_ylabel("Count")
    ax.set_title(f"Method 2: AUC-PR Distribution ({len(results_df):,} random formulas)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = OUT_DIR / "auc_pr_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved {hist_path}")

    
    # ── Part E2: CV-based winner selection ───────────────────────────────────────
    print("\n=== CV-based winner selection (top-50 by train AUC-PR) ===")
    tr_clean = train_df[features + ["is_case"]].dropna()
    te_clean = test_df[features + ["is_case"]].dropna()

    # Rank all formulas by train AUC-PR (single pass, no frozen test used)
    from sklearn.metrics import average_precision_score as _aps, roc_auc_score as _roc
    train_scores_list = []
    for formula in all_formula_strs:
        sc = eval_formula_scores(formula, tr_clean, features)
        if sc is None or not np.isfinite(sc).any():
            continue
        y_tr_cv = tr_clean["is_case"].values
        if y_tr_cv.sum() < 5:
            continue
        try:
            auc_roc_tr = float(_roc(y_tr_cv, sc))
            if auc_roc_tr < 0.5:
                sc = -sc
            train_scores_list.append({"formula": formula, "train_auc_pr": float(_aps(y_tr_cv, sc))})
        except Exception:
            pass

    train_rank_df = pd.DataFrame(train_scores_list).sort_values("train_auc_pr", ascending=False)
    top50_formulas = train_rank_df.head(50)["formula"].tolist()
    print(f"  Top-50 pool size: {len(top50_formulas)}")

    # CV on the top-50 pool
    cv_folds = get_cv_folds(train_df)
    cv_rows = []
    for i, formula in enumerate(top50_formulas):
        fold_prs = []
        for fold_train_df, fold_val_df in cv_folds:
            ft = fold_train_df[features + ["is_case"]].dropna()
            fv = fold_val_df[features + ["is_case"]].dropna()
            if ft["is_case"].sum() < 5 or fv["is_case"].sum() < 5:
                continue
            sc_ft = eval_formula_scores(formula, ft, features)
            sc_fv = eval_formula_scores(formula, fv, features)
            if sc_ft is None or sc_fv is None:
                continue
            try:
                auc_roc_ft = float(_roc(ft["is_case"].values, sc_ft))
                if auc_roc_ft < 0.5:
                    sc_fv = -sc_fv
                fold_prs.append(float(_aps(fv["is_case"].values, sc_fv)))
            except Exception:
                pass

        if len(fold_prs) < 3:
            print(f"  Formula {i}: fewer than 3 valid CV folds — skipped")
            continue

        s = cv_summary(fold_prs)
        train_auc_pr = train_rank_df.loc[train_rank_df["formula"] == formula, "train_auc_pr"].values[0]
        cv_rows.append({
            "formula": formula,
            "train_auc_pr": train_auc_pr,
            "cv_auc_pr_mean": s["mean"],
            "cv_auc_pr_std": s["std"],
            "cv_auc_pr_ci95_low": s["ci95_low"],
            "cv_auc_pr_ci95_high": s["ci95_high"],
            "frozen_test_auc_pr": None,
        })

    if not cv_rows:
        print("  [WARN] No valid CV results — falling back to frozen test winner")
        cv_winner_formula = results_df.iloc[0]["formula"]
        cv_winner_frozen  = results_df.iloc[0]["auc_pr"]
    else:
        cv_df = pd.DataFrame(cv_rows).sort_values("cv_auc_pr_mean", ascending=False).reset_index(drop=True)
        cv_winner_formula = cv_df.iloc[0]["formula"]
        # Evaluate CV winner on frozen test exactly once
        cv_winner_metrics = evaluate_formula_full(cv_winner_formula, train_df, test_df, features)
        cv_winner_frozen  = cv_winner_metrics["auc_pr"] if cv_winner_metrics else float("nan")
        cv_df.loc[cv_df["formula"] == cv_winner_formula, "frozen_test_auc_pr"] = cv_winner_frozen
        cv_df.to_csv(OUT_DIR / "top_formulas_cv.csv", index=False)
        print(f"  CV winner: {cv_winner_formula}")
        print(f"  CV AUC-PR mean: {cv_df.iloc[0]['cv_auc_pr_mean']:.4f}  Frozen test: {cv_winner_frozen:.4f}")
        print(f"  Saved {OUT_DIR}/top_formulas_cv.csv")

    # ── Part F: Per-k best results ────────────────────────────────────────────
    per_k_bl = load_per_k_baselines(disease.name, args.split_salt)

    per_k_rows = []
    for k in range(1, len(features) + 1):
        k_df = results_df[results_df["num_features"] == k]
        if k_df.empty:
            continue
        best_k = k_df.iloc[0]
        bl_pr = per_k_bl.get(k, {}).get("auc_pr")
        per_k_rows.append({
            "K":                 k,
            "Best_Formula":      best_k["formula"],
            "AUC_PR":            best_k["auc_pr"],
            "AUC_ROC":           best_k["auc_roc"],
            "N_Formulas_Tested": len(k_df),
            "Baseline_AUC_PR":   bl_pr,
            "Delta_vs_Baseline": round(best_k["auc_pr"] - bl_pr, 4) if bl_pr is not None else None,
        })

    per_k_path = OUT_DIR / "per_k_best.csv"
    pd.DataFrame(per_k_rows).to_csv(per_k_path, index=False)
    print(f"Saved per-k best formulas to {per_k_path}")

    print("\n=== Per-k best vs LR baseline ===")
    for row in per_k_rows:
        bl_str = f"baseline={row['Baseline_AUC_PR']:.4f}  delta={row['Delta_vs_Baseline']:+.4f}" if row["Baseline_AUC_PR"] is not None else "baseline=N/A"
        print(f"  k={row['K']}: AUC-PR={row['AUC_PR']:.4f}  {bl_str}")

    # ── Part G: Master Summary Aggregation ────────────────────────────────────
    best = results_df.iloc[0]
    best_k = int(best["num_features"])
    best_bl_pr = per_k_bl.get(best_k, {}).get("auc_pr")

    new_m2_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Disease": disease.name,
        "Split_Salt": args.split_salt,
        "Best_Random_Formula": best["formula"],
        "Best_Random_AUC_PR": round(best["auc_pr"], 4),
        "Best_Random_AUC_ROC": round(best["auc_roc"], 4),
        "Num_Features_Best": best_k,
        "Baseline_AUC_PR_At_K": best_bl_pr,
        "Delta_vs_Baseline_K": round(best["auc_pr"] - best_bl_pr, 4) if best_bl_pr is not None else None,
        "N_Formulas_Tested": len(results_df),
        "CV_Winner_Formula": cv_winner_formula,
        "CV_Winner_Frozen_Test_AUC_PR": round(cv_winner_frozen, 4) if np.isfinite(cv_winner_frozen) else None,
    }

    M2_MASTER_PATH = RESULTS_DIR / "method2_random" / "master_m2_summary.csv"
    ensure_dir(M2_MASTER_PATH.parent)

    if M2_MASTER_PATH.exists():
        m2_master = pd.concat([pd.read_csv(M2_MASTER_PATH), pd.DataFrame([new_m2_row])], ignore_index=True)
    else:
        m2_master = pd.DataFrame([new_m2_row])

    m2_master.to_csv(M2_MASTER_PATH, index=False)
    print(f"Updated master Method 2 summary at: {M2_MASTER_PATH}")
    print("\nMethod 2 complete.")

if __name__ == "__main__":
    main()
