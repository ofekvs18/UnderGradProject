"""
Issue #17 / Method 3: Genetic Programming with gplearn.

Uses SymbolicTransformer to evolve formula-based biomarkers from CBC data.
Custom fitness combines AUC-ROC + 2*AUC-PR (AUC-PR gets 2x weight as primary
metric). Hall-of-fame programs are evaluated via program.execute(X) directly —
no formula string parsing.

Research question: can evolutionary search discover CBC combinations that
outperform random search (Method 2, AUC-PR=0.0174) and logistic regression
(AUC-PR=0.017)?
"""

# Standard library
import argparse
import sys
import warnings

# Third-party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness

# Local
sys.path.insert(0, "src")
from utils import (
    load_data_for, load_disease_config, load_ml_config, get_splits, compute_binary_metrics,
    find_youden_threshold, precision_at_recall_levels, ensure_dir, RESULTS_DIR,
)

_ml = load_ml_config()

M2_BEST_AUC_PR = _ml.baselines.m2_best_auc_pr
LR_AUC_PR      = _ml.baselines.lr_auc_pr
LR_AUC_ROC     = _ml.baselines.lr_auc_roc
SEED           = _ml.seed

SMALL_CONFIG  = dict(**_ml.method3.gp_configs.small)
MEDIUM_CONFIG = dict(**_ml.method3.gp_configs.medium)
LARGE_CONFIG  = dict(**_ml.method3.gp_configs.large)

PHASE1_AUC_PR_THRESHOLD = _ml.method3.phase1_auc_pr_threshold
BAD_FRAC                = _ml.method3.bad_frac
HALL_OF_FAME            = _ml.method3.hall_of_fame
N_COMPONENTS            = _ml.method3.n_components

CONFIG_ATTEMPTS = [
    dict(parsimony_coefficient=a.parsimony_coefficient,
         function_set=list(a.function_set))
    for a in _ml.method3.config_attempts
]


def _combined_auc(y, y_pred, sample_weight):
    """
    Fitness = AUC-ROC + 2 * AUC-PR.
    AUC-PR gets 2x weight since it is the primary metric in low-prevalence
    settings. Handles inverse predictivity and bad values. Returns 0.0 on
    failure so gplearn can safely discard the program.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    bad = ~np.isfinite(y_pred)
    if bad.mean() > BAD_FRAC:
        return 0.0
    if bad.any():
        y_pred = y_pred.copy()
        y_pred[bad] = np.nanmedian(y_pred[~bad]) if (~bad).any() else 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc_roc = roc_auc_score(y, y_pred)
            auc_pr  = average_precision_score(y, y_pred)
        if auc_roc < 0.5:
            auc_roc = 1.0 - auc_roc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc_pr = average_precision_score(y, -y_pred)
        return float(auc_roc + 2.0 * auc_pr)
    except Exception:
        return 0.0


combined_auc_fitness = make_fitness(function=_combined_auc, greater_is_better=True)


def evaluate_program(program, X_train, y_train, X_test, y_test):
    """
    Evaluate a gplearn program using program.execute(X) directly.
    Returns a metrics dict or None if the program produces invalid output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score_tr = np.asarray(program.execute(X_train), dtype=float)
        score_te = np.asarray(program.execute(X_test),  dtype=float)

    for scores in (score_tr, score_te):
        bad = ~np.isfinite(scores)
        if bad.mean() > BAD_FRAC:
            return None
        if bad.any():
            scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0

    if y_train.sum() < 5 or y_test.sum() < 5:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc_roc = float(roc_auc_score(y_test, score_te))
            auc_pr  = float(average_precision_score(y_test, score_te))
    except Exception:
        return None

    if auc_roc < 0.5:
        score_tr = -score_tr
        score_te = -score_te
        auc_roc  = 1.0 - auc_roc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc_pr = float(average_precision_score(y_test, score_te))

    threshold, _, _ = find_youden_threshold(y_train, score_tr)
    preds = (score_te >= threshold).astype(int)
    m   = compute_binary_metrics(y_test, preds)
    par = precision_at_recall_levels(score_tr, y_train, score_te, y_test)

    return {
        "formula":                str(program),
        "auc_roc":                round(auc_roc, 4),
        "auc_pr":                 round(auc_pr, 4),
        "precision":              round(m["precision"], 4),
        "recall":                 round(m["recall"], 4),
        "f1":                     round(m["f1"], 4),
        "f2":                     round(m["f2"], 4),
        "precision_at_recall_25": par[0.25][0],
        "precision_at_recall_50": par[0.50][0],
        "precision_at_recall_75": par[0.75][0],
    }


def run_gp_attempt(size_config, parsimony_coefficient, function_set, attempt_label, features,
                   X_train, y_train, X_test, y_test):
    """
    Fit SymbolicTransformer and return evaluated programs.

    Args:
        size_config: Dict with 'population_size' and 'generations'
        parsimony_coefficient: Penalty for program complexity (0.0 = no penalty)
        function_set: List of function names for GP primitives
        attempt_label: Description for logging
        features: List of feature names
        X_train, y_train, X_test, y_test: Data arrays

    Returns:
        List of (program, metrics_dict) tuples sorted by AUC-PR descending
    """
    print(f"\n=== {attempt_label} ===")
    print(f"  pop={size_config['population_size']}  gen={size_config['generations']}"
          f"  parsimony={parsimony_coefficient}"
          f"  functions={function_set}\n")

    gp = SymbolicTransformer(
        **size_config,
        hall_of_fame=HALL_OF_FAME,
        n_components=N_COMPONENTS,
        feature_names=features,
        function_set=function_set,
        metric=combined_auc_fitness,
        parsimony_coefficient=parsimony_coefficient,
        random_state=SEED,
        n_jobs=1,
        verbose=1,
    )
    gp.fit(X_train, y_train)
    print("\nFitting complete.")

    programs = gp._best_programs
    evaluated = []
    skipped = 0
    for prog in programs:
        if prog is None:
            skipped += 1
            continue
        row = evaluate_program(prog, X_train, y_train, X_test, y_test)
        if row is None:
            skipped += 1
        else:
            evaluated.append((prog, row))

    evaluated.sort(key=lambda x: x[1]["auc_pr"], reverse=True)
    best_auc_pr = evaluated[0][1]["auc_pr"] if evaluated else 0.0
    print(f"Evaluated {len(evaluated)} valid programs ({skipped} skipped) — "
          f"best AUC-PR={best_auc_pr:.4f}")
    return evaluated


def main():
    parser = argparse.ArgumentParser(description="Method 3: Genetic Programming")
    parser.add_argument("--disease", default="ra", help="Disease slug (e.g. ra, dm1)")
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    OUT_DIR = RESULTS_DIR / "method3_gp"
    ensure_dir(OUT_DIR)

    # ── Part A: Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    df, features = load_data_for(disease.name)
    train_df, test_df = get_splits(df)
    tr_clean = train_df[features + ["is_case"]].dropna()
    te_clean = test_df[features + ["is_case"]].dropna()

    X_train = tr_clean[features].values.astype(float)
    y_train = tr_clean["is_case"].values
    X_test  = te_clean[features].values.astype(float)
    y_test  = te_clean["is_case"].values

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Features: {features}\n")

    # ── Part B: Run with retry logic ──────────────────────────────────────────────
    evaluated = []
    used_attempt = None

    for attempt_num, cfg in enumerate(CONFIG_ATTEMPTS):
        label = (f"Phase 1 — LARGE config (attempt {attempt_num})"
                 if attempt_num > 0
                 else "Phase 1 — LARGE config (baseline attempt)")
        evaluated = run_gp_attempt(
            LARGE_CONFIG, **cfg, attempt_label=label,
            features=features, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )
        used_attempt = attempt_num

        if not evaluated:
            print(f"  ->No valid programs. Trying next config.\n")
            continue

        best_auc_pr = evaluated[0][1]["auc_pr"]
        if best_auc_pr >= PHASE1_AUC_PR_THRESHOLD:
            print(f"\n  ->AUC-PR={best_auc_pr:.4f} >= {PHASE1_AUC_PR_THRESHOLD} threshold. "
                  f"Phase 1 PASSED.\n")
            break
        elif attempt_num < len(CONFIG_ATTEMPTS) - 1:
            print(f"\n  ->AUC-PR={best_auc_pr:.4f} < {PHASE1_AUC_PR_THRESHOLD} threshold. "
                  f"Trying config adjustment {attempt_num + 1}.\n")
        else:
            print(f"\n  ->AUC-PR={best_auc_pr:.4f} after {attempt_num + 1} attempts. "
                  f"Discontinuing scale-up.\n")

    if not evaluated:
        print("No valid programs found after all attempts. Exiting.")
        sys.exit(1)

    best_prog, best_row = evaluated[0]
    results_df = pd.DataFrame([row for _, row in evaluated])

    all_path = OUT_DIR / "all_programs.csv"
    results_df.to_csv(all_path, index=False)
    print(f"Saved {all_path}")

    top_path = OUT_DIR / "top_formulas.csv"
    results_df.head(10).to_csv(top_path, index=False)
    print(f"Saved {top_path}\n")

    # Print top 10
    print("=== Top 10 GP programs by AUC-PR ===")
    print(f"{'Rank':<5} {'AUC-ROC':>8} {'AUC-PR':>8} {'P@R25':>7} {'P@R50':>7} "
          f"{'P@R75':>7} {'F1':>6} {'F2':>6}  formula")
    print("-" * 120)
    for rank, (_, row) in enumerate(evaluated[:10], 1):
        short = row["formula"][:60] + "..." if len(row["formula"]) > 60 else row["formula"]
        print(f"{rank:<5} {row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
              f"{row['precision_at_recall_25']:>7.4f} {row['precision_at_recall_50']:>7.4f} "
              f"{row['precision_at_recall_75']:>7.4f} {row['f1']:>6.4f} {row['f2']:>6.4f}  {short}")

    # ── Part C: Validation checks ─────────────────────────────────────────────────
    print(f"\n=== Validation ===")
    auc_roc_ok = 0.52 < best_row["auc_roc"] < 0.95
    print(f"Best AUC-ROC: {best_row['auc_roc']:.4f}  (must be 0.52-0.95): "
          f"{'PASS' if auc_roc_ok else 'FAIL'}")
    if not auc_roc_ok:
        if best_row["auc_roc"] <= 0.52:
            print("  WARNING: GP did not beat random chance — try more generations.")
        else:
            print("  WARNING: AUC-ROC suspiciously high — check for data leakage.")

    # ── Part D: Compare vs baselines ──────────────────────────────────────────────
    print(f"\n=== GP vs baselines ===")
    print(f"Baseline (all-features LR)  : AUC-ROC={LR_AUC_ROC:.4f}  AUC-PR={LR_AUC_PR:.4f}")
    print(f"Method 2 best random formula: AUC-PR={M2_BEST_AUC_PR:.4f}")
    print(f"Method 3 GP best            : AUC-ROC={best_row['auc_roc']:.4f}  "
          f"AUC-PR={best_row['auc_pr']:.4f}")
    print(f"Beats Method 2 AUC-PR: {'YES' if best_row['auc_pr'] > M2_BEST_AUC_PR else 'NO'}")
    print(f"Best formula: {best_row['formula']}")

    # ── Part E: Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    prevalence = y_test.mean()

    # PR curves for top 5
    n_top = min(5, len(evaluated))
    fig, axes = plt.subplots(1, n_top, figsize=(4 * n_top, 4), sharey=True)
    if n_top == 1:
        axes = [axes]

    for i, (prog, row) in enumerate(evaluated[:n_top]):
        ax = axes[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = np.asarray(prog.execute(X_test), dtype=float)
        bad = ~np.isfinite(score)
        if bad.any():
            score[bad] = np.nanmedian(score[~bad]) if (~bad).any() else 0.0
        if roc_auc_score(y_test, score) < 0.5:
            score = -score
        prec_c, rec_c, _ = precision_recall_curve(y_test, score)
        ax.plot(rec_c, prec_c, color="#4C72B0", lw=1.5, label=f"AUC-PR={row['auc_pr']:.4f}")
        ax.axhline(prevalence, color="gray", linestyle=":", lw=1, label=f"Prevalence ({prevalence:.4f})")
        ax.set_xlim(0, 1)
        ax.set_title(f"#{i+1}", fontsize=10)
        ax.set_xlabel("Recall", fontsize=8)
        if i == 0:
            ax.set_ylabel("Precision", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    fig.suptitle("Method 3 (GP): PR Curves — Top 5 Programs", fontsize=12)
    plt.tight_layout()
    pr_path = OUT_DIR / "top_pr_curves.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"Saved {pr_path}")

    # AUC-PR histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(results_df["auc_pr"], bins=min(20, len(results_df)),
            color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(LR_AUC_PR,      color="red",    linestyle="--", lw=1.5,
               label=f"LR baseline ({LR_AUC_PR})")
    ax.axvline(M2_BEST_AUC_PR, color="orange", linestyle="--", lw=1.5,
               label=f"Method 2 best ({M2_BEST_AUC_PR})")
    ax.axvline(best_row["auc_pr"], color="green", linestyle="--", lw=1.5,
               label=f"GP best ({best_row['auc_pr']:.4f})")
    ax.set_xlabel("AUC-PR")
    ax.set_ylabel("Count")
    ax.set_title(f"Method 3 (GP): AUC-PR Distribution ({len(results_df)} programs)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = OUT_DIR / "auc_pr_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved {hist_path}")

    # Method comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    method_labels = ["LR baseline", "Method 2\n(random)", "Method 3\n(GP)"]
    pr_vals  = [LR_AUC_PR,  M2_BEST_AUC_PR,   best_row["auc_pr"]]
    roc_vals = [LR_AUC_ROC, results_df["auc_roc"].max(), best_row["auc_roc"]]
    colors   = ["#999999", "#4C72B0", "#DD8452"]

    for ax, vals, ylabel in zip(axes, [pr_vals, roc_vals], ["AUC-PR", "AUC-ROC"]):
        bars = ax.bar(method_labels, vals, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} Comparison")
        ax.set_ylim(0, max(vals) * 1.30)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.4f}", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Method 3 (GP) vs Baselines", fontsize=12)
    plt.tight_layout()
    comp_path = OUT_DIR / "comparison_chart.png"
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"Saved {comp_path}")

    print("\nMethod 3 (GP) complete.")


if __name__ == "__main__":
    main()
