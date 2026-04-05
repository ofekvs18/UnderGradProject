"""
Wrapper for method3_gp that runs with LARGE_CONFIG (pop=500, gen=100).
Place this in src/ alongside method3_gp.py.

Usage: python -u src/method3_gp_large.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness

sys.path.insert(0, "src")
from utils import (
    load_data, get_splits, compute_binary_metrics, find_youden_threshold,
    precision_at_recall_levels, ensure_dir, RESULTS_DIR,
)
# Re-use evaluate_program and fitness from the original
from method3_gp import (
    combined_auc_fitness, evaluate_program,
    M2_BEST_AUC_PR, LR_AUC_PR, LR_AUC_ROC, SEED,
    features, X_train, y_train, X_test, y_test,
)

# ── Override: use LARGE config ────────────────────────────────────────────────
SIZE_CONFIG = dict(population_size=500, generations=100)
OUT_DIR = RESULTS_DIR / "method3_gp_large"
ensure_dir(OUT_DIR)

# Best config from Phase 1 small runs (attempt 0 baseline)
PARSIMONY   = 0.005
FUNC_SET    = ["add", "sub", "mul", "div", "sqrt", "log", "abs"]

print("=" * 80)
print("METHOD 3 GP — LARGE CONFIG")
print(f"  Population: {SIZE_CONFIG['population_size']}")
print(f"  Generations: {SIZE_CONFIG['generations']}")
print(f"  Parsimony: {PARSIMONY}")
print(f"  Functions: {FUNC_SET}")
print(f"  n_jobs: -1 (all available CPUs)")
print("=" * 80)

# ── Fit ───────────────────────────────────────────────────────────────────────
gp = SymbolicTransformer(
    **SIZE_CONFIG,
    hall_of_fame=50,
    n_components=50,
    feature_names=features,
    function_set=FUNC_SET,
    metric=combined_auc_fitness,
    parsimony_coefficient=PARSIMONY,
    random_state=SEED,
    n_jobs=-1,          # use ALL allocated CPUs
    verbose=1,
)

print("\nFitting GP (this will take a while)...")
gp.fit(X_train, y_train)
print("Fitting complete.\n")

# ── Evaluate hall of fame ─────────────────────────────────────────────────────
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
print(f"Evaluated {len(evaluated)} valid programs ({skipped} skipped)")

if not evaluated:
    print("ERROR: No valid programs found. Exiting.")
    sys.exit(1)

best_prog, best_row = evaluated[0]
results_df = pd.DataFrame([row for _, row in evaluated])

# ── Save results ──────────────────────────────────────────────────────────────
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

# ── Validation ────────────────────────────────────────────────────────────────
print(f"\n=== Validation ===")
auc_roc_ok = 0.52 < best_row["auc_roc"] < 0.95
print(f"Best AUC-ROC: {best_row['auc_roc']:.4f}  (must be 0.52-0.95): "
      f"{'PASS' if auc_roc_ok else 'FAIL'}")

# ── Compare vs baselines ─────────────────────────────────────────────────────
print(f"\n=== GP LARGE vs baselines ===")
print(f"Baseline (all-features LR)  : AUC-ROC={LR_AUC_ROC:.4f}  AUC-PR={LR_AUC_PR:.4f}")
print(f"Method 2 best random formula: AUC-PR={M2_BEST_AUC_PR:.4f}")
print(f"Method 3 GP LARGE best      : AUC-ROC={best_row['auc_roc']:.4f}  "
      f"AUC-PR={best_row['auc_pr']:.4f}")
print(f"Beats Method 2 AUC-PR: {'YES' if best_row['auc_pr'] > M2_BEST_AUC_PR else 'NO'}")
print(f"Best formula: {best_row['formula']}")

# ── Plots ─────────────────────────────────────────────────────────────────────
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

fig.suptitle("Method 3 (GP LARGE): PR Curves — Top 5 Programs", fontsize=12)
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
           label=f"GP LARGE best ({best_row['auc_pr']:.4f})")
ax.set_xlabel("AUC-PR")
ax.set_ylabel("Count")
ax.set_title(f"Method 3 GP LARGE: AUC-PR Distribution ({len(results_df)} programs)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
hist_path = OUT_DIR / "auc_pr_histogram.png"
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"Saved {hist_path}")

# Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
method_labels = ["LR baseline", "Method 2\n(random)", "Method 3\n(GP LARGE)"]
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

fig.suptitle("Method 3 (GP LARGE) vs Baselines", fontsize=12)
plt.tight_layout()
comp_path = OUT_DIR / "comparison_chart.png"
plt.savefig(comp_path, dpi=150)
plt.close()
print(f"Saved {comp_path}")

print("\nMethod 3 GP LARGE complete.")
