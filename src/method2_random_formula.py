"""
Issue #10 / Method 2: Random formula generation.

Generates 10,000 random mathematical formulas combining 2-4 CBC features
and evaluates them on the frozen test set. Serves as a baseline for more
sophisticated search methods (genetic programming, LLM-guided).

Research question: can random CBC combinations beat single-feature thresholds
(AUC-PR=0.014) or all-features logistic regression (AUC-PR=0.017)?
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
)

sys.path.insert(0, "src")
from utils import (
    load_data, get_splits, compute_binary_metrics, find_youden_threshold,
    precision_at_recall_levels, ensure_dir, RESULTS_DIR,
)

OUT_DIR = RESULTS_DIR / "method2_random"
ensure_dir(OUT_DIR)

N_FORMULAS  = 10_000
SEED        = 42
BAD_FRAC    = 0.10   # skip formula if > 10% of rows produce NaN/inf
BASELINE_AUC_PR = 0.0170

# ── Part A: Load data ─────────────────────────────────────────────────────────
print("Loading data...")
df, features = load_data()
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


def random_formula(rng, features=features):
    """
    Generate a random formula string combining 2-4 features with random
    unary transforms and binary operators. Division is made safe by adding 1e-6.
    """
    n = rng.randint(2, 5)  # 2, 3, or 4 features
    chosen = list(rng.choice(features, size=n, replace=False))
    terms = [_random_term(f, rng) for f in chosen]

    result = terms[0]
    for term in terms[1:]:
        op = rng.choice(["+", "-", "*", "/"])
        if op == "/":
            result = f"({result}) / (abs({term})+1e-6)"
        else:
            result = f"({result}) {op} ({term})"
    return result


def generate_formulas(n=N_FORMULAS, seed=SEED):
    """Generate n unique random formula strings."""
    rng = np.random.RandomState(seed)
    seen = set()
    formulas = []
    attempts = 0
    while len(formulas) < n:
        f = random_formula(rng)
        attempts += 1
        if f not in seen:
            seen.add(f)
            formulas.append(f)
        if attempts > n * 10:
            break  # safety — shouldn't happen
    print(f"Generated {len(formulas):,} unique formulas ({attempts:,} attempts)\n")
    return formulas


# ── Part C: Formula evaluation ────────────────────────────────────────────────

def _eval_formula_on_df(formula, df):
    """
    Safely evaluate a formula string against a DataFrame.
    Returns a numpy array of scores, or None if too many bad values.
    """
    local = {f: df[f].values.astype(float) for f in features}
    local["sqrt"] = np.sqrt
    local["log"]  = np.log1p
    local["abs"]  = np.abs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores = eval(formula, {"__builtins__": {}}, local)  # noqa: S307
        except Exception:
            return None
    scores = np.asarray(scores, dtype=float)
    bad = ~np.isfinite(scores)
    if bad.mean() > BAD_FRAC:
        return None
    # Replace residual NaN/inf with median so AUC can be computed
    if bad.any():
        scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0
    return scores


def evaluate_formula(formula, train_df, test_df):
    """
    Evaluate one formula. Returns a metrics dict or None if invalid.
    Threshold is chosen via Youden's index on TRAIN, applied to TEST.
    """
    # Drop rows with NaN in any feature used (safe: eval handles it, but
    # we still need y labels aligned)
    tr_clean = train_df[features + ["is_case"]].dropna()
    te_clean = test_df[features + ["is_case"]].dropna()

    score_tr = _eval_formula_on_df(formula, tr_clean)
    score_te = _eval_formula_on_df(formula, te_clean)
    if score_tr is None or score_te is None:
        return None

    y_tr = tr_clean["is_case"].values
    y_te = te_clean["is_case"].values

    if y_tr.sum() < 5 or y_te.sum() < 5:
        return None

    # AUC metrics (threshold-independent)
    try:
        auc_roc = float(roc_auc_score(y_te, score_te))
        auc_pr  = float(average_precision_score(y_te, score_te))
    except Exception:
        return None

    # If AUC-ROC < 0.5, flip scores (formula is inversely predictive)
    if auc_roc < 0.5:
        score_tr = -score_tr
        score_te = -score_te
        auc_roc  = 1.0 - auc_roc
        auc_pr   = float(average_precision_score(y_te, score_te))

    # Youden threshold on train → test
    threshold, _, _ = find_youden_threshold(y_tr, score_tr)
    preds = (score_te >= threshold).astype(int)
    m = compute_binary_metrics(y_te, preds)

    # Precision@recall levels
    par = precision_at_recall_levels(score_tr, y_tr, score_te, y_te)

    return {
        "formula":               formula,
        "auc_roc":               round(auc_roc, 4),
        "auc_pr":                round(auc_pr, 4),
        "precision":             round(m["precision"], 4),
        "recall":                round(m["recall"], 4),
        "f1":                    round(m["f1"], 4),
        "f2":                    round(m["f2"], 4),
        "precision_at_recall_25": par[0.25][0],
        "precision_at_recall_50": par[0.50][0],
        "precision_at_recall_75": par[0.75][0],
    }


# ── Part D: Run experiment ────────────────────────────────────────────────────
print("=== Generating formulas ===")
formulas = generate_formulas()

print("=== Evaluating formulas ===")
results = []
skipped = 0
for i, formula in enumerate(formulas):
    if (i + 1) % 1000 == 0:
        print(f"  {i+1:,}/{len(formulas):,}  valid={len(results):,}  skipped={skipped}")
    row = evaluate_formula(formula, train_df, test_df)
    if row is None:
        skipped += 1
    else:
        results.append(row)

print(f"\nDone: {len(results):,} valid formulas, {skipped:,} skipped\n")

results_df = pd.DataFrame(results).sort_values("auc_pr", ascending=False).reset_index(drop=True)

# Save all results
all_path = OUT_DIR / "all_formulas.csv"
results_df.to_csv(all_path, index=False)
print(f"Saved {all_path}")

# Save top 10 by AUC-PR
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

print("\nMethod 2 complete.")
