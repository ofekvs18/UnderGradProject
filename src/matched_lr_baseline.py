"""
Compute matched-LR baselines: for each disease and winning method formula (M2, M3, M4),
train a LogisticRegression using exactly the same features that appear in that formula,
then evaluate on the frozen test set.

Also evaluates the actual formula on the test set and computes 95% bootstrap CIs for both
the formula and the matched LR, so the comparison is uncertainty-aware.

Output: results/matched_lr_baseline.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import CBC_FEATURE_LIST, bootstrap_ci, get_scores, get_splits, load_data_for

DISEASES = ["ra", "crhn", "t1d", "t2d", "psr", "lup"]
RESULTS_DIR = Path("results")


def extract_features(formula: str) -> list[str]:
    return [f for f in CBC_FEATURE_LIST if re.search(rf"\b{re.escape(f)}\b", formula)]


def compute_lr(disease: str, features: list[str]):
    """
    Fit LR on the matched feature subset.
    Returns (metrics, proba, y_te, test_df) or ({}, None, None, None) when skipped.
    """
    df, _ = load_data_for(disease)
    train_df, test_df = get_splits(df)
    prevalence = float(df["is_case"].mean())

    tr = train_df[features + ["is_case"]].dropna()
    te = test_df[features + ["is_case"]].dropna()
    y_tr, y_te = tr["is_case"].values, te["is_case"].values

    if y_tr.sum() < 5 or y_te.sum() < 5:
        return {}, None, None, None

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(tr[features].values, y_tr)
    proba = clf.predict_proba(te[features].values)[:, 1]

    auc_pr = float(average_precision_score(y_te, proba))
    auc_roc = float(roc_auc_score(y_te, proba))

    terms = " + ".join(f"{c:.4g}*{f}" for c, f in zip(clf.coef_[0], features))
    lr_formula = f"{terms} + {clf.intercept_[0]:.4g}"

    metrics = {
        "auc_pr": round(auc_pr, 4),
        "auc_roc": round(auc_roc, 4),
        "lift": round(auc_pr / prevalence, 3) if prevalence > 0 else float("nan"),
        "prevalence": round(prevalence, 4),
        "lr_formula": lr_formula,
    }
    return metrics, proba, y_te, test_df


def main():
    comp_path = RESULTS_DIR / "methods_comparison.csv"
    if not comp_path.exists():
        print(f"ERROR: {comp_path} not found — run compare_methods.py first.")
        sys.exit(1)

    comp = pd.read_csv(comp_path)
    rows = []

    for disease in DISEASES:
        dr = comp[comp["Disease"] == disease]
        if dr.empty:
            print(f"  {disease}: not in methods_comparison.csv, skipping")
            continue
        r = dr.iloc[0]

        for method in ("M2", "M3", "M4", "M5"):
            col = f"{method}_Best_Formula"
            if col not in r.index or pd.isna(r[col]):
                continue
            formula = str(r[col])
            features = extract_features(formula)
            if not features:
                print(f"  {disease} {method}: no features parsed, skipping")
                continue

            print(f"  {disease} {method}: n={len(features)} features={features}", flush=True)
            metrics, proba, y_te, test_df = compute_lr(disease, features)
            if not metrics:
                print(f"    -> skipped (too few positives)")
                continue
            print(f"    LR: AUC-PR={metrics['auc_pr']:.4f}  lift={metrics['lift']:.3f}×")

            # Bootstrap CI for LR
            lr_ci = bootstrap_ci(y_te, proba)
            lr_auc_pr_lo = round(lr_ci[0], 4) if lr_ci else None
            lr_auc_pr_hi = round(lr_ci[1], 4) if lr_ci else None
            lr_auc_roc_lo = round(lr_ci[2], 4) if lr_ci else None
            lr_auc_roc_hi = round(lr_ci[3], 4) if lr_ci else None

            # Formula point estimates + CI (use full CBC feature list for formula eval)
            formula_auc_pr = formula_auc_roc = None
            form_auc_pr_lo = form_auc_pr_hi = form_auc_roc_lo = form_auc_roc_hi = None
            formula_scores, y_f = get_scores(formula, test_df, CBC_FEATURE_LIST)
            if formula_scores is not None:
                try:
                    formula_auc_pr = round(float(average_precision_score(y_f, formula_scores)), 4)
                    formula_auc_roc = round(float(roc_auc_score(y_f, formula_scores)), 4)
                except Exception:
                    pass
                form_ci = bootstrap_ci(y_f, formula_scores)
                if form_ci:
                    form_auc_pr_lo = round(form_ci[0], 4)
                    form_auc_pr_hi = round(form_ci[1], 4)
                    form_auc_roc_lo = round(form_ci[2], 4)
                    form_auc_roc_hi = round(form_ci[3], 4)
                print(f"    Formula: AUC-PR={formula_auc_pr}  CI=[{form_auc_pr_lo}, {form_auc_pr_hi}]")
            else:
                print(f"    Formula: evaluation failed")

            rows.append({
                "disease": disease,
                "method": method,
                "formula": formula,
                "features": ",".join(features),
                "n_features": len(features),
                "formula_auc_pr": formula_auc_pr,
                "formula_auc_roc": formula_auc_roc,
                "formula_auc_pr_lo": form_auc_pr_lo,
                "formula_auc_pr_hi": form_auc_pr_hi,
                "formula_auc_roc_lo": form_auc_roc_lo,
                "formula_auc_roc_hi": form_auc_roc_hi,
                "lr_formula": metrics["lr_formula"],
                "auc_pr": metrics["auc_pr"],
                "auc_roc": metrics["auc_roc"],
                "lift": metrics["lift"],
                "prevalence": metrics["prevalence"],
                "lr_auc_pr_lo": lr_auc_pr_lo,
                "lr_auc_pr_hi": lr_auc_pr_hi,
                "lr_auc_roc_lo": lr_auc_roc_lo,
                "lr_auc_roc_hi": lr_auc_roc_hi,
            })

    out = RESULTS_DIR / "matched_lr_baseline.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
