"""
nhanes_evaluate.py — Evaluate method formulas on the NHANES dataset.

Loads master summary CSVs from all four methods and evaluates each formula
on the NHANES modeling data produced by nhanes_data.py. Uses the NHANES
train split to fit the Youden threshold and the test split for all metrics,
matching the MIMIC-IV evaluation protocol.

Survey weights are not applied; each participant is treated equally (mirrors
the MIMIC-IV pipeline for comparability).

Formula format handling:
  - Method 1 (threshold): "rbc < 4.0"  →  evaluated as a 0/1 score
  - Method 2 (random):    "(rbc**2) / (abs(hct)+1e-6)"  →  standard eval
  - Method 3 (GP):        "div(add(rbc,hct), sqrt(wbc))"  →  prefix eval
  - Method 4 (LLM):       "(hct - 40) * (plt - 100)"  →  standard eval

Usage:
    python src/nhanes_evaluate.py --disease ra          # all methods
    python src/nhanes_evaluate.py --disease ra --method m2
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "src")
from utils import (
    load_disease_config, ensure_dir, DATA_DIR, RESULTS_DIR,
    get_splits, compute_binary_metrics, find_youden_threshold,
    precision_at_recall_levels,
)

# ── Master summary specs: path + which column(s) hold the formula ─────────────
METHOD_SPECS = {
    "m1": {
        "path":     RESULTS_DIR / "method1_threshold" / "master_m1_summary.csv",
        "formulas": [("Best_Lit_Formula", "lit"), ("Best_DD_Formula", "dd")],
        "label":    "Method 1 (Threshold)",
    },
    "m2": {
        "path":     RESULTS_DIR / "method2_random" / "master_m2_summary.csv",
        "formulas": [("Best_Random_Formula", "random")],
        "label":    "Method 2 (Random)",
    },
    "m3": {
        "path":     RESULTS_DIR / "method3_gp" / "master_gp_summary.csv",
        "formulas": [("Best_GP_Formula", "gp")],
        "label":    "Method 3 (GP)",
    },
    "m4": {
        "path":     RESULTS_DIR / "method4_llm" / "method4_master_summary.csv",
        "formulas": [("Best_LLM_Formula", "llm")],
        "label":    "Method 4 (LLM)",
    },
}


# ── Formula evaluation ────────────────────────────────────────────────────────

def eval_formula_scores_extended(formula, df, features, bad_frac=0.10):
    """
    Evaluate a formula string against a DataFrame. Supports:
      - Standard infix (methods 2, 4): "(rbc**2) / (abs(hct)+1e-6)"
      - GP prefix notation (method 3): "div(add(rbc, hct), sqrt(wbc))"
      - Threshold comparisons (method 1): "rbc < 4.0"

    Returns a float numpy array of scores, or None if invalid/too many NaNs.
    """
    local = {f: df[f].values.astype(float) for f in features if f in df.columns}
    local["sqrt"] = lambda x: np.sqrt(np.abs(x))
    local["log"]  = lambda x: np.log1p(np.abs(x))
    local["abs"]  = np.abs
    local["neg"]  = lambda x: -x
    local["div"]  = lambda x, y: np.where(np.abs(y) > 1e-6, x / y, 0.0)
    local["mul"]  = lambda x, y: x * y
    local["add"]  = lambda x, y: x + y
    local["sub"]  = lambda x, y: x - y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores = eval(formula, {"__builtins__": {}}, local)  # noqa: S307
        except Exception:
            return None

    scores = np.asarray(scores, dtype=float)
    bad = ~np.isfinite(scores)
    if bad.mean() > bad_frac:
        return None
    if bad.any():
        scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0
    return scores


def evaluate_formula_on_nhanes(formula, train_df, test_df, features):
    """
    Evaluate one formula on the NHANES train/test split.
    Threshold is fitted via Youden's index on train, applied to test.
    Returns a metrics dict or None if evaluation fails.
    """
    tr = train_df[features + ["is_case"]].dropna()
    te = test_df[features + ["is_case"]].dropna()

    score_tr = eval_formula_scores_extended(formula, tr, features)
    score_te = eval_formula_scores_extended(formula, te, features)
    if score_tr is None or score_te is None:
        return None

    y_tr = tr["is_case"].values
    y_te = te["is_case"].values

    if y_tr.sum() < 3 or y_te.sum() < 3:
        return None

    try:
        auc_roc = float(roc_auc_score(y_te, score_te))
        auc_pr  = float(average_precision_score(y_te, score_te))
    except Exception:
        return None

    if auc_roc < 0.5:
        score_tr = -score_tr
        score_te = -score_te
        auc_roc  = 1.0 - auc_roc
        auc_pr   = float(average_precision_score(y_te, score_te))

    threshold, _, _ = find_youden_threshold(y_tr, score_tr)
    preds = (score_te >= threshold).astype(int)
    m   = compute_binary_metrics(y_te, preds)
    par = precision_at_recall_levels(score_tr, y_tr, score_te, y_te)

    return {
        "formula":                formula,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate method formulas on the NHANES external validation set"
    )
    parser.add_argument("--disease", default="ra", help="Disease slug (default: ra)")
    parser.add_argument("--method", default="all",
                        help="Method key to evaluate: m1, m2, m3, m4, or all (default)")
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    data_path = DATA_DIR / f"{args.disease}_nhanes_data.csv"
    if not data_path.exists():
        sys.exit(
            f"ERROR: {data_path} not found.\n"
            f"Run: python src/nhanes_data.py --nhanes-dir <path> --disease {args.disease}"
        )

    out_dir = RESULTS_DIR / "nhanes"
    ensure_dir(out_dir)

    print("=" * 70)
    print(f"NHANES Evaluation — {disease.full_name} ({disease.name})")
    print("=" * 70)

    df = pd.read_csv(data_path)
    features = [c for c in df.columns if c not in {"subject_id", "is_case", "split"}]
    train_df, test_df = get_splits(df)
    prevalence = df["is_case"].mean()

    print(f"NHANES data: {len(df):,} participants | "
          f"train={len(train_df):,} | test={len(test_df):,} | "
          f"prevalence={prevalence:.2%}")
    print(f"Features: {features}\n")

    methods_to_run = list(METHOD_SPECS.keys()) if args.method == "all" else [args.method]
    if args.method != "all" and args.method not in METHOD_SPECS:
        sys.exit(f"ERROR: Unknown method '{args.method}'. Choose from: {list(METHOD_SPECS.keys())}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results = []

    for method_key in methods_to_run:
        spec = METHOD_SPECS[method_key]
        if not spec["path"].exists():
            print(f"[SKIP] {spec['label']}: master summary not found at {spec['path']}")
            continue

        summary_df = pd.read_csv(spec["path"])
        disease_rows = summary_df[summary_df["Disease"] == args.disease]
        if disease_rows.empty:
            print(f"[SKIP] {spec['label']}: no rows for disease '{args.disease}'")
            continue

        print(f"=== {spec['label']} ({len(disease_rows)} row(s)) ===")
        method_results = []

        for formula_col, variant in spec["formulas"]:
            if formula_col not in disease_rows.columns:
                print(f"  [SKIP] Column '{formula_col}' not in summary")
                continue

            for _, row in disease_rows.iterrows():
                formula = str(row.get(formula_col, ""))
                if not formula or formula.lower() == "nan":
                    continue

                result = evaluate_formula_on_nhanes(formula, train_df, test_df, features)
                if result is None:
                    print(f"  [{variant}] FAIL: {formula[:70]}")
                    continue

                out_row = {
                    "Timestamp": timestamp,
                    "Disease":   args.disease,
                    "Method":    method_key,
                    "Variant":   variant,
                    **result,
                }
                for extra in ("Config_Used", "Winning_Strategy", "Winning_Temp",
                              "Best_Lit_Feature", "Best_DD_Feature"):
                    if extra in disease_rows.columns:
                        out_row[extra] = row.get(extra, "")

                method_results.append(out_row)
                print(f"  [{variant}] AUC-PR={result['auc_pr']:.4f}  "
                      f"AUC-ROC={result['auc_roc']:.4f}  "
                      f"F2={result['f2']:.4f}  | {formula[:65]}")

        all_results.extend(method_results)

        if method_results:
            method_df = pd.DataFrame(method_results)
            out_path = out_dir / f"{args.disease}_{method_key}_eval.csv"
            if out_path.exists():
                method_df = pd.concat([pd.read_csv(out_path), method_df], ignore_index=True)
            method_df.to_csv(out_path, index=False)
            print(f"  Saved {len(method_results)} results → {out_path}\n")
        else:
            print()

    if not all_results:
        print("No formulas evaluated. Check that master summaries exist and "
              "contain this disease.")
        return

    combined_df = pd.DataFrame(all_results)
    combined_path = out_dir / f"{args.disease}_all_methods_eval.csv"
    if combined_path.exists():
        combined_df = pd.concat([pd.read_csv(combined_path), combined_df], ignore_index=True)
    combined_df.to_csv(combined_path, index=False)

    print("=" * 70)
    print(f"Combined results ({len(all_results)} rows) → {combined_path}")

    best = combined_df.sort_values("auc_pr", ascending=False).iloc[0]
    print(f"Best on NHANES: [{best['Method']}/{best['Variant']}]  "
          f"AUC-PR={best['auc_pr']:.4f}  AUC-ROC={best['auc_roc']:.4f}")
    print(f"  Formula: {str(best['formula'])[:80]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
