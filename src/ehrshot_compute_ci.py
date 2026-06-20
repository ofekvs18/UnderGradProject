"""
ehrshot_compute_ci.py — Bootstrap 95% CIs for EHRSHOT external validation results.

For each method, finds the best formula (by AUC-PR) from the EHRSHOT evaluation
CSVs and runs stratified percentile bootstrap on the EHRSHOT test split.

Stratified bootstrap (resample within positives and negatives separately) is
used to ensure every bootstrap sample has at least one positive — important for
diseases with small EHRSHOT cohorts.

Writes:
  results/ehrshot/{disease}_ci_data.csv  — one row per method, for forest plot

Usage:
    python src/ehrshot_compute_ci.py --disease ra
    python src/ehrshot_compute_ci.py --disease ra --n-bootstrap 1000
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from utils import DATA_DIR, RESULTS_DIR, ensure_dir, load_disease_config

N_BOOTSTRAP = 500
CI_LO = 2.5
CI_HI = 97.5
SEED = 42


# ── Formula evaluation ────────────────────────────────────────────────────────

def eval_formula_scores(formula, df, features, bad_frac=0.10):
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


def get_scores(formula, test_df, features):
    """Return (scores, y_true) with sign flip if needed, or (None, None)."""
    te = test_df[features + ["is_case"]].dropna()
    y = te["is_case"].values
    scores = eval_formula_scores(formula, te, features)
    if scores is None:
        return None, None
    try:
        auc_roc = float(roc_auc_score(y, scores))
        if auc_roc < 0.5:
            scores = -scores
    except Exception:
        pass
    return scores, y


# ── Stratified bootstrap ───────────────────────────────────────────────────────

def bootstrap_ci_stratified(y_true, scores, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Percentile bootstrap 95% CI for AUC-PR and AUC-ROC with stratification.

    Resamples positives and negatives independently so every bootstrap sample
    has the same class counts as the original.

    Returns (pr_lo, pr_hi, roc_lo, roc_hi) or None if too few valid samples.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return None

    auc_prs, auc_rocs = [], []

    for _ in range(n_bootstrap):
        bi_pos = rng.integers(0, len(pos_idx), size=len(pos_idx))
        bi_neg = rng.integers(0, len(neg_idx), size=len(neg_idx))
        idx = np.concatenate([pos_idx[bi_pos], neg_idx[bi_neg]])
        yb, sb = y_true[idx], scores[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc_prs.append(float(average_precision_score(yb, sb)))
                auc_rocs.append(float(roc_auc_score(yb, sb)))
        except Exception:
            pass

    if len(auc_prs) < 20:
        return None

    return (
        float(np.percentile(auc_prs, CI_LO)),
        float(np.percentile(auc_prs, CI_HI)),
        float(np.percentile(auc_rocs, CI_LO)),
        float(np.percentile(auc_rocs, CI_HI)),
    )


# ── Best formula per method ────────────────────────────────────────────────────

METHOD_LABEL = {
    "m1": "M1: Threshold",
    "m2": "M2: Random Search",
    "m3": "M3: Genetic Programming",
    "m4": "M4: LLM",
    "m5": "M5: Seeded GP",
}


def get_best_formula(disease_name, method_key):
    """
    Read the per-method EHRSHOT eval CSV and return the single row with the
    highest AUC-PR.  Returns None if the file is missing or empty.
    """
    path = RESULTS_DIR / "ehrshot" / f"{disease_name}_{method_key}_eval.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["Disease"] == disease_name] if "Disease" in df.columns else df
    if df.empty:
        return None
    row = df.sort_values("auc_pr", ascending=False).iloc[0]
    return {
        "method":  method_key,
        "label":   METHOD_LABEL.get(method_key, method_key),
        "formula": str(row["formula"]),
        "auc_pr":  float(row["auc_pr"]),
        "auc_roc": float(row["auc_roc"]),
        "variant": str(row.get("Variant", "")),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stratified bootstrap CIs for EHRSHOT external validation"
    )
    parser.add_argument("--disease", default="ra")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    disease = load_disease_config(args.disease)

    data_path = DATA_DIR / f"{args.disease}_ehrshot_data.csv"
    if not data_path.exists():
        sys.exit(
            f"ERROR: {data_path} not found.\n"
            f"Run: python src/ehrshot_bq_data.py --disease {args.disease}"
        )

    df = pd.read_csv(data_path)
    features = [c for c in df.columns if c not in {"subject_id", "is_case", "split"}]

    from utils import get_splits
    _, test_df = get_splits(df)

    n_pos = int(test_df["is_case"].sum())
    n_tot = len(test_df)
    print(f"Disease : {disease.name}")
    print(f"EHRSHOT test set: {n_tot:,} rows, {n_pos} positives  "
          f"(prevalence={n_pos/n_tot:.3%})")
    print(f"Bootstrap iterations: {args.n_bootstrap}  [stratified]\n")

    ci_rows = []

    for method_key in ("m1", "m2", "m3", "m4", "m5"):
        best = get_best_formula(disease.name, method_key)
        if best is None:
            print(f"[{method_key}] no EHRSHOT eval found — skipped")
            continue

        formula = best["formula"]
        print(f"[{method_key}] variant={best['variant']}  "
              f"stored AUC-PR={best['auc_pr']:.4f}")
        print(f"         formula: {formula[:80]}")

        scores, y = get_scores(formula, test_df, features)
        if scores is None:
            print(f"  [WARN] formula evaluation failed — skipped\n")
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eval_auc_pr  = round(float(average_precision_score(y, scores)), 4)
                eval_auc_roc = round(float(roc_auc_score(y, scores)), 4)
        except Exception:
            eval_auc_pr  = best["auc_pr"]
            eval_auc_roc = best["auc_roc"]

        ci = bootstrap_ci_stratified(y, scores, n_bootstrap=args.n_bootstrap)
        row = {
            "method":  method_key,
            "label":   best["label"],
            "formula": formula,
            "auc_pr":  eval_auc_pr,
            "auc_roc": eval_auc_roc,
            "variant": best["variant"],
        }
        if ci:
            row.update({
                "AUC_PR_CI_Low":   round(ci[0], 4),
                "AUC_PR_CI_High":  round(ci[1], 4),
                "AUC_ROC_CI_Low":  round(ci[2], 4),
                "AUC_ROC_CI_High": round(ci[3], 4),
            })
            print(f"  AUC-PR={eval_auc_pr:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]  "
                  f"AUC-ROC={eval_auc_roc:.4f}  CI=[{ci[2]:.4f}, {ci[3]:.4f}]")
        else:
            print(f"  [WARN] bootstrap failed — too few valid samples; CI omitted")
        print()

        ci_rows.append(row)

    if not ci_rows:
        print("No results produced. Check that EHRSHOT eval CSVs exist for this disease.")
        return

    out_dir = RESULTS_DIR / "ehrshot"
    ensure_dir(out_dir)
    out_path = out_dir / f"{disease.name}_ci_data.csv"
    pd.DataFrame(ci_rows).to_csv(out_path, index=False)

    print(f"Wrote CI data ({len(ci_rows)} rows): {out_path}")
    print(pd.DataFrame(ci_rows)[
        ["method", "auc_pr", "AUC_PR_CI_Low", "AUC_PR_CI_High"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
