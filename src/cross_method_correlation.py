"""
cross_method_correlation.py — Cross-method score vector correlation analysis.

For a given disease, computes pairwise Pearson correlations between the score
vectors that each method's winning formula produces on the test set. Answers:
did the four methods find the same underlying biomarker signal, or genuinely
different predictive structures?

Usage:
    python src/cross_method_correlation.py --disease ra
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from utils import (
    load_data_for, get_splits, eval_formula_scores, ensure_dir, RESULTS_DIR,
)

OUT_DIR        = RESULTS_DIR / "cross_method"
MASTER_SUMMARY = OUT_DIR / "master_correlation_summary.csv"
METHODS        = ["M1", "M2", "M3", "M4", "Baseline"]

# ── gplearn prefix parser ─────────────────────────────────────────────────────

_BINARY = {"add", "sub", "mul", "div"}
_UNARY  = {"sqrt", "log", "abs", "neg"}


def _tokenize_gp(expr):
    return re.findall(
        r'[a-zA-Z_]\w*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[(),]',
        expr.replace(" ", ""),
    )


def _eval_node(tokens, idx, arrays):
    token = tokens[idx[0]]
    idx[0] += 1

    if token in _BINARY:
        assert tokens[idx[0]] == "("
        idx[0] += 1
        a = _eval_node(tokens, idx, arrays)
        assert tokens[idx[0]] == ","
        idx[0] += 1
        b = _eval_node(tokens, idx, arrays)
        assert tokens[idx[0]] == ")"
        idx[0] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            if token == "add": return a + b
            if token == "sub": return a - b
            if token == "mul": return a * b
            if token == "div": return np.where(np.abs(b) > 1e-9, a / b, 0.0)

    if token in _UNARY:
        assert tokens[idx[0]] == "("
        idx[0] += 1
        a = _eval_node(tokens, idx, arrays)
        assert tokens[idx[0]] == ")"
        idx[0] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            if token == "sqrt": return np.sqrt(np.abs(a))
            if token == "log":  return np.log1p(np.abs(a))
            if token == "abs":  return np.abs(a)
            if token == "neg":  return -a

    if token in arrays:
        return arrays[token].copy()

    n = next(iter(arrays.values())).shape[0]
    return np.full(n, float(token))


def eval_gp_formula(formula, df, features):
    """Parse a gplearn prefix-notation formula and return a score array, or None."""
    arrays = {f: df[f].values.astype(float) for f in features}
    try:
        tokens = _tokenize_gp(formula)
        idx    = [0]
        scores = _eval_node(tokens, idx, arrays)
        scores = np.asarray(scores, dtype=float)
    except Exception as e:
        print(f"    [WARN] gplearn parse failed: {e}")
        return None
    bad = ~np.isfinite(scores)
    if bad.mean() > 0.10:
        return None
    if bad.any():
        scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0
    return scores


# ── Winning formula loaders ───────────────────────────────────────────────────

def _get_m1_formula(disease):
    path = RESULTS_DIR / "method1_threshold" / "master_m1_summary.csv"
    if not path.exists():
        return None, "M1 master summary not found"
    df = pd.read_csv(path)
    row = df[df["Disease"] == disease]
    if row.empty:
        return None, f"M1: no row for '{disease}'"
    feat = row.iloc[0]["Best_DD_Feature"]
    if pd.isna(feat) or not str(feat).strip():
        return None, "M1: Best_DD_Feature is empty"
    return str(feat).strip(), f"M1 DD feature: {feat}"


def _get_m2_formula(disease):
    path = RESULTS_DIR / "method2_random" / disease / "top_formulas.csv"
    if not path.exists():
        return None, f"M2 top_formulas.csv not found"
    df = pd.read_csv(path)
    if df.empty or "formula" not in df.columns:
        return None, "M2: top_formulas.csv empty or missing column"
    return str(df.iloc[0]["formula"]).strip(), "M2 top formula"


def _get_m3_formula(disease):
    path = RESULTS_DIR / "method3_gp" / "master_gp_summary.csv"
    if not path.exists():
        return None, "M3 master_gp_summary.csv not found"
    df   = pd.read_csv(path)
    rows = df[df["Disease"] == disease]
    if rows.empty:
        return None, f"M3: no rows for '{disease}'"
    best = rows.loc[rows["Best_GP_AUC_PR"].idxmax()]
    formula = str(best["Best_GP_Formula"]).strip()
    return formula, f"M3 tier={best['Config_Used']} AUC-PR={best['Best_GP_AUC_PR']:.4f}"


def _get_m4_formula(disease):
    path = RESULTS_DIR / "method4_llm" / disease / "method4_results.csv"
    if not path.exists():
        return None, f"M4 method4_results.csv not found"
    df = pd.read_csv(path)
    if df.empty or "formula" not in df.columns:
        return None, "M4: results CSV empty or missing column"
    top = df.sort_values("auc_pr", ascending=False).iloc[0]
    return str(top["formula"]).strip(), "M4 top formula"


# ── Score vector computation ───────────────────────────────────────────────────

def _get_score_vector(method, formula, test_clean, features, y_test):
    """Return (scores, flipped) or (None, False)."""
    if method == "Baseline":
        return np.full(len(y_test), float(y_test.mean())), False

    if method == "M3":
        scores = eval_gp_formula(formula, test_clean, features)
    else:
        scores = eval_formula_scores(formula, test_clean, features)

    if scores is None:
        return None, False

    try:
        if roc_auc_score(y_test, scores) < 0.5:
            return -scores, True
    except Exception:
        pass

    return scores, False


# ── Correlation matrix ────────────────────────────────────────────────────────

def _correlation_matrix(score_dict):
    methods = list(score_dict.keys())
    n   = len(methods)
    mat = np.full((n, n), np.nan)

    for i in range(n):
        si = score_dict[methods[i]]
        if si is None:
            continue
        for j in range(n):
            sj = score_dict[methods[j]]
            if sj is None:
                continue
            if i == j:
                mat[i, j] = 1.0
                continue
            if np.std(si) < 1e-12 or np.std(sj) < 1e-12:
                continue
            try:
                r, _ = pearsonr(si, sj)
                mat[i, j] = round(float(r), 4)
            except Exception:
                pass

    return pd.DataFrame(mat, index=methods, columns=methods)


# ── Heatmap ───────────────────────────────────────────────────────────────────

def _save_heatmap(corr_df, disease, out_path):
    methods = corr_df.index.tolist()
    n   = len(methods)
    mat = corr_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im   = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_yticklabels(methods, fontsize=12)

    for i in range(n):
        for j in range(n):
            val  = mat[i, j]
            text = f"{val:.2f}" if np.isfinite(val) else "N/A"
            col  = "white" if (np.isfinite(val) and abs(val) > 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=11, color=col)

    plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
    ax.set_title(f"{disease.upper()} — Cross-Method Score Correlation", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_for_disease(disease):
    print(f"\n{'='*60}")
    print(f"Cross-method correlation — {disease.upper()}")
    print(f"{'='*60}")

    ensure_dir(OUT_DIR)

    try:
        df, features = load_data_for(disease)
    except FileNotFoundError as e:
        print(f"  ERROR loading data: {e}")
        return

    _, test_df  = get_splits(df)
    test_clean  = test_df.dropna(subset=features + ["is_case"])
    y_test      = test_clean["is_case"].values
    print(f"  Test set: {len(test_clean)} rows, {int(y_test.sum())} positives ({y_test.mean()*100:.2f}%)")

    # Collect formulas
    getters = {"M1": _get_m1_formula, "M2": _get_m2_formula,
                "M3": _get_m3_formula, "M4": _get_m4_formula}
    formulas = {}
    for method, getter in getters.items():
        formula, note = getter(disease)
        print(f"  {method}: {note}")
        formulas[method] = formula

    # Compute score vectors
    score_dict = {}
    for method in METHODS:
        formula = formulas.get(method)
        if method != "Baseline" and formula is None:
            score_dict[method] = None
            print(f"  Score {method}: SKIPPED (no formula)")
            continue
        scores, flipped = _get_score_vector(method, formula, test_clean, features, y_test)
        score_dict[method] = scores
        status = "OK" if scores is not None else "FAILED"
        print(f"  Score {method}: {status}{' (flipped)' if flipped else ''}")

    # Save score vectors
    sv_path = OUT_DIR / f"{disease}_score_vectors.csv"
    sv = {"subject_id": test_clean["subject_id"].values if "subject_id" in test_clean.columns
          else np.arange(len(test_clean))}
    for m in METHODS:
        sv[m] = score_dict[m] if score_dict[m] is not None else np.nan
    pd.DataFrame(sv).to_csv(sv_path, index=False)
    print(f"  Saved score vectors -> {sv_path}")

    # Correlation matrix
    corr_df   = _correlation_matrix(score_dict)
    corr_path = OUT_DIR / f"{disease}_score_correlation.csv"
    corr_df.to_csv(corr_path)
    print(f"  Saved correlation matrix -> {corr_path}")
    print(corr_df.round(3).to_string())

    # Heatmap
    heatmap_path = OUT_DIR / f"{disease}_correlation_heatmap.png"
    _save_heatmap(corr_df, disease, heatmap_path)
    print(f"  Saved heatmap -> {heatmap_path}")

    # Append to master summary
    def _r(m1, m2):
        try:
            v = float(corr_df.loc[m1, m2])
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    row_df = pd.DataFrame([{
        "Disease":    disease,
        "M1_M2_r":    _r("M1", "M2"),
        "M1_M3_r":    _r("M1", "M3"),
        "M1_M4_r":    _r("M1", "M4"),
        "M2_M3_r":    _r("M2", "M3"),
        "M2_M4_r":    _r("M2", "M4"),
        "M3_M4_r":    _r("M3", "M4"),
        "M1_formula": formulas.get("M1") or "",
        "M2_formula": formulas.get("M2") or "",
        "M3_formula": formulas.get("M3") or "",
        "M4_formula": formulas.get("M4") or "",
    }])

    if MASTER_SUMMARY.exists():
        row_df = pd.concat([pd.read_csv(MASTER_SUMMARY), row_df], ignore_index=True)
    row_df.to_csv(MASTER_SUMMARY, index=False)
    print(f"  Appended to master summary -> {MASTER_SUMMARY}")


def main():
    parser = argparse.ArgumentParser(description="Cross-method score vector correlation")
    parser.add_argument("--disease", required=True, help="Disease slug (e.g. ra, t1d)")
    args = parser.parse_args()
    run_for_disease(args.disease)


if __name__ == "__main__":
    main()
