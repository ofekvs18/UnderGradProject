"""
build_dashboard_data.py — Aggregate best-formula rows from all four method
master summaries into a single dashboard CSV.

For Method 3 (GP), the raw gplearn S-expression is converted to human-readable
infix notation so the dashboard can show it without mul( / div( clutter.

Output: results/dashboard/{disease}_dashboard_data.csv
Columns: disease, method, variant, formula, formula_display, auc_pr, auc_roc

Usage:
    python src/build_dashboard_data.py --disease ra
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from utils import ensure_dir, RESULTS_DIR


# ── GP S-expression → infix converter ────────────────────────────────────────

def gp_prefix_to_infix(formula: str) -> str:
    """Convert gplearn S-expression to human-readable infix notation."""
    BINARY_OPS = {"mul": "*", "div": "/", "add": "+", "sub": "-"}
    UNARY_OPS = {"sqrt", "log", "abs", "neg"}

    def split_top_args(s):
        depth, args, cur = 0, [], []
        for c in s:
            if c == "(":
                depth += 1
                cur.append(c)
            elif c == ")":
                depth -= 1
                cur.append(c)
            elif c == "," and depth == 0:
                args.append("".join(cur).strip())
                cur = []
            else:
                cur.append(c)
        if cur:
            args.append("".join(cur).strip())
        return args

    def parse(s):
        s = s.strip()
        m = re.match(r'^(\w+)\(', s)
        if not m:
            return s
        name = m.group(1)
        i, depth = m.end() - 1, 0
        while i < len(s):
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        inner = s[m.end():i]
        if name in BINARY_OPS:
            a, b = split_top_args(inner)
            return f"({parse(a)} {BINARY_OPS[name]} {parse(b)})"
        elif name == "neg":
            return f"(-{parse(inner)})"
        elif name in UNARY_OPS:
            return f"{name}({parse(inner)})"
        return s

    try:
        result = parse(formula.strip())
        if result.startswith("(") and result.endswith(")"):
            result = result[1:-1]
        return result
    except Exception:
        return formula


def _best_row(df: pd.DataFrame, auc_col: str):
    """Return the row with the highest value in auc_col."""
    if df.empty:
        return None
    return df.loc[df[auc_col].idxmax()]


def build_dashboard_data(disease: str) -> pd.DataFrame:
    rows = []

    # ── M1: threshold ────────────────────────────────────────────────────────
    m1_path = RESULTS_DIR / "method1_threshold" / "master_m1_summary.csv"
    if m1_path.exists():
        m1 = pd.read_csv(m1_path)
        m1_dis = m1[m1["Disease"] == disease]
        if not m1_dis.empty:
            # Best literature formula
            r = _best_row(m1_dis, "Best_Lit_AUC_PR")
            if r is not None and str(r.get("Best_Lit_Formula", "")).lower() not in ("", "nan"):
                formula = str(r["Best_Lit_Formula"])
                rows.append({
                    "disease": disease, "method": "m1", "variant": "lit",
                    "formula": formula, "formula_display": formula,
                    "auc_pr": float(r["Best_Lit_AUC_PR"]),
                    "auc_roc": float(r["Best_Lit_AUC_ROC"]),
                })
            # Best data-driven formula
            r = _best_row(m1_dis, "Best_DD_AUC_PR")
            if r is not None and str(r.get("Best_DD_Formula", "")).lower() not in ("", "nan"):
                formula = str(r["Best_DD_Formula"])
                rows.append({
                    "disease": disease, "method": "m1", "variant": "dd",
                    "formula": formula, "formula_display": formula,
                    "auc_pr": float(r["Best_DD_AUC_PR"]),
                    "auc_roc": float(r["Best_DD_AUC_ROC"]),
                })
        print(f"[m1] {len([r for r in rows if r['method'] == 'm1'])} row(s) loaded")
    else:
        print(f"[m1] master summary not found: {m1_path}")

    # ── M2: random formula search ────────────────────────────────────────────
    m2_path = RESULTS_DIR / "method2_random" / "master_m2_summary.csv"
    if m2_path.exists():
        m2 = pd.read_csv(m2_path)
        m2_dis = m2[m2["Disease"] == disease]
        if not m2_dis.empty:
            # Use CV-selected formula + frozen test AUC-PR — not train-set best (overfit)
            r = _best_row(m2_dis, "CV_Winner_Frozen_Test_AUC_PR")
            if r is not None:
                formula = str(r["CV_Winner_Formula"])
                rows.append({
                    "disease": disease, "method": "m2", "variant": "random",
                    "formula": formula, "formula_display": formula,
                    "auc_pr": float(r["CV_Winner_Frozen_Test_AUC_PR"]),
                    "auc_roc": float("nan"),
                })
        print(f"[m2] {len([r for r in rows if r['method'] == 'm2'])} row(s) loaded")
    else:
        print(f"[m2] master summary not found: {m2_path}")

    # ── M3: genetic programming ──────────────────────────────────────────────
    m3_path = RESULTS_DIR / "method3_gp" / "master_gp_summary.csv"
    if m3_path.exists():
        m3 = pd.read_csv(m3_path)
        m3_dis = m3[m3["Disease"] == disease]
        if not m3_dis.empty:
            # Only use CV-selected rows; pick best by Frozen_Test_AUC_PR_Final
            cv_sel = m3_dis[m3_dis["CV_Selected"] == True]
            if cv_sel.empty:
                cv_sel = m3_dis  # fallback
            r = _best_row(cv_sel, "Frozen_Test_AUC_PR_Final")
            if r is not None:
                formula = str(r["Best_GP_Formula"])
                formula_display = gp_prefix_to_infix(formula)
                rows.append({
                    "disease": disease, "method": "m3", "variant": "gp",
                    "formula": formula, "formula_display": formula_display,
                    "auc_pr": float(r["Frozen_Test_AUC_PR_Final"]),
                    "auc_roc": float(r.get("Best_GP_AUC_ROC", float("nan"))),
                })
                print(f"[m3] formula_display: {formula_display[:80]}")
        print(f"[m3] {len([r for r in rows if r['method'] == 'm3'])} row(s) loaded")
    else:
        print(f"[m3] master summary not found: {m3_path}")

    # ── M4: LLM-generated formulas ───────────────────────────────────────────
    m4_path = RESULTS_DIR / "method4_llm" / "method4_master_summary.csv"
    if m4_path.exists():
        m4 = pd.read_csv(m4_path)
        m4_dis = m4[m4["Disease"] == disease]
        if not m4_dis.empty:
            r = _best_row(m4_dis, "Best_LLM_AUC_PR")
            if r is not None:
                formula = str(r["Best_LLM_Formula"])
                rows.append({
                    "disease": disease, "method": "m4", "variant": "llm",
                    "formula": formula, "formula_display": formula,
                    "auc_pr": float(r["Best_LLM_AUC_PR"]),
                    "auc_roc": float(r["Best_LLM_AUC_ROC"]),
                })
        print(f"[m4] {len([r for r in rows if r['method'] == 'm4'])} row(s) loaded")
    else:
        print(f"[m4] master summary not found: {m4_path}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build dashboard CSV from method master summaries"
    )
    parser.add_argument("--disease", default="ra", help="Disease slug (default: ra)")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "dashboard"
    ensure_dir(out_dir)
    out_path = out_dir / f"{args.disease}_dashboard_data.csv"

    print(f"Building dashboard data for disease: {args.disease}")
    df = build_dashboard_data(args.disease)

    if df.empty:
        print("WARNING: no rows collected — check that master summaries exist.")
        return

    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows -> {out_path}")
    print(df[["method", "variant", "auc_pr", "auc_roc", "formula_display"]].to_string(index=False))


if __name__ == "__main__":
    main()
