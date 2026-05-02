"""
Aggregate master summaries from all methods into a single comparison table.

Reads:
    results/sanity_check/master_sanity_summary.csv
    results/method1_threshold/master_m1_summary.csv
    results/method2_random/master_m2_summary.csv
    results/method3_gp/master_gp_summary.csv
    results/method4_llm/method4_master_summary.csv

Output (one row per disease+split):
    Disease, Split_Salt,
    M1_Best_Formula, M1_Best_AUC_PR,
    M2_Best_Formula, M2_Best_AUC_PR,
    M3_Best_Formula, M3_Best_AUC_PR,
    M4_Best_Formula, M4_Best_AUC_PR,
    Best_Method, Best_Formula, Best_AUC_PR

Saved to: results/methods_comparison.csv
"""

import pandas as pd
from pathlib import Path

from utils import RESULTS_DIR, ensure_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [SKIP] Not found: {path}")
        return None
    df = pd.read_csv(path)
    if "Split_Salt" not in df.columns:
        df["Split_Salt"] = ""
    df["Split_Salt"] = df["Split_Salt"].fillna("")
    return df


def _best_per_group(df: pd.DataFrame, auc_col: str) -> pd.DataFrame:
    """Return the row with the highest AUC-PR per (Disease, Split_Salt) group."""
    return (
        df.sort_values(auc_col, ascending=False)
          .groupby(["Disease", "Split_Salt"], sort=False)
          .first()
          .reset_index()
    )


# ── Load & distill each method ────────────────────────────────────────────────

def load_m1() -> pd.DataFrame | None:
    df = _load(RESULTS_DIR / "method1_threshold" / "master_m1_summary.csv")
    if df is None:
        return None
    # Pick the better of lit vs data-driven per row, then take best per group
    df["M1_Best_AUC_PR"] = df[["Best_Lit_AUC_PR", "Best_DD_AUC_PR"]].max(axis=1)
    df["M1_Best_Formula"] = df.apply(
        lambda r: r["Best_Lit_Formula"]
        if r["Best_Lit_AUC_PR"] >= r["Best_DD_AUC_PR"]
        else r["Best_DD_Formula"],
        axis=1,
    )
    best = _best_per_group(df, "M1_Best_AUC_PR")
    return best[["Disease", "Split_Salt", "M1_Best_Formula", "M1_Best_AUC_PR"]]


def load_m2() -> pd.DataFrame | None:
    df = _load(RESULTS_DIR / "method2_random" / "master_m2_summary.csv")
    if df is None:
        return None
    best = _best_per_group(df, "Best_Random_AUC_PR")
    best = best.rename(columns={
        "Best_Random_Formula": "M2_Best_Formula",
        "Best_Random_AUC_PR":  "M2_Best_AUC_PR",
    })
    return best[["Disease", "Split_Salt", "M2_Best_Formula", "M2_Best_AUC_PR"]]


def load_m3() -> pd.DataFrame | None:
    df = _load(RESULTS_DIR / "method3_gp" / "master_gp_summary.csv")
    if df is None:
        return None
    best = _best_per_group(df, "Best_GP_AUC_PR")
    best = best.rename(columns={
        "Best_GP_Formula": "M3_Best_Formula",
        "Best_GP_AUC_PR":  "M3_Best_AUC_PR",
    })
    return best[["Disease", "Split_Salt", "M3_Best_Formula", "M3_Best_AUC_PR"]]


def load_m4() -> pd.DataFrame | None:
    df = _load(RESULTS_DIR / "method4_llm" / "method4_master_summary.csv")
    if df is None:
        return None
    best = _best_per_group(df, "Best_LLM_AUC_PR")
    best = best.rename(columns={
        "Best_LLM_Formula": "M4_Best_Formula",
        "Best_LLM_AUC_PR":  "M4_Best_AUC_PR",
    })
    return best[["Disease", "Split_Salt", "M4_Best_Formula", "M4_Best_AUC_PR"]]


# ── Build comparison table ────────────────────────────────────────────────────

def build_comparison() -> pd.DataFrame:
    METHOD_COLS = {
        "Method1_Threshold": ("M1_Best_Formula", "M1_Best_AUC_PR"),
        "Method2_Random":    ("M2_Best_Formula", "M2_Best_AUC_PR"),
        "Method3_GP":        ("M3_Best_Formula", "M3_Best_AUC_PR"),
        "Method4_LLM":       ("M4_Best_Formula", "M4_Best_AUC_PR"),
    }

    loaders = [load_m1, load_m2, load_m3, load_m4]
    frames = [fn() for fn in loaders]
    frames = [f for f in frames if f is not None]

    if not frames:
        print("No master summary files found — nothing to compare.")
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["Disease", "Split_Salt"], how="outer")

    # Determine overall winner per row
    def _winner(row):
        best_method, best_formula, best_auc = None, None, -1.0
        for method, (formula_col, auc_col) in METHOD_COLS.items():
            if auc_col in row and pd.notna(row[auc_col]) and row[auc_col] > best_auc:
                best_auc    = row[auc_col]
                best_formula = row.get(formula_col)
                best_method  = method
        return pd.Series({
            "Best_Method":  best_method,
            "Best_Formula": best_formula,
            "Best_AUC_PR":  round(best_auc, 4) if best_auc >= 0 else None,
        })

    winner_cols = merged.apply(_winner, axis=1)
    result = pd.concat([merged, winner_cols], axis=1)

    # Column order
    ordered = ["Disease", "Split_Salt", "Best_Method", "Best_Formula", "Best_AUC_PR"]
    for _, (formula_col, auc_col) in METHOD_COLS.items():
        ordered += [formula_col, auc_col]

    cols = [c for c in ordered if c in result.columns]
    return result[cols].sort_values(["Disease", "Split_Salt"]).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building methods comparison table...")

    comparison = build_comparison()
    if comparison.empty:
        return

    out_path = RESULTS_DIR / "methods_comparison.csv"
    ensure_dir(out_path.parent)
    comparison.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
