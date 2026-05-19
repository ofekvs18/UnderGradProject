"""
nhanes_sanity.py — Descriptive statistics for NHANES modeling data.

Reads the NHANES modeling CSV produced by nhanes_data.py and reports:
  - Dataset overview (N, cases, prevalence, train/test split)
  - Per-feature CBC coverage (fraction of participants with each measurement)
  - Per-feature descriptive statistics (mean, std, quartiles)
  - Case vs. control mean comparison per feature

Usage:
    python src/nhanes_sanity.py --disease ra
    python src/nhanes_sanity.py --disease t2d
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from utils import load_disease_config, ensure_dir, DATA_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="NHANES sanity check and descriptive stats")
    parser.add_argument("--disease", default="ra", help="Disease slug (default: ra)")
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
    print(f"NHANES Sanity Check — {disease.full_name} ({disease.name})")
    print("=" * 70)

    df = pd.read_csv(data_path)
    meta_cols = {"subject_id", "is_case", "split"}
    features = [c for c in df.columns if c not in meta_cols]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Dataset overview ─────────────────────────────────────────────────────
    n_total    = len(df)
    n_cases    = int(df["is_case"].sum())
    n_controls = n_total - n_cases
    prevalence = round(df["is_case"].mean(), 4)
    n_train    = int((df["split"] == "train").sum())
    n_test     = int((df["split"] == "test").sum())
    n_all_cbc  = int(df[features].notna().all(axis=1).sum())

    print(f"\n=== Dataset Overview ===")
    print(f"  Total participants : {n_total:,}")
    print(f"  Cases              : {n_cases:,} ({prevalence:.2%})")
    print(f"  Controls           : {n_controls:,}")
    print(f"  Train / Test       : {n_train:,} / {n_test:,}")
    print(f"  All {len(features)} CBC present  : {n_all_cbc:,} ({n_all_cbc/n_total:.1%})")

    overview_row = {
        "Timestamp":   timestamp,
        "Disease":     args.disease,
        "N_Total":     n_total,
        "N_Cases":     n_cases,
        "N_Controls":  n_controls,
        "Prevalence":  prevalence,
        "N_Train":     n_train,
        "N_Test":      n_test,
        "N_All_CBC":   n_all_cbc,
        "Pct_All_CBC": round(n_all_cbc / n_total, 4),
    }

    # ── CBC feature coverage ─────────────────────────────────────────────────
    print(f"\n=== CBC Feature Coverage ===")
    coverage_rows = []
    for feat in features:
        n_obs = int(df[feat].notna().sum())
        pct = round(n_obs / n_total, 4)
        coverage_rows.append({
            "Timestamp":  timestamp,
            "Disease":    args.disease,
            "Feature":    feat,
            "N_Observed": n_obs,
            "Coverage":   pct,
        })
        print(f"  {feat:6s}: {n_obs:,} / {n_total:,} ({pct:.1%})")

    # ── Per-feature descriptive statistics ───────────────────────────────────
    print(f"\n=== Feature Descriptive Statistics ===")
    stat_rows = []
    for feat in features:
        vals = df[feat].dropna()
        if vals.empty:
            continue
        row = {
            "Timestamp": timestamp,
            "Disease":   args.disease,
            "Feature":   feat,
            "N":         len(vals),
            "Mean":      round(float(vals.mean()), 4),
            "Std":       round(float(vals.std()), 4),
            "Min":       round(float(vals.min()), 4),
            "P25":       round(float(vals.quantile(0.25)), 4),
            "Median":    round(float(vals.median()), 4),
            "P75":       round(float(vals.quantile(0.75)), 4),
            "Max":       round(float(vals.max()), 4),
        }
        stat_rows.append(row)
        print(f"  {feat:6s}: mean={row['Mean']:.3f}  std={row['Std']:.3f}  "
              f"median={row['Median']:.3f}  [{row['Min']:.3f}, {row['Max']:.3f}]")

    # ── Case vs. control mean comparison ────────────────────────────────────
    print(f"\n=== Case vs. Control Feature Means ===")
    case_df = df[df["is_case"] == 1]
    ctrl_df = df[df["is_case"] == 0]
    comp_rows = []
    for feat in features:
        case_mean = round(float(case_df[feat].mean()), 4) if case_df[feat].notna().any() else np.nan
        ctrl_mean = round(float(ctrl_df[feat].mean()), 4) if ctrl_df[feat].notna().any() else np.nan
        delta = (round(case_mean - ctrl_mean, 4)
                 if not (np.isnan(case_mean) or np.isnan(ctrl_mean)) else np.nan)
        comp_rows.append({
            "Timestamp":    timestamp,
            "Disease":      args.disease,
            "Feature":      feat,
            "Case_Mean":    case_mean,
            "Control_Mean": ctrl_mean,
            "Delta":        delta,
        })
        print(f"  {feat:6s}: cases={case_mean:.3f}  controls={ctrl_mean:.3f}  Δ={delta:+.3f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    def append_csv(path, new_df):
        if path.exists():
            new_df = pd.concat([pd.read_csv(path), new_df], ignore_index=True)
        new_df.to_csv(path, index=False)

    overview_path = out_dir / "nhanes_overview.csv"
    append_csv(overview_path, pd.DataFrame([overview_row]))
    print(f"\nSaved overview      → {overview_path}")

    coverage_path = out_dir / f"{args.disease}_cbc_coverage.csv"
    append_csv(coverage_path, pd.DataFrame(coverage_rows))
    print(f"Saved CBC coverage  → {coverage_path}")

    stats_path = out_dir / f"{args.disease}_feature_stats.csv"
    append_csv(stats_path, pd.DataFrame(stat_rows))
    print(f"Saved feature stats → {stats_path}")

    comp_path = out_dir / f"{args.disease}_case_ctrl_comparison.csv"
    append_csv(comp_path, pd.DataFrame(comp_rows))
    print(f"Saved case/ctrl     → {comp_path}")


if __name__ == "__main__":
    main()
