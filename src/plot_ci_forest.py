"""
plot_ci_forest.py — Forest plot of AUC-PR point estimates with 95% bootstrap CIs.

Reads results/figures/{disease}_ci_data.csv (written by compute_ci_all.py) and
generates a horizontal forest plot saved to results/figures/ci_forest_auc_pr.png.

Usage:
    python src/plot_ci_forest.py --disease ra
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from utils import ensure_dir, RESULTS_DIR

METHOD_ORDER = ["m1", "m2", "m3", "m4"]
METHOD_LABELS = {
    "m1": "M1: Threshold",
    "m2": "M2: Random Search",
    "m3": "M3: Genetic Programming",
    "m4": "M4: LLM",
}
METHOD_COLORS = {
    "m1": "#4878CF",
    "m2": "#6ACC65",
    "m3": "#D65F5F",
    "m4": "#B47CC7",
}


def main():
    parser = argparse.ArgumentParser(description="Forest plot of AUC-PR CIs across methods")
    parser.add_argument("--disease", default="ra")
    args = parser.parse_args()

    fig_dir = RESULTS_DIR / "figures"
    ensure_dir(fig_dir)

    ci_path = fig_dir / f"{args.disease}_ci_data.csv"
    if not ci_path.exists():
        print(f"ERROR: CI data not found at {ci_path}")
        print("Run: python src/compute_ci_all.py --disease", args.disease)
        sys.exit(1)

    df = pd.read_csv(ci_path)
    print(f"Loaded CI data: {len(df)} rows")

    # Separate LR baseline from method rows
    lr_row = df[df["method"] == "lr_baseline"]
    lr_auc_pr = float(lr_row["auc_pr"].iloc[0]) if not lr_row.empty else None

    methods_df = df[df["method"].isin(METHOD_ORDER)].copy()
    methods_df["_order"] = methods_df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    methods_df = methods_df.sort_values("_order", ascending=False).reset_index(drop=True)

    n = len(methods_df)
    fig, ax = plt.subplots(figsize=(8, max(3, 1.2 * n + 1.5)))

    y_positions = np.arange(n)

    for i, (_, row) in enumerate(methods_df.iterrows()):
        method = row["method"]
        point = float(row["auc_pr"])
        color = METHOD_COLORS.get(method, "#888888")
        label = METHOD_LABELS.get(method, row.get("label", method))

        has_ci = (
            "AUC_PR_CI_Low" in row and "AUC_PR_CI_High" in row
            and pd.notna(row["AUC_PR_CI_Low"]) and pd.notna(row["AUC_PR_CI_High"])
        )

        if has_ci:
            lo, hi = float(row["AUC_PR_CI_Low"]), float(row["AUC_PR_CI_High"])
            ax.errorbar(
                point, y_positions[i],
                xerr=[[max(0.0, point - lo)], [max(0.0, hi - point)]],
                fmt="o", color=color, markersize=8,
                elinewidth=2, capsize=5, capthick=2,
                label=label,
            )
            ax.text(
                hi + 0.0003, y_positions[i],
                f"{point:.4f} [{lo:.4f}–{hi:.4f}]",
                va="center", ha="left", fontsize=8.5, color=color,
            )
        else:
            ax.plot(point, y_positions[i], "o", color=color, markersize=8, label=label)
            ax.text(
                point + 0.0003, y_positions[i],
                f"{point:.4f}",
                va="center", ha="left", fontsize=8.5, color=color,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [METHOD_LABELS.get(row["method"], row.get("label", row["method"]))
         for _, row in methods_df.iterrows()],
        fontsize=10,
    )

    if lr_auc_pr is not None:
        ax.axvline(
            lr_auc_pr, color="gray", linestyle="--", linewidth=1.5,
            label=f"LR baseline ({lr_auc_pr:.4f})",
        )

    disease_upper = args.disease.upper()
    ax.set_xlabel("AUC-PR (Average Precision)", fontsize=11)
    ax.set_title(f"{disease_upper} — AUC-PR with 95% Bootstrap CI", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = fig_dir / "ci_forest_auc_pr.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved forest plot: {out_path}")


if __name__ == "__main__":
    main()
