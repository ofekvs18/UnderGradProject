"""Generate AUC-PR lift bar charts for all 6 diseases."""
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from utils import load_data_for, get_splits, load_disease_config

DISEASES = [
    ("ra",   "Rheumatoid Arthritis"),
    ("crhn", "Crohn's Disease"),
    ("t1d",  "Type 1 Diabetes"),
    ("t2d",  "Type 2 Diabetes"),
    ("psr",  "Psoriasis"),
    ("lup",  "Lupus"),
]

KEYS   = ["m1", "m2", "m3", "m4"]
LABELS = ["M1\nThreshold", "M2\nRandom", "M3\nGP", "M4\nLLM"]
COLORS = ["#F59E0B", "#10B981", "#0D9488", "#8B5CF6"]
DARKER = ["#B45309", "#059669", "#0F766E", "#6D28D9"]


def make_chart(slug: str, full_name: str):
    # prevalence from actual test split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disease = load_disease_config(slug)
        df_data, _ = load_data_for(disease.name)
    _, test_df = get_splits(df_data)
    prevalence = test_df["is_case"].mean()

    ci_path = f"results/figures/{slug}_ci_data.csv"
    ci_df = pd.read_csv(ci_path)

    aucprs, ci_lows, ci_highs = [], [], []
    for key in KEYS:
        row = ci_df[ci_df["method"] == key].iloc[0]
        aucprs.append(float(row["auc_pr"]))
        ci_lows.append(float(row["AUC_PR_CI_Low"]) if pd.notna(row["AUC_PR_CI_Low"]) else np.nan)
        ci_highs.append(float(row["AUC_PR_CI_High"]) if pd.notna(row["AUC_PR_CI_High"]) else np.nan)

    lr_aucpr = float(ci_df[ci_df["method"] == "lr_baseline"].iloc[0]["auc_pr"])

    lifts      = [v / prevalence for v in aucprs]
    ci_lo_lift = [v / prevalence if not np.isnan(v) else np.nan for v in ci_lows]
    ci_hi_lift = [v / prevalence if not np.isnan(v) else np.nan for v in ci_highs]
    lr_lift    = lr_aucpr / prevalence

    yerr_lo = [max(lifts[i] - ci_lo_lift[i], 0) if not np.isnan(ci_lo_lift[i]) else 0
               for i in range(len(lifts))]
    yerr_hi = [max(ci_hi_lift[i] - lifts[i], 0) if not np.isnan(ci_hi_lift[i]) else 0
               for i in range(len(lifts))]

    y_min = min(lifts) * 0.85
    y_max = max(
        ci_hi_lift[i] if not np.isnan(ci_hi_lift[i]) else lifts[i]
        for i in range(len(lifts))
    ) * 1.18

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="none")
    ax.set_facecolor("none")

    x    = np.arange(len(LABELS))
    bars = ax.bar(x, lifts, width=0.55, color=COLORS, zorder=3,
                  edgecolor="white", linewidth=0.8)

    for i in range(len(lifts)):
        ax.errorbar(
            x[i], lifts[i],
            yerr=[[yerr_lo[i]], [yerr_hi[i]]],
            fmt="none",
            ecolor=DARKER[i],
            elinewidth=1.5,
            capsize=6,
            capthick=1.5,
            zorder=4,
        )

    for i, (bar, lift) in enumerate(zip(bars, lifts)):
        top = ci_hi_lift[i] if not np.isnan(ci_hi_lift[i]) else lift
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(top, lift) + (y_max - y_min) * 0.02,
                f"{lift:.2f}x",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#1e293b")

    ax.axhline(lr_lift, color="#64748B", linewidth=1.6, linestyle="--", zorder=2)
    ax.text(-0.48, lr_lift + (y_max - y_min) * 0.015,
            f"LR all-features  ({lr_lift:.2f}x)",
            ha="left", va="bottom", fontsize=10, color="#64748B")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=13, color="#1e293b")
    ax.set_ylabel("AUC-PR Lift  (AUC-PR / prevalence)", fontsize=12, color="#475569", labelpad=10)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_tick_params(labelsize=11, colors="#475569")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.96, f"{full_name} — AUC-PR Lift by Method",
             ha="center", va="top", fontsize=17, fontweight="bold", color="#0F172A")
    fig.text(0.5, 0.905,
             f"Lift = AUC-PR / prevalence  |  prevalence = {prevalence:.2%}  |  dashed = LR baseline",
             ha="center", va="top", fontsize=11, color="#64748B")

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out = f"results/{slug}_aucpr_lift.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out}  (prevalence={prevalence:.3%})")


for slug, full_name in DISEASES:
    print(f"\n{slug} — {full_name}")
    make_chart(slug, full_name)

print("\nDone.")
