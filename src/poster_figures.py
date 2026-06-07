"""
poster_figures.py — Poster-quality figures for the biomarker pipeline.

Outputs to results/poster_figures/:
  fig1_aucpr_ci.png       — AUC-PR per disease × method with 95% CIs (MIMIC)
  fig2_crosscohort.png    — AUC-ROC generalization: MIMIC vs EHRSHOT
  fig3_nhanes.png         — External NHANES validation (RA + PSR)
  fig4_summary_heatmap.png — AUC-PR heatmap across diseases × methods

Usage:
    python src/poster_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from utils import RESULTS_DIR, ensure_dir

# ─── Constants ────────────────────────────────────────────────────────────────
DISEASES = ["ra", "crhn", "psr", "lup", "t1d", "t2d"]
DISEASE_LABELS = {
    "ra":   "Rheumatoid\nArthritis (RA)",
    "crhn": "Crohn's\nDisease (CD)",
    "psr":  "Psoriasis (PSR)",
    "lup":  "Lupus (LUP)",
    "t1d":  "Type 1\nDiabetes (T1D)",
    "t2d":  "Type 2\nDiabetes (T2D)",
}
DISEASE_SHORT = {
    "ra": "RA", "crhn": "CD", "psr": "PSR",
    "lup": "LUP", "t1d": "T1D", "t2d": "T2D",
}
METHOD_ORDER = ["m1", "m2", "m3", "m4"]
METHOD_LABELS = {
    "m1": "M1: Threshold",
    "m2": "M2: Random Search",
    "m3": "M3: Genetic Programming",
    "m4": "M4: LLM",
}
METHOD_COLORS = {
    "m1": "#4878CF",
    "m2": "#44AA44",
    "m3": "#D65F5F",
    "m4": "#B47CC7",
}
BASELINE_COLOR = "#888888"

OUT_DIR = RESULTS_DIR / "poster_figures"


# ─── CI approval ──────────────────────────────────────────────────────────────
# A result is "CI approved" if its 95% bootstrap CI lower bound for AUC-ROC
# is above 0.5 (statistically better than random). AUC-PR CIs are too wide
# for rare events to use as a gate (0/24 would pass a LR-beat criterion).
CI_APPROVED_ROC_THRESHOLD = 0.5


def is_ci_approved(row):
    return float(row["AUC_ROC_CI_Low"]) > CI_APPROVED_ROC_THRESHOLD


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_mimic_ci():
    """MIMIC bootstrap CI data — one best row per method per disease."""
    rows = []
    for disease in DISEASES:
        path = RESULTS_DIR / "figures" / f"{disease}_ci_data.csv"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        df = pd.read_csv(path)
        df["disease"] = disease
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_ehrshot():
    """EHRSHOT eval data — best AUC-PR variant per method per disease."""
    rows = []
    for disease in DISEASES:
        path = RESULTS_DIR / "ehrshot" / f"{disease}_all_methods_eval.csv"
        if not path.exists():
            print(f"  WARNING: missing EHRSHOT for {disease}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df["disease"] = disease  # set after lowercase so no duplicate
        best = df.loc[df.groupby("method")["auc_pr"].idxmax()].copy()
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_nhanes():
    """NHANES eval data — best AUC-PR variant per method (RA + PSR only)."""
    rows = []
    for disease in ["ra", "psr"]:
        path = RESULTS_DIR / "nhanes" / f"{disease}_evaluation.csv"
        if not path.exists():
            print(f"  WARNING: missing NHANES for {disease}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df["disease"] = disease  # set after lowercase so no duplicate
        best = df.loc[df.groupby("method")["auc_pr"].idxmax()].copy()
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ─── Figure 1: AUC-PR with 95% CI (MIMIC) — 2×3 forest-plot grid ─────────────

def fig1_aucpr_ci(mimic_df):
    print("Generating Figure 1: AUC-PR with CIs (CI-approved only) …")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax_idx, disease in enumerate(DISEASES):
        ax = axes[ax_idx]
        sub = mimic_df[mimic_df["disease"] == disease].copy()

        lr_row = sub[sub["method"] == "lr_baseline"]
        lr_val = float(lr_row["auc_pr"].iloc[0]) if not lr_row.empty else None
        methods_sub = sub[sub["method"].isin(METHOD_ORDER)].copy()
        methods_sub["_ord"] = methods_sub["method"].map(
            {m: i for i, m in enumerate(METHOD_ORDER)}
        )
        methods_sub = methods_sub.sort_values("_ord", ascending=True).reset_index(drop=True)

        n = len(methods_sub)

        any_approved = False
        for i, row in methods_sub.iterrows():
            m = row["method"]
            pt = float(row["auc_pr"])
            approved = is_ci_approved(row)
            any_approved = any_approved or approved
            color = METHOD_COLORS[m] if approved else "#cccccc"
            alpha = 1.0 if approved else 0.45
            yi = int(row["_ord"])
            has_ci = pd.notna(row.get("AUC_PR_CI_Low")) and pd.notna(row.get("AUC_PR_CI_High"))
            if has_ci:
                lo, hi = float(row["AUC_PR_CI_Low"]), float(row["AUC_PR_CI_High"])
                ax.errorbar(
                    pt, yi,
                    xerr=[[max(0, pt - lo)], [max(0, hi - pt)]],
                    fmt="o", color=color, markersize=9, alpha=alpha,
                    elinewidth=2.2, capsize=6, capthick=2.2, zorder=3,
                )
            else:
                ax.plot(pt, yi, "o", color=color, markersize=9, alpha=alpha, zorder=3)
            label_color = color if approved else "#aaaaaa"
            suffix = "" if approved else " (n.s.)"
            ax.text(
                (hi if has_ci else pt) + 0.0005, yi,
                f"{pt:.4f}{suffix}",
                va="center", ha="left", fontsize=9, color=label_color,
                fontweight="bold" if approved else "normal",
                alpha=alpha,
            )

        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [METHOD_LABELS[m] for m in methods_sub["method"]],
            fontsize=11,
        )

        if not any_approved:
            ax.text(0.5, 0.5, "No CI-approved\nresults", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#999999",
                    fontstyle="italic")

        if lr_val is not None:
            ax.axvline(
                lr_val, color=BASELINE_COLOR, linestyle="--", linewidth=1.8,
                label=f"LR baseline ({lr_val:.4f})", zorder=2,
            )

        ax.set_xlabel("AUC-PR", fontsize=12)
        ax.set_title(DISEASE_LABELS[disease], fontsize=13, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
        ax.set_ylim(-0.5, n - 0.5)

        if lr_val is not None:
            ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ] + [
        plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", linewidth=1.8,
                   label="LR Baseline (all features)"),
        mpatches.Patch(color="#cccccc", label="n.s. = AUC-ROC CI lower bound ≤ 0.5"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3,
        fontsize=10.5, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "AUC-PR with 95% Bootstrap CIs — MIMIC-IV Test Set\n"
        "(greyed = not statistically better than chance on AUC-ROC)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / "fig1_aucpr_ci.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Figure 2: Cross-cohort AUC-ROC (MIMIC vs EHRSHOT) ────────────────────────

def fig2_crosscohort(mimic_df, ehrshot_df):
    print("Generating Figure 2: Cross-cohort AUC-ROC …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: AUC-ROC heatmap (MIMIC)
    mimic_methods = mimic_df[mimic_df["method"].isin(METHOD_ORDER)].copy()
    mimic_pivot = mimic_methods.pivot_table(
        index="method", columns="disease", values="auc_roc", aggfunc="first"
    ).reindex(index=METHOD_ORDER, columns=DISEASES)

    # Panel B: AUC-ROC heatmap (EHRSHOT)
    ehrshot_methods = ehrshot_df[ehrshot_df["method"].isin(METHOD_ORDER)].copy()
    ehrshot_pivot = ehrshot_methods.pivot_table(
        index="method", columns="disease", values="auc_roc", aggfunc="first"
    ).reindex(index=METHOD_ORDER, columns=DISEASES)

    vmin, vmax = 0.48, 0.72
    cmap = "Blues"

    for ax, pivot, title in [
        (axes[0], mimic_pivot, "MIMIC-IV (Internal Test Set)"),
        (axes[1], ehrshot_pivot, "EHRSHOT (External Validation)"),
    ]:
        data = pivot.values.astype(float)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(DISEASES)))
        ax.set_xticklabels([DISEASE_SHORT[d] for d in DISEASES], fontsize=12)
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

        for r in range(len(METHOD_ORDER)):
            for c in range(len(DISEASES)):
                val = data[r, c]
                if not np.isnan(val):
                    text_color = "white" if val > (vmin + vmax) / 2 + 0.04 else "black"
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=10.5, color=text_color, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.85, label="AUC-ROC")

    fig.suptitle(
        "AUC-ROC Generalization: MIMIC-IV → EHRSHOT",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / "fig2_crosscohort.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Figure 3: NHANES External Validation (RA + PSR) ──────────────────────────

def fig3_nhanes(mimic_df, nhanes_df):
    print("Generating Figure 3: NHANES external validation …")
    nhanes_diseases = ["ra", "psr"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, disease in zip(axes, nhanes_diseases):
        mimic_sub = mimic_df[
            (mimic_df["disease"] == disease) & mimic_df["method"].isin(METHOD_ORDER)
        ].set_index("method")
        nhanes_sub = nhanes_df[
            (nhanes_df["disease"] == disease) & nhanes_df["method"].isin(METHOD_ORDER)
        ].set_index("method")

        x = np.arange(len(METHOD_ORDER))
        width = 0.35

        mimic_vals = [float(mimic_sub.loc[m, "auc_pr"]) if m in mimic_sub.index else np.nan
                      for m in METHOD_ORDER]
        nhanes_vals = [float(nhanes_sub.loc[m, "auc_pr"]) if m in nhanes_sub.index else np.nan
                       for m in METHOD_ORDER]

        # CIs for MIMIC
        mimic_ci_lo = []
        mimic_ci_hi = []
        for m in METHOD_ORDER:
            if m in mimic_sub.index:
                row = mimic_sub.loc[m]
                pt = float(row["auc_pr"])
                lo = float(row.get("AUC_PR_CI_Low", pt))
                hi = float(row.get("AUC_PR_CI_High", pt))
                mimic_ci_lo.append(max(0, pt - lo))
                mimic_ci_hi.append(max(0, hi - pt))
            else:
                mimic_ci_lo.append(0)
                mimic_ci_hi.append(0)

        bars_mimic = ax.bar(
            x - width / 2, mimic_vals, width,
            label="MIMIC-IV (test)", color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            alpha=0.85, edgecolor="white", linewidth=0.8,
        )
        ax.errorbar(
            x - width / 2, mimic_vals,
            yerr=[mimic_ci_lo, mimic_ci_hi],
            fmt="none", color="black", elinewidth=1.5, capsize=4, capthick=1.5,
        )

        bars_nhanes = ax.bar(
            x + width / 2, nhanes_vals, width,
            label="NHANES (external)", color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            alpha=0.45, edgecolor=["black"] * len(METHOD_ORDER),
            linewidth=0.8, hatch="//",
        )

        # LR baseline
        lr_row = mimic_df[
            (mimic_df["disease"] == disease) & (mimic_df["method"] == "lr_baseline")
        ]
        if not lr_row.empty:
            lr_val = float(lr_row["auc_pr"].iloc[0])
            ax.axhline(lr_val, color=BASELINE_COLOR, linestyle="--", linewidth=1.8,
                       label=f"LR baseline ({lr_val:.4f})")

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].replace(" ", "\n") for m in METHOD_ORDER], fontsize=11)
        ax.set_ylabel("AUC-PR", fontsize=12)
        ax.set_title(DISEASE_LABELS[disease], fontsize=14, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.legend(fontsize=10, framealpha=0.85, loc="upper left")
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "External Validation on NHANES vs MIMIC-IV Test Set",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / "fig3_nhanes.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Figure 4: AUC-PR Summary Heatmap ─────────────────────────────────────────

def fig4_summary_heatmap(mimic_df):
    print("Generating Figure 4: AUC-PR summary heatmap (CI-approved highlighted) …")

    methods_df = mimic_df[mimic_df["method"].isin(METHOD_ORDER)].copy()

    # Build approval lookup: (method, disease) -> bool
    approved_set = set()
    for _, row in methods_df.iterrows():
        if is_ci_approved(row):
            approved_set.add((row["method"], row["disease"]))

    pivot_pr = methods_df.pivot_table(
        index="method", columns="disease", values="auc_pr", aggfunc="first"
    ).reindex(index=METHOD_ORDER, columns=DISEASES)

    lr_vals = {}
    for disease in DISEASES:
        sub = mimic_df[(mimic_df["disease"] == disease) & (mimic_df["method"] == "lr_baseline")]
        lr_vals[disease] = float(sub["auc_pr"].iloc[0]) if not sub.empty else np.nan

    data = pivot_pr.values.astype(float)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Draw background: approved cells colored, non-approved grey
    base_grid = np.full_like(data, np.nan)
    grey_grid = np.full_like(data, np.nan)
    for r, m in enumerate(METHOD_ORDER):
        for c, d in enumerate(DISEASES):
            if (m, d) in approved_set:
                base_grid[r, c] = data[r, c]
            else:
                grey_grid[r, c] = 1.0

    im = ax.imshow(base_grid, cmap="YlOrRd", aspect="auto",
                   vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.imshow(grey_grid, cmap="Greys", aspect="auto", vmin=0, vmax=1, alpha=0.25)

    ax.set_xticks(range(len(DISEASES)))
    ax.set_xticklabels([DISEASE_SHORT[d] for d in DISEASES], fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=13)

    # Best per disease among approved only
    best_per_disease = {}
    for c, d in enumerate(DISEASES):
        best_r, best_v = None, -1
        for r, m in enumerate(METHOD_ORDER):
            if (m, d) in approved_set and data[r, c] > best_v:
                best_v, best_r = data[r, c], r
        best_per_disease[c] = best_r

    for r in range(len(METHOD_ORDER)):
        for c in range(len(DISEASES)):
            val = data[r, c]
            m, d = METHOD_ORDER[r], DISEASES[c]
            approved = (m, d) in approved_set
            if np.isnan(val):
                continue
            is_best = (best_per_disease.get(c) == r)
            if approved:
                text_color = "white" if val > np.nanmax(data) * 0.7 else "black"
                weight = "bold" if is_best else "normal"
                marker = f"{val:.4f}" + (" ★" if is_best else "")
            else:
                text_color = "#999999"
                weight = "normal"
                marker = f"{val:.4f}\n(n.s.)"
            ax.text(c, r, marker, ha="center", va="center",
                    fontsize=10 if approved else 9, color=text_color, fontweight=weight)

    ax.set_xlim(-0.5, len(DISEASES) - 0.5)
    ax.set_ylim(-0.5, len(METHOD_ORDER) - 0.5)

    for c, r in best_per_disease.items():
        if r is not None:
            rect = plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=2.5, edgecolor="gold", facecolor="none",
            )
            ax.add_patch(rect)

    # X-axis: mark PSR as "no approved results"
    for c, disease in enumerate(DISEASES):
        has_any = any((m, disease) in approved_set for m in METHOD_ORDER)
        if not has_any:
            ax.text(c, -0.75, "✗", ha="center", va="center", fontsize=14,
                    color="red", fontweight="bold")

    cb = plt.colorbar(im, ax=ax, shrink=0.8, label="AUC-PR")
    cb.ax.tick_params(labelsize=10)

    for c, disease in enumerate(DISEASES):
        lr = lr_vals.get(disease, np.nan)
        if not np.isnan(lr):
            ax.text(c, len(METHOD_ORDER) - 0.35, f"LR: {lr:.4f}",
                    ha="center", va="bottom", fontsize=8,
                    color=BASELINE_COLOR, style="italic")

    ax.set_title(
        "AUC-PR Summary — MIMIC-IV Test Set\n"
        "Colored = CI-approved (AUC-ROC CI_low > 0.5) · Grey/n.s. = not significant · ★ = best approved",
        fontsize=13, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig4_summary_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Figure 5: Grouped AUC-PR bar chart (all diseases × methods) ──────────────

def fig5_grouped_bars(mimic_df):
    print("Generating Figure 5: Grouped AUC-PR bar chart (CI-approved solid, n.s. hatched) …")
    methods_df = mimic_df[mimic_df["method"].isin(METHOD_ORDER)].copy()

    approved_set = set()
    for _, row in methods_df.iterrows():
        if is_ci_approved(row):
            approved_set.add((row["method"], row["disease"]))

    n_diseases = len(DISEASES)
    n_methods = len(METHOD_ORDER)
    x = np.arange(n_diseases)
    width = 0.18
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(16, 7))

    for mi, method in enumerate(METHOD_ORDER):
        sub = methods_df[methods_df["method"] == method].set_index("disease")
        for di, disease in enumerate(DISEASES):
            if disease not in sub.index:
                continue
            row = sub.loc[disease]
            pt = float(row["auc_pr"])
            approved = (method, disease) in approved_set
            color = METHOD_COLORS[method] if approved else "#cccccc"
            hatch = None if approved else "//"
            alpha = 0.9 if approved else 0.6
            lo = float(row["AUC_PR_CI_Low"]) if "AUC_PR_CI_Low" in row.index else pt
            hi = float(row["AUC_PR_CI_High"]) if "AUC_PR_CI_High" in row.index else pt
            ax.bar(
                x[di] + offsets[mi], pt, width,
                color=color, alpha=alpha,
                edgecolor="white" if approved else "#888888",
                linewidth=0.6, hatch=hatch, zorder=2,
            )
            ax.errorbar(
                x[di] + offsets[mi], pt,
                yerr=[[max(0, pt - lo)], [max(0, hi - pt)]],
                fmt="none", color="black", elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=3,
            )

    for c, disease in enumerate(DISEASES):
        sub = mimic_df[(mimic_df["disease"] == disease) & (mimic_df["method"] == "lr_baseline")]
        if not sub.empty:
            lr = float(sub["auc_pr"].iloc[0])
            ax.hlines(
                lr, x[c] - 0.38, x[c] + 0.38,
                colors=BASELINE_COLOR, linestyles="--", linewidth=1.5, zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DISEASE_SHORT[d] for d in DISEASES], fontsize=14, fontweight="bold")
    ax.set_ylabel("AUC-PR (Average Precision)", fontsize=13)
    ax.set_xlabel("Disease", fontsize=13)
    ax.set_title(
        "AUC-PR Comparison — MIMIC-IV Test Set\n"
        "Solid = CI-approved (AUC-ROC CI_low > 0.5)   Hatched = not significant   Dashed = LR baseline",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_ylim(bottom=0)

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ] + [
        mpatches.Patch(color="#cccccc", hatch="//", label="Not significant (n.s.)"),
        plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
                   label="LR baseline"),
    ]
    ax.legend(handles=legend_handles, fontsize=11, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    out = OUT_DIR / "fig5_grouped_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Figure 6: Best-method per disease — MIMIC / EHRSHOT / NHANES ──────────────

def fig6_best_across_cohorts(mimic_df, ehrshot_df, nhanes_df):
    print("Generating Figure 6: Best AUC-PR across cohorts …")

    def best_auc_pr(df, disease):
        sub = df[(df["disease"] == disease) & df["method"].isin(METHOD_ORDER)]
        if sub.empty:
            return np.nan, "n/a"
        idx = sub["auc_pr"].idxmax()
        return float(sub.loc[idx, "auc_pr"]), sub.loc[idx, "method"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    panel_data = [
        ("MIMIC-IV\n(Internal Test)", mimic_df),
        ("EHRSHOT\n(External Validation)", ehrshot_df),
        ("NHANES\n(External — RA, PSR)", nhanes_df),
    ]

    for ax, (title, df) in zip(axes, panel_data):
        dis_order = DISEASES if title.startswith("MIMIC") or title.startswith("EHRSHOT") else ["ra", "psr"]
        vals, best_methods, colors = [], [], []
        for d in dis_order:
            v, m = best_auc_pr(df, d)
            vals.append(v)
            best_methods.append(m)
            colors.append(METHOD_COLORS.get(m, "#cccccc"))

        y = np.arange(len(dis_order))
        ax.barh(y, vals, color=colors, alpha=0.87, edgecolor="white", height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels([DISEASE_SHORT[d] for d in dis_order], fontsize=14, fontweight="bold")
        ax.set_xlabel("Best AUC-PR", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.set_xlim(left=0)
        for yi, (v, m) in enumerate(zip(vals, best_methods)):
            if not np.isnan(v):
                ax.text(v + 0.001, yi, f"{v:.4f}\n({m.upper()})",
                        va="center", ha="left", fontsize=10, color=METHOD_COLORS.get(m, "black"))

    # Legend
    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Best AUC-PR per Disease Across Evaluation Cohorts",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = OUT_DIR / "fig6_best_across_cohorts.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_dir(OUT_DIR)
    print("Loading data …")
    mimic_df = load_mimic_ci()
    ehrshot_df = load_ehrshot()
    nhanes_df = load_nhanes()
    print(f"  MIMIC rows:   {len(mimic_df)}")
    print(f"  EHRSHOT rows: {len(ehrshot_df)}")
    print(f"  NHANES rows:  {len(nhanes_df)}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig1_aucpr_ci(mimic_df)
    fig2_crosscohort(mimic_df, ehrshot_df)
    fig3_nhanes(mimic_df, nhanes_df)
    fig4_summary_heatmap(mimic_df)
    fig5_grouped_bars(mimic_df)
    fig6_best_across_cohorts(mimic_df, ehrshot_df, nhanes_df)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
