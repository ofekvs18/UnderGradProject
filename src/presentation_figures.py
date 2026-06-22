"""
Presentation figures for the biomarker pipeline CBC project.

Produces docs/figures/:
  figA_{disease}.png               – per-disease AUC-PR with 95% CI (6 separate files)
  figB_all_diseases_aupr.png       – all diseases grouped bar chart
  figC_complexity.png              – heatmap: N_Features (color) + AUC-PR (text)
  figD_ehrshot_generalization.png  – MIMIC vs EHRSHOT side-by-side (2x3 grid, AUC-ROC)
  figE_nhanes_generalization.png   – MIMIC vs NHANES for RA + Psoriasis (AUC-ROC)
  figF_mimic_lift.png              – MIMIC AUC-PR / test-set prevalence
  figG_ehrshot_lift.png            – EHRSHOT AUC-PR / cohort prevalence

Run:
    python src/presentation_figures.py
    python src/presentation_figures.py --refresh-ci   # refresh CI files first (slow, ~10 min)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DISEASES = ["ra", "crhn", "lup", "psr", "t1d", "t2d"]
DISEASE_NAMES = {
    "ra":   "RA",
    "crhn": "Crohn's",
    "lup":  "Lupus",
    "psr":  "Psoriasis",
    "t1d":  "T1D",
    "t2d":  "T2D",
}
METHODS = ["m1", "m2", "m3", "m4", "m5"]
LABELS  = ["M1 Threshold", "M2 Random", "M3 GP", "M4 LLM", "M5 Seeded GP ★"]
COLORS  = ["#9E9E9E", "#429BF4", "#34A853", "#FBBC05", "#0B6027"]
METHOD_LABELS = dict(zip(METHODS, LABELS))
METHOD_COLORS = dict(zip(METHODS, COLORS))

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    "#FAFAFA",
    "figure.facecolor":  "white",
    "font.size":         11,
})


def _savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_ci(subdir, disease):
    p = RES_DIR / subdir / f"{disease}_ci_data.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_mimic_ci(disease):   return _load_ci("figures", disease)
def load_ehrshot_ci(disease): return _load_ci("ehrshot", disease)
def load_nhanes_ci(disease):  return _load_ci("nhanes",  disease)


def _get_row(df, method):
    if df.empty:
        return None
    rows = df[df["method"] == method]
    return rows.iloc[0] if not rows.empty else None


def _safe_yerr(row, val, ci_lo="AUC_PR_CI_Low", ci_hi="AUC_PR_CI_High"):
    """Standard asymmetric error bar — returns None if CI is inverted or missing."""
    try:
        lo = val - float(row[ci_lo])
        hi = float(row[ci_hi]) - val
        if lo < 0 or hi < 0 or np.isnan(lo) or np.isnan(hi):
            return None
        return [[lo], [hi]]
    except (KeyError, TypeError, ValueError):
        return None


def _draw_ci_bracket(ax, xi, row, ci_lo="AUC_PR_CI_Low", ci_hi="AUC_PR_CI_High"):
    """
    Draw an absolute CI bracket [ci_lo, ci_hi] as a vertical line with end caps.
    Works even when the CI is inverted (bracket sits below the bar), which is an
    honest visual: the point estimate is outside its own bootstrap CI.
    """
    try:
        lo = float(row[ci_lo])
        hi = float(row[ci_hi])
        if np.isnan(lo) or np.isnan(hi):
            return
        cap = 0.11
        ax.vlines(xi, lo, hi, color="#222", linewidth=1.8, zorder=5)
        ax.hlines([lo, hi], xi - cap, xi + cap, color="#222", linewidth=1.5, zorder=5)
    except (KeyError, TypeError, ValueError):
        pass


def _method_legend_handles():
    return [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
            for m in METHODS]


# ── Prevalence helpers ────────────────────────────────────────────────────────

def mimic_prevalence(disease):
    p = ROOT / "data" / f"{disease}_modeling_data.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["is_case", "split"])
    test = df[df["split"] == "test"]
    return float(test["is_case"].mean()) if len(test) > 0 else None


def ehrshot_prevalence(disease):
    p = ROOT / "data" / f"{disease}_ehrshot_data.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["is_case"])
    return float(df["is_case"].mean()) if len(df) > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# Fig A  –  Per-disease AUC-PR with 95% CI bracket  (one PNG per disease)
# ══════════════════════════════════════════════════════════════════════════════

def figA_per_disease():
    for disease in DISEASES:
        df = load_mimic_ci(disease)
        if df.empty:
            print(f"  [figA] no CI data for {disease}, skipping")
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_facecolor("white")

        lr = _get_row(df, "lr_baseline")
        if lr is not None:
            ax.axhline(float(lr["auc_pr"]), color="#555", linewidth=1.3,
                       linestyle="--", alpha=0.65, zorder=2)

        present = [m for m in METHODS if _get_row(df, m) is not None]
        x = np.arange(len(present))

        for xi, m in enumerate(present):
            row = _get_row(df, m)
            auc = float(row["auc_pr"])
            ax.bar(xi, auc, 0.55, color=METHOD_COLORS[m], zorder=3,
                   edgecolor="white", linewidth=0.5)
            # Always draw absolute CI bracket — even if inverted, shows where CI is
            _draw_ci_bracket(ax, xi, row)
            ax.text(xi, auc + 0.0008, f"{auc:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=METHOD_COLORS[m], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in present], fontsize=11)
        ax.set_ylabel("AUC-PR", fontsize=12)
        ax.set_title(
            f"{DISEASE_NAMES[disease]} — AUC-PR per Method  (MIMIC Test Set, 95% Bootstrap CI)",
            fontsize=13, fontweight="bold", color="#1F397D", pad=10)
        ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
        ax.set_ylim(0, None)

        handles = _method_legend_handles()
        if lr is not None:
            handles += [plt.Line2D([0], [0], color="#555", linewidth=1.4,
                                   linestyle="--", label="LR Baseline (all features)")]
        handles += [plt.Line2D([0], [0], color="#222", linewidth=1.8,
                               label="95% Bootstrap CI")]
        ax.legend(handles=handles, fontsize=9, ncol=3, loc="upper right", framealpha=0.85)

        fig.tight_layout()
        _savefig(fig, f"figA_{disease}.png")


def figA_grid():
    """Same as figA_per_disease but as a 2×3 grid in one PNG."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, disease in enumerate(DISEASES):
        ax = axes[idx]
        df = load_mimic_ci(disease)
        if df.empty:
            ax.set_visible(False)
            continue

        lr = _get_row(df, "lr_baseline")
        if lr is not None:
            ax.axhline(float(lr["auc_pr"]), color="#555", linewidth=1.3,
                       linestyle="--", alpha=0.65, zorder=2)

        present = [m for m in METHODS if _get_row(df, m) is not None]
        x = np.arange(len(present))

        for xi, m in enumerate(present):
            row = _get_row(df, m)
            auc = float(row["auc_pr"])
            ax.bar(xi, auc, 0.55, color=METHOD_COLORS[m], zorder=3,
                   edgecolor="white", linewidth=0.5)
            _draw_ci_bracket(ax, xi, row)
            ax.text(xi, auc + 0.0006, f"{auc:.3f}", ha="center", va="bottom",
                    fontsize=8, color=METHOD_COLORS[m], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in present],
                           fontsize=8.5, rotation=15, ha="right")
        ax.set_ylabel("AUC-PR", fontsize=10)
        ax.set_title(DISEASE_NAMES[disease], fontsize=13, fontweight="bold")
        ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
        ax.set_ylim(0, None)

    handles = _method_legend_handles()
    handles += [plt.Line2D([0], [0], color="#555", linewidth=1.4,
                           linestyle="--", label="LR Baseline (all features)")]
    handles += [plt.Line2D([0], [0], color="#222", linewidth=1.8,
                           label="95% Bootstrap CI")]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("AUC-PR per Method by Disease — MIMIC Test Set  (95% Bootstrap CI)",
                 fontsize=14, fontweight="bold", y=1.05, color="#1F397D")
    fig.tight_layout(pad=2.2)
    _savefig(fig, "figA_per_disease_aupr.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig B  –  All-diseases grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════

def figB_all_diseases():
    data = {m: [] for m in METHODS}
    for disease in DISEASES:
        df = load_mimic_ci(disease)
        for m in METHODS:
            row = _get_row(df, m) if not df.empty else None
            data[m].append(float(row["auc_pr"]) if row is not None else np.nan)

    x       = np.arange(len(DISEASES))
    width   = 0.14
    offsets = np.linspace(-(len(METHODS) - 1) / 2 * width,
                          (len(METHODS) - 1) / 2 * width, len(METHODS))

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, m in enumerate(METHODS):
        vals = data[m]
        xs = [x[j] + offsets[i] for j in range(len(DISEASES)) if not np.isnan(vals[j])]
        vv = [v for v in vals if not np.isnan(v)]
        ax.bar(xs, vv, width, label=METHOD_LABELS[m], color=METHOD_COLORS[m],
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([DISEASE_NAMES[d] for d in DISEASES], fontsize=13)
    ax.set_ylabel("AUC-PR", fontsize=12)
    ax.set_title("AUC-PR Across All 6 Diseases — MIMIC Test Set",
                 fontsize=14, fontweight="bold", color="#1F397D", pad=10)
    ax.legend(fontsize=9, ncol=5, loc="upper right", framealpha=0.85)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    _savefig(fig, "figB_all_diseases_aupr.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig C  –  Complexity heatmap: N_Features (color) + AUC-PR (cell text)
# ══════════════════════════════════════════════════════════════════════════════

def figC_complexity():
    comp = pd.read_csv(RES_DIR / "methods_comparison.csv")
    orig = comp[comp["Split_Salt"].isna() | (comp["Split_Salt"] == "")]

    n_feat = np.full((len(DISEASES), len(METHODS)), np.nan)
    auc_pr = np.full((len(DISEASES), len(METHODS)), np.nan)

    for i, slug in enumerate(DISEASES):
        row = orig[orig["Disease"] == slug]
        if row.empty:
            continue
        r = row.iloc[0]
        for j, m in enumerate(METHODS):
            f_col = f"{m.upper()}_N_Features"
            a_col = f"{m.upper()}_Best_AUC_PR"
            if f_col in r.index and pd.notna(r[f_col]):
                n_feat[i, j] = float(r[f_col])
            if a_col in r.index and pd.notna(r[a_col]):
                auc_pr[i, j] = float(r[a_col])

    fig, ax = plt.subplots(figsize=(12, 6))
    masked = np.ma.array(n_feat, mask=np.isnan(n_feat))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#EEEEEE")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=1, vmax=14)

    for i in range(len(DISEASES)):
        for j in range(len(METHODS)):
            auc = auc_pr[i, j]
            nf  = n_feat[i, j]
            if np.isnan(auc):
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#999")
            else:
                nf_str = f"{int(nf)}f" if not np.isnan(nf) else ""
                ax.text(j, i, f"{auc:.3f}\n({nf_str})",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if (not np.isnan(nf) and nf > 7) else "#222")

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticks(range(len(DISEASES)))
    ax.set_yticklabels([DISEASE_NAMES[d] for d in DISEASES], fontsize=12)
    ax.set_title("Formula Complexity — AUC-PR (MIMIC) with feature count\n"
                 "Color = # CBC features used  (lighter = simpler formula)",
                 fontsize=13, fontweight="bold", color="#1F397D", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("# Features Used", fontsize=10)
    cbar.set_ticks([1, 4, 7, 10, 14])
    fig.tight_layout()
    _savefig(fig, "figC_complexity.png")


# ══════════════════════════════════════════════════════════════════════════════
# Shared helper for MIMIC-vs-external panels
# ══════════════════════════════════════════════════════════════════════════════

def _draw_mimic_vs_external(ax, mimic_df, ext_df, disease, metric="auc_roc"):
    ci_lo = "AUC_ROC_CI_Low"  if metric == "auc_roc" else "AUC_PR_CI_Low"
    ci_hi = "AUC_ROC_CI_High" if metric == "auc_roc" else "AUC_PR_CI_High"

    present = [m for m in METHODS
               if _get_row(mimic_df, m) is not None or _get_row(ext_df, m) is not None]
    if not present:
        ax.set_visible(False)
        return

    x, width = np.arange(len(present)), 0.35

    for xi, m in enumerate(present):
        mrow = _get_row(mimic_df, m)
        erow = _get_row(ext_df,   m)
        col  = METHOD_COLORS[m]

        if mrow is not None and pd.notna(mrow.get(metric)):
            mv   = float(mrow[metric])
            merr = _safe_yerr(mrow, mv, ci_lo, ci_hi)
            ax.bar(x[xi] - width / 2, mv, width, color=col, zorder=3,
                   edgecolor="white", linewidth=0.5)
            if merr:
                ax.errorbar(x[xi] - width / 2, mv, yerr=merr, fmt="none",
                            color="#333", capsize=3, linewidth=1.1, zorder=4)

        if erow is not None and pd.notna(erow.get(metric)):
            ev   = float(erow[metric])
            eerr = _safe_yerr(erow, ev, ci_lo, ci_hi)
            ax.bar(x[xi] + width / 2, ev, width, color=col, zorder=3,
                   edgecolor="white", linewidth=0.5, hatch="//", alpha=0.75)
            if eerr:
                ax.errorbar(x[xi] + width / 2, ev, yerr=eerr, fmt="none",
                            color="#333", capsize=3, linewidth=1.1, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in present],
                       fontsize=8.5, rotation=18, ha="right")
    ax.set_ylabel("AUC-ROC" if metric == "auc_roc" else "AUC-PR", fontsize=9)
    ax.set_title(DISEASE_NAMES[disease], fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_ylim(0, None)


def _generalization_legend(ext_label):
    return [
        mpatches.Patch(color="#888", label="MIMIC (solid)"),
        mpatches.Patch(facecolor="#aaa", hatch="//", alpha=0.75,
                       edgecolor="#777", label=f"{ext_label} (hatched)"),
    ] + _method_legend_handles()


# ══════════════════════════════════════════════════════════════════════════════
# Fig D  –  EHRSHOT generalisation  (2×3 grid, AUC-ROC)
# ══════════════════════════════════════════════════════════════════════════════

def figD_ehrshot():
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()
    for idx, disease in enumerate(DISEASES):
        _draw_mimic_vs_external(axes[idx],
                                load_mimic_ci(disease),
                                load_ehrshot_ci(disease),
                                disease, metric="auc_roc")
    fig.legend(handles=_generalization_legend("EHRSHOT"),
               loc="upper center", ncol=7, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "Generalisation to EHRSHOT (Stanford EHR) — MIMIC vs EHRSHOT AUC-ROC  (95% CI)\n"
        "AUC-ROC is prevalence-independent — both cohorts on the same scale",
        fontsize=13, fontweight="bold", y=1.05, color="#1F397D")
    fig.tight_layout(pad=2.2)
    _savefig(fig, "figD_ehrshot_generalization.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig E  –  NHANES generalisation  (1×2, AUC-ROC)
# ══════════════════════════════════════════════════════════════════════════════

def figE_nhanes():
    nhanes_diseases = ["ra", "psr"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, disease in enumerate(nhanes_diseases):
        _draw_mimic_vs_external(axes[idx],
                                load_mimic_ci(disease),
                                load_nhanes_ci(disease),
                                disease, metric="auc_roc")
    fig.legend(handles=_generalization_legend("NHANES"),
               loc="upper center", ncol=7, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Generalisation to NHANES (US Population Survey) — AUC-ROC  (95% CI)\n"
        "RA and Psoriasis only",
        fontsize=13, fontweight="bold", y=1.07, color="#1F397D")
    fig.tight_layout(pad=2.2)
    _savefig(fig, "figE_nhanes_generalization.png")


# ══════════════════════════════════════════════════════════════════════════════
# Shared lift bar chart
# ══════════════════════════════════════════════════════════════════════════════

def _lift_chart(ax, lift_data, title):
    diseases = list(lift_data.keys())
    x        = np.arange(len(diseases))
    width    = 0.14
    offsets  = np.linspace(-(len(METHODS) - 1) / 2 * width,
                           (len(METHODS) - 1) / 2 * width, len(METHODS))
    Y_CAP = 8.0

    for i, m in enumerate(METHODS):
        vals_raw = [lift_data[d].get(m) for d in diseases]
        vals_cap = [min(v, Y_CAP) if v is not None else np.nan for v in vals_raw]
        xs   = [x[j] + offsets[i] for j in range(len(diseases)) if vals_raw[j] is not None]
        vc   = [vals_cap[j] for j in range(len(diseases)) if vals_raw[j] is not None]
        vr   = [vals_raw[j] for j in range(len(diseases)) if vals_raw[j] is not None]
        bars = ax.bar(xs, vc, width, label=METHOD_LABELS[m], color=METHOD_COLORS[m],
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, vraw in zip(bars, vr):
            s = f"{vraw:.2f}" if vraw <= Y_CAP else f"{vraw:.1f}↑"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(vraw, Y_CAP) + 0.12, s,
                    ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.axhline(1.0, color="#444", linewidth=1.2, linestyle="--", alpha=0.5,
               label="Lift = 1 (baseline)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=12)
    ax.set_ylabel("AUC-PR Lift  (formula / prevalence)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#1F397D", pad=10)
    ax.set_ylim(0, Y_CAP * 1.18)
    ax.legend(fontsize=9, ncol=3, loc="upper right", framealpha=0.85)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)


# ══════════════════════════════════════════════════════════════════════════════
# Fig F  –  MIMIC AUC-PR Lift
# ══════════════════════════════════════════════════════════════════════════════

def figF_mimic_lift():
    lift_data = {}
    for disease in DISEASES:
        prev = mimic_prevalence(disease)
        if not prev:
            continue
        df     = load_mimic_ci(disease)
        d_name = DISEASE_NAMES[disease]
        lift_data[d_name] = {
            m: float(_get_row(df, m)["auc_pr"]) / prev
            for m in METHODS
            if _get_row(df, m) is not None
        }

    if not lift_data:
        print("  [figF] no data — skipping")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    _lift_chart(ax, lift_data,
                "AUC-PR Lift Across 6 Diseases — MIMIC Test Set\n"
                "(AUC-PR / disease prevalence, y-axis capped at 8×)")
    fig.tight_layout()
    _savefig(fig, "figF_mimic_lift.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig G  –  EHRSHOT AUC-PR Lift
# ══════════════════════════════════════════════════════════════════════════════

def figG_ehrshot_lift():
    lift_data = {}
    for disease in DISEASES:
        prev = ehrshot_prevalence(disease)
        if not prev:
            continue
        df     = load_ehrshot_ci(disease)
        d_name = DISEASE_NAMES[disease]
        lift_data[d_name] = {
            m: float(_get_row(df, m)["auc_pr"]) / prev
            for m in METHODS
            if _get_row(df, m) is not None
        }

    if not lift_data:
        print("  [figG] no data — skipping")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    _lift_chart(ax, lift_data,
                "AUC-PR Lift Across Diseases — EHRSHOT (Stanford EHR)\n"
                "(AUC-PR / EHRSHOT cohort prevalence, y-axis capped at 8×)")
    fig.tight_layout()
    _savefig(fig, "figG_ehrshot_lift.png")


# ══════════════════════════════════════════════════════════════════════════════

def refresh_ci():
    python = sys.executable
    for disease in DISEASES:
        print(f"\n  Refreshing CIs for {disease} ...")
        subprocess.run(
            [python, "src/mimic_compute_ci.py", "--disease", disease],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-ci", action="store_true",
                        help="Re-run bootstrap CIs before generating figures (slow)")
    args = parser.parse_args()

    if args.refresh_ci:
        print("=== Refreshing CI files ===")
        refresh_ci()

    print(f"\nExporting figures to {OUT_DIR.resolve()}/\n")
    figA_per_disease()
    figA_grid()
    figB_all_diseases()
    figC_complexity()
    figD_ehrshot()
    figE_nhanes()
    figF_mimic_lift()
    figG_ehrshot_lift()
    print("\nDone.")
