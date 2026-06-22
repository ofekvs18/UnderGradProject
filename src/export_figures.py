"""
Export all presentation figures as PNG files for review.

Produces docs/figures/:
  fig1_mimic_lift.png        – AUC-PR lift across 6 diseases, all 5 methods
  fig2_ehrshot_table.png     – EHRSHOT lift table (M1–M5)
  fig3_complexity.png        – M2 vs M5: MIMIC vs EHRSHOT (generalisation story)
  fig4_nhanes.png            – NHANES AUC-PR for RA + Psoriasis (M1–M5)

Run:  python src/export_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────

# MIMIC: formula_auc_pr / disease_prevalence
MIMIC_LIFTS = {
    "RA":        {"M1": 1.32, "M2": 3.59, "M3": 1.45, "M4": 1.39, "M5": 1.52},
    "Crohn":     {"M1": 2.38, "M2": 4.61, "M3": 3.20, "M4": 2.39, "M5": 2.65},
    "T1D":       {"M1": 1.43, "M2": 3.57, "M3": 1.19, "M4": 1.22, "M5": 1.68},
    "T2D":       {"M1": 1.19, "M2": 1.22, "M3": 1.26, "M4": 1.12},
    "Psoriasis": {"M1": 2.35, "M2": 7.02, "M3": 1.28, "M4": 2.40, "M5": 1.28},
    "Lupus":     {"M1": 2.03, "M2": 11.36, "M3": 1.54, "M4": 3.56, "M5": 1.05},
}

# EHRSHOT: best_auc_pr / ehrshot_prevalence
EHRSHOT_LIFTS = {
    "RA":        {"M1": 2.04, "M2": 2.19, "M3": 2.56, "M4": 2.53, "M5": 2.94},
    "Crohn":     {"M1": 1.21, "M2": 5.13, "M3": 5.28, "M4": 3.29, "M5": 5.35},
    "T1D":       {"M1": 1.41, "M2": 2.87, "M3": 3.02, "M4": 2.64, "M5": 3.07},
    "T2D":       {"M1": 1.10, "M2": 1.21, "M3": 1.21, "M4": 1.09},
    "Psoriasis": {"M1": 1.14, "M2": 1.58, "M3": 1.44, "M4": 1.48, "M5": 1.44},
    "Lupus":     {"M1": 1.36, "M2": 1.73, "M3": 2.37, "M4": 1.81, "M5": 1.84},
}

# NHANES: best AUC-PR per method (RA + Psoriasis only)
NHANES_AUC_PR = {
    "RA":        {"M1": 0.0882, "M2": 0.1212, "M3": 0.1224, "M4": 0.1442, "M5": 0.1224},
    "Psoriasis": {"M1": 0.0247, "M2": 0.0334, "M3": 0.0382, "M4": 0.0354, "M5": 0.0382},
}

# n features used by best formula (original split)
N_FEATURES = {
    "RA":        {"M2": 9,  "M3": 4, "M4": 3, "M5": 4},
    "Crohn":     {"M2": 13, "M3": 4, "M4": 3, "M5": 4},
    "T1D":       {"M2": 6,  "M3": 5, "M4": 4, "M5": 9},
    "T2D":       {"M2": 13, "M3": 5, "M4": 3},
    "Psoriasis": {"M2": 4,  "M3": 3, "M4": 4, "M5": 3},
    "Lupus":     {"M2": 8,  "M3": 5, "M4": 3, "M5": 6},
}

# ── Shared style ─────────────────────────────────────────────────────────────
METHODS   = ["M1", "M2", "M3", "M4", "M5"]
LABELS    = ["B1 Threshold", "M2 Random", "M3 GP", "M4 LLM", "M5 Seeded GP ★"]
COLORS    = ["#9E9E9E", "#429BF4", "#34A853", "#FBBC05", "#0B6027"]
COL_MAP   = dict(zip(METHODS, COLORS))
LABEL_MAP = dict(zip(METHODS, LABELS))

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "white",
})


def _savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 1 – MIMIC AUC-PR Lift (all 6 diseases, M1–M5, y-axis capped at 8×)
# ════════════════════════════════════════════════════════════════════════════════

def fig1_mimic_lift():
    diseases = list(MIMIC_LIFTS.keys())
    n_d = len(diseases)
    x = np.arange(n_d)
    n_m = len(METHODS)
    width = 0.14
    offsets = np.linspace(-(n_m - 1) / 2 * width, (n_m - 1) / 2 * width, n_m)
    Y_CAP = 8.0

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, m in enumerate(METHODS):
        vals_raw = [MIMIC_LIFTS[d].get(m) for d in diseases]
        vals_cap = [min(v, Y_CAP) if v is not None else np.nan for v in vals_raw]
        xs  = [x[j] + offsets[i] for j in range(n_d) if vals_raw[j] is not None]
        vc  = [vals_cap[j] for j in range(n_d) if vals_raw[j] is not None]
        vr  = [vals_raw[j] for j in range(n_d) if vals_raw[j] is not None]
        bars = ax.bar(xs, vc, width, label=LABEL_MAP[m], color=COL_MAP[m],
                      edgecolor="white", linewidth=0.6, zorder=3)
        bold = m in ("M3", "M5")
        for bar, vraw in zip(bars, vr):
            label_str = f"{vraw:.2f}" if vraw <= Y_CAP else f"{vraw:.1f}↑"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(vraw, Y_CAP) + 0.12,
                    label_str, ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold" if bold else "normal",
                    color=COL_MAP[m] if m in ("M3", "M5") else "#333333")

    ax.axhline(1.0, color="#444", linewidth=1.2, linestyle="--", alpha=0.55, label="Lift = 1 (random)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=13)
    ax.set_ylabel("AUC-PR Lift  (formula / prevalence)", fontsize=12)
    ax.set_title("AUC-PR Lift Across All 6 Diseases — MIMIC Test Set\n"
                 f"Y-axis capped at {Y_CAP:.0f}×  (M2 Lupus = 11.36×,  M2 Psoriasis = 7.02×)",
                 fontsize=13, pad=10)
    ax.set_ylim(0, Y_CAP * 1.18)
    ax.legend(fontsize=9, ncol=3, loc="upper right", framealpha=0.85)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    _savefig(fig, "fig1_mimic_lift.png")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 2 – EHRSHOT AUC-PR Lift table
# ════════════════════════════════════════════════════════════════════════════════

def fig2_ehrshot_table():
    diseases   = list(EHRSHOT_LIFTS.keys())
    col_labels = ["Disease"] + [LABEL_MAP[m] for m in METHODS]

    rows = []
    for d in diseases:
        row = [d]
        for m in METHODS:
            v = EHRSHOT_LIFTS[d].get(m)
            row.append(f"{v:.2f}×" if v is not None else "—")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.65)

    # Header row
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#1F397D")
        cell.set_text_props(color="white", fontweight="bold")
        if j > 0:
            cell.set_facecolor(COL_MAP[METHODS[j - 1]])
            cell.set_text_props(color="white", fontweight="bold")

    # Colour best per row + dim missing
    for i, d in enumerate(diseases):
        vals = {m: EHRSHOT_LIFTS[d].get(m, 0.0) for m in METHODS}
        best_m = max(vals, key=vals.get)
        for j, m in enumerate(METHODS):
            cell = tbl[i + 1, j + 1]
            if vals[m] == 0.0:
                cell.set_facecolor("#F5F5F5")
                cell.set_text_props(color="#AAAAAA")
            elif m == best_m:
                cell.set_facecolor("#C8E6C9")
                cell.set_text_props(fontweight="bold")
            elif m == "M5":
                cell.set_facecolor("#E8F5E9")

    ax.set_title("EHRSHOT (Stanford EHR) — AUC-PR Lift  (AUC-PR / cohort prevalence)",
                 fontsize=12, pad=14, fontweight="bold", color="#1F397D")
    fig.tight_layout(pad=1.5)
    _savefig(fig, "fig2_ehrshot_table.png")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 3 – Complexity: M2 vs M5  (MIMIC left, EHRSHOT right)
# ════════════════════════════════════════════════════════════════════════════════

def fig3_complexity():
    diseases_5 = [d for d in MIMIC_LIFTS if "M5" in MIMIC_LIFTS[d]]  # no T2D
    x = np.arange(len(diseases_5))
    width = 0.32
    Y_CAP = 8.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    panels = [
        (axes[0], {d: MIMIC_LIFTS[d]   for d in diseases_5}, "MIMIC Test Set",
         "(training domain — capped at 8×)"),
        (axes[1], {d: EHRSHOT_LIFTS[d] for d in diseases_5}, "EHRSHOT (Stanford)",
         "(external, zero-shot)"),
    ]

    for ax, lifts_dict, title, subtitle in panels:
        m2 = [lifts_dict[d]["M2"] for d in diseases_5]
        m5 = [lifts_dict[d]["M5"] for d in diseases_5]
        m2c = [min(v, Y_CAP) for v in m2]
        m5c = [min(v, Y_CAP) for v in m5]

        b2 = ax.bar(x - width / 2, m2c, width, color="#429BF4", label="M2 Random", zorder=3)
        b5 = ax.bar(x + width / 2, m5c, width, color="#0B6027", label="M5 Seeded GP ★", zorder=3)

        for bar, vraw in zip(b2, m2):
            s = f"{vraw:.2f}" if vraw <= Y_CAP else f"{vraw:.1f}↑"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(vraw, Y_CAP) + 0.10,
                    s, ha="center", va="bottom", fontsize=8.5)
        for bar, v in zip(b5, m5):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.10, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#0B6027")

        ax.axhline(1.0, color="#444", linewidth=1.1, linestyle="--", alpha=0.5, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(diseases_5, fontsize=12)
        ax.set_ylabel("AUC-PR Lift  (×)", fontsize=11)
        ymax = max(max(m2c), max(m5c)) * 1.22
        ax.set_ylim(0, min(ymax, Y_CAP * 1.18))
        ax.legend(fontsize=10, loc="upper right", framealpha=0.85)
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=8)
        ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)

    # Feature count annotation
    feat_note = "Features used:  " + "   |   ".join(
        f"{d}: M2={N_FEATURES[d]['M2']}, M5={N_FEATURES[d]['M5']}"
        for d in diseases_5
    )
    fig.text(0.5, -0.04, feat_note, ha="center", fontsize=9,
             color="#555555", style="italic")

    fig.suptitle("Why M5?  M2 Overfits — M5 Generalises",
                 fontsize=14, fontweight="bold", color="#1F397D", y=1.03)
    fig.tight_layout(pad=1.8)
    _savefig(fig, "fig3_complexity.png")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 4 – NHANES AUC-PR (RA + Psoriasis, M1–M5)
# ════════════════════════════════════════════════════════════════════════════════

def fig4_nhanes():
    diseases = list(NHANES_AUC_PR.keys())
    n_d = len(diseases)
    x = np.arange(n_d)
    n_m = len(METHODS)
    width = 0.14
    offsets = np.linspace(-(n_m - 1) / 2 * width, (n_m - 1) / 2 * width, n_m)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, m in enumerate(METHODS):
        vals = [NHANES_AUC_PR[d].get(m, np.nan) for d in diseases]
        xs   = [x[j] + offsets[i] for j in range(n_d) if not np.isnan(vals[j])]
        vv   = [v for v in vals if not np.isnan(v)]
        bars = ax.bar(xs, vv, width, label=LABEL_MAP[m], color=COL_MAP[m],
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, v in zip(bars, vv):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold" if m in ("M3", "M5") else "normal",
                    color=COL_MAP[m] if m in ("M3", "M5") else "#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=13)
    ax.set_ylabel("AUC-PR", fontsize=12)
    ax.set_title("NHANES (US population survey) — AUC-PR\n"
                 "Only RA and Psoriasis available  ·  lift not computed (prevalence unknown)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9, ncol=3, loc="upper right", framealpha=0.85)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    fig.tight_layout()
    _savefig(fig, "fig4_nhanes.png")


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Exporting figures to {OUT_DIR.resolve()}/\n")
    fig1_mimic_lift()
    fig2_ehrshot_table()
    fig3_complexity()
    fig4_nhanes()
    print("\nAll figures exported.")
