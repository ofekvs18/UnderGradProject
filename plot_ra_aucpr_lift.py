import re
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from utils import load_data_for, get_splits, load_disease_config, RESULTS_DIR

# ── Complexity helpers ────────────────────────────────────────────────────────
FEATURES = [
    "hct", "hgb", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc",
    "neut_pct", "lym_pct", "mono_pct", "eos_pct", "baso_pct",
]

def _is_gp(formula: str) -> bool:
    return bool(re.match(r"^\s*(mul|div|add|sub|neg|sqrt|log|abs)\s*\(", formula))

def strip_stability_constants(formula: str) -> str:
    """Remove tiny additive stability constants used to prevent div-by-zero."""
    f = re.sub(r"\s*\+\s*\d*\.?\d+[eE]-\d+", "", formula)  # +1e-6, +2.5e-8 …
    f = re.sub(r"\s*\+\s*0\.0+\d*\b", "", f)                 # +0.01, +0.001 …
    return f

def count_features(formula: str) -> int:
    return len({f for f in FEATURES if re.search(r"\b" + f + r"\b", formula)})

def count_ops(formula: str) -> int:
    formula = strip_stability_constants(formula)
    if _is_gp(formula):
        return len(re.findall(r"\b(mul|div|add|sub|neg|sqrt|log|abs)\s*\(", formula))
    ops = 0
    ops += len(re.findall(r"\*\*", formula))
    tmp = re.sub(r"\*\*", "PP", formula)
    ops += tmp.count("*")
    ops += formula.count("/")
    ops += formula.count("+")
    ops += len(re.findall(r"(?<=[0-9a-zA-Z_)])\s*-", formula))
    ops += len(re.findall(r"\b(sqrt|abs|log)\b", formula))
    ops += len(re.findall(r"[<>]=?", formula))
    return ops

def lr_complexity(formula: str):
    """Parse logit formula string → (n_nonzero_features, n_nonzero_features)."""
    terms = re.findall(r"\((\d+\.\d+)\s*\*\s*(\w+)\)", formula)
    nonzero = [(float(c), f) for c, f in terms if float(c) != 0.0]
    return len(nonzero), len(nonzero)   # (n_features, n_ops) = same for LR

# ── Load data & prevalence ────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    disease = load_disease_config("ra")
    df_data, _ = load_data_for(disease.name)
_, test_df = get_splits(df_data)
prevalence = test_df["is_case"].mean()

ci_df = pd.read_csv("results/figures/ra_ci_data.csv")
san_df = pd.read_csv(str(RESULTS_DIR / "sanity_check" / "master_sanity_summary.csv"))
lr_row = san_df[san_df["Disease"] == "ra"].sort_values("Timestamp").iloc[-1]
lr_formula = str(lr_row["All_Feat_Formula"])

# ── Build per-method records ──────────────────────────────────────────────────
KEYS   = ["m1", "m2", "m3", "m4"]
LABELS = ["M1\nThreshold", "M2\nRandom", "M3\nGP", "M4\nLLM"]
COLORS = ["#F59E0B", "#10B981", "#0D9488", "#8B5CF6"]
DARKER = ["#B45309", "#059669", "#0F766E", "#6D28D9"]
NICE   = ["M1: Threshold", "M2: Random", "M3: GP", "M4: LLM"]

aucprs, ci_lows, ci_highs, formulas = [], [], [], []
for key in KEYS:
    row = ci_df[ci_df["method"] == key].iloc[0]
    aucprs.append(float(row["auc_pr"]))
    ci_lows.append(float(row["AUC_PR_CI_Low"]) if pd.notna(row["AUC_PR_CI_Low"]) else np.nan)
    ci_highs.append(float(row["AUC_PR_CI_High"]) if pd.notna(row["AUC_PR_CI_High"]) else np.nan)
    formulas.append(str(row.get("formula", "")))

lr_aucpr = float(ci_df[ci_df["method"] == "lr_baseline"].iloc[0]["auc_pr"])

n_feat = [count_features(f) for f in formulas]
n_ops  = [count_ops(f) for f in formulas]

lifts      = [v / prevalence for v in aucprs]
ci_lo_lift = [v / prevalence if not np.isnan(v) else np.nan for v in ci_lows]
ci_hi_lift = [v / prevalence if not np.isnan(v) else np.nan for v in ci_highs]
lr_lift    = lr_aucpr / prevalence

# ── Export CSV ────────────────────────────────────────────────────────────────
lr_nf, lr_no = lr_complexity(lr_formula)

csv_rows = []
for i, key in enumerate(KEYS):
    csv_rows.append({
        "method": key,
        "label": NICE[i],
        "formula": formulas[i],
        "n_features": n_feat[i],
        "n_ops": n_ops[i],
        "auc_pr": round(aucprs[i], 4),
        "lift": round(lifts[i], 3),
        "auc_pr_ci_low":  round(ci_lows[i],  4) if not np.isnan(ci_lows[i])  else "",
        "auc_pr_ci_high": round(ci_highs[i], 4) if not np.isnan(ci_highs[i]) else "",
        "lift_ci_low":  round(ci_lo_lift[i], 3) if not np.isnan(ci_lo_lift[i]) else "",
        "lift_ci_high": round(ci_hi_lift[i], 3) if not np.isnan(ci_hi_lift[i]) else "",
    })
csv_rows.append({
    "method": "lr_baseline",
    "label": "LR Baseline",
    "formula": lr_formula,
    "n_features": lr_nf,
    "n_ops": lr_no,
    "auc_pr": round(lr_aucpr, 4),
    "lift": round(lr_lift, 3),
    "auc_pr_ci_low": "", "auc_pr_ci_high": "",
    "lift_ci_low": "", "lift_ci_high": "",
})

csv_out = "results/ra_complexity.csv"
pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
print(f"CSV: {csv_out}")
for r in csv_rows:
    print(f"  {r['label']:20s}  {r['n_features']} feat. | {r['n_ops']} ops")

# ── Plot ──────────────────────────────────────────────────────────────────────
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
        fmt="none", ecolor=DARKER[i],
        elinewidth=1.5, capsize=6, capthick=1.5, zorder=4,
    )

for i, (bar, lift) in enumerate(zip(bars, lifts)):
    top = ci_hi_lift[i] if not np.isnan(ci_hi_lift[i]) else lift
    ax.text(bar.get_x() + bar.get_width() / 2,
            top + (y_max - y_min) * 0.02,
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

trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for i in range(len(lifts)):
    ax.text(x[i], -0.22, f"{n_feat[i]} feat.",
            transform=trans, ha="center", va="top",
            fontsize=8, color="#94A3B8")
    ax.text(x[i], -0.30, f"{n_ops[i]} ops",
            transform=trans, ha="center", va="top",
            fontsize=8, color="#94A3B8")

fig.text(0.5, 0.96, "Rheumatoid Arthritis — AUC-PR Lift by Method",
         ha="center", va="top", fontsize=17, fontweight="bold", color="#0F172A")
fig.text(0.5, 0.905,
         "Lift = AUC-PR / prevalence  |  prevalence = 1.03%  |  dashed = LR baseline",
         ha="center", va="top", fontsize=11, color="#64748B")

plt.tight_layout(rect=[0, 0.10, 1, 0.90])

out = "results/ra_aucpr_lift.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="none", transparent=True)
print(f"Chart: {out}")
