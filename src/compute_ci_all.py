"""
compute_ci_all.py — Bootstrap 95% CIs for the best formula from each method.

Reads master summaries for M1–M4, evaluates each best formula on the test set
with bootstrap resampling, and writes:
  - results/method3_gp/{disease}/master_summary.csv  (M3 rows + CI columns)
  - results/figures/{disease}_ci_data.csv            (one row per method, for forest plot)

Usage:
    python src/compute_ci_all.py --disease ra
"""

import argparse
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from utils import (
    RESULTS_DIR,
    ensure_dir,
    eval_formula_scores,
    get_splits,
    load_data_for,
    load_disease_config,
)

N_BOOTSTRAP = 500
CI_LO = 2.5
CI_HI = 97.5
SEED = 42


# ── gplearn S-expression evaluator ────────────────────────────────────────────

def _split_top_args(s):
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


def eval_gp_sexpr(expr, df, features):
    """Recursively evaluate a gplearn S-expression against a DataFrame row-set."""
    expr = expr.strip()

    if expr in features:
        return df[expr].values.astype(float)

    try:
        val = float(expr)
        return np.full(len(df), val)
    except ValueError:
        pass

    m = re.match(r"^(\w+)\((.*)\)$", expr, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse S-expr token: {expr[:60]!r}")

    fname, args_str = m.group(1), m.group(2)
    args = [eval_gp_sexpr(a, df, features) for a in _split_top_args(args_str)]

    SAFE_DIV = 1e-8
    if fname == "add":
        return args[0] + args[1]
    if fname == "sub":
        return args[0] - args[1]
    if fname == "mul":
        return args[0] * args[1]
    if fname == "div":
        denom = np.where(np.abs(args[1]) < SAFE_DIV, SAFE_DIV, args[1])
        return args[0] / denom
    if fname == "neg":
        return -args[0]
    if fname == "sqrt":
        return np.sqrt(np.abs(args[0]))
    if fname == "log":
        return np.log1p(np.abs(args[0]))
    if fname == "abs":
        return np.abs(args[0])
    raise ValueError(f"Unknown gplearn function: {fname}")


def is_gp_sexpr(formula: str) -> bool:
    return bool(re.match(r"^\s*(mul|div|add|sub|neg|sqrt|log|abs)\s*\(", formula))


# ── Score computation helpers ─────────────────────────────────────────────────

def get_scores(formula, test_df, features):
    """Return (scores, y_true) or (None, None). Flips scores if AUC-ROC < 0.5."""
    te = test_df[features + ["is_case"]].dropna()
    y = te["is_case"].values

    if is_gp_sexpr(formula):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = eval_gp_sexpr(formula, te, features).astype(float)
        except Exception as exc:
            print(f"  [WARN] S-expr eval failed: {exc}")
            return None, None
    else:
        scores = eval_formula_scores(formula, te, features)
        if scores is None:
            return None, None

    bad = ~np.isfinite(scores)
    if bad.mean() > 0.10:
        return None, None
    if bad.any():
        scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0

    try:
        auc_roc = float(roc_auc_score(y, scores))
        if auc_roc < 0.5:
            scores = -scores
    except Exception:
        pass

    return scores, y


def bootstrap_ci(y_true, scores, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Percentile bootstrap 95% CI for AUC-PR and AUC-ROC.
    Returns (pr_lo, pr_hi, roc_lo, roc_hi) or None if too few valid samples.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_prs, auc_rocs = [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb, sb = y_true[idx], scores[idx]
        if yb.sum() < 2 or (1 - yb).sum() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc_prs.append(float(average_precision_score(yb, sb)))
                auc_rocs.append(float(roc_auc_score(yb, sb)))
        except Exception:
            pass

    if len(auc_prs) < 20:
        return None

    return (
        float(np.percentile(auc_prs, CI_LO)),
        float(np.percentile(auc_prs, CI_HI)),
        float(np.percentile(auc_rocs, CI_LO)),
        float(np.percentile(auc_rocs, CI_HI)),
    )


# ── Per-method formula extraction ─────────────────────────────────────────────

def get_m1_best(disease_name):
    path = RESULTS_DIR / "method1_threshold" / "master_m1_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    dis = df[df["Disease"] == disease_name]
    if dis.empty:
        return None
    r = dis.sort_values("Best_DD_AUC_PR", ascending=False).iloc[0]
    return {
        "method": "m1", "label": "M1: Threshold",
        "formula": str(r.get("Best_DD_Formula", "")),
        "auc_pr": float(r["Best_DD_AUC_PR"]),
        "auc_roc": float(r["Best_DD_AUC_ROC"]),
    }


def get_m2_best(disease_name):
    path = RESULTS_DIR / "method2_random" / "master_m2_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    dis = df[df["Disease"] == disease_name]
    if dis.empty:
        return None
    r = dis.sort_values("Best_Random_AUC_PR", ascending=False).iloc[0]
    return {
        "method": "m2", "label": "M2: Random Search",
        "formula": str(r["Best_Random_Formula"]),
        "auc_pr": float(r["Best_Random_AUC_PR"]),
        "auc_roc": float(r["Best_Random_AUC_ROC"]),
    }


def get_m4_best(disease_name):
    path = RESULTS_DIR / "method4_llm" / "method4_master_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    dis = df[df["Disease"] == disease_name]
    if dis.empty:
        return None
    r = dis.sort_values("Best_LLM_AUC_PR", ascending=False).iloc[0]
    return {
        "method": "m4", "label": "M4: LLM",
        "formula": str(r["Best_LLM_Formula"]),
        "auc_pr": float(r["Best_LLM_AUC_PR"]),
        "auc_roc": float(r["Best_LLM_AUC_ROC"]),
    }


def get_lr_baseline(disease_name):
    path = RESULTS_DIR / "sanity_check" / "master_sanity_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    dis = df[df["Disease"] == disease_name]
    if dis.empty:
        return None
    r = dis.sort_values("Timestamp", ascending=False).iloc[0]
    return float(r["All_Feat_AUC_PR"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for all method best formulas")
    parser.add_argument("--disease", default="ra")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    df, features = load_data_for(disease.name)
    _, test_df = get_splits(df)

    n_pos = int(test_df["is_case"].sum())
    print(f"Disease: {disease.name}")
    print(f"Test set: {len(test_df)} rows, {n_pos} positives  (prevalence={n_pos/len(test_df):.3%})")
    print(f"Bootstrap iterations: {args.n_bootstrap}")

    ci_rows = []

    # ── M1 ────────────────────────────────────────────────────────────────────
    m1 = get_m1_best(disease.name)
    if m1:
        print(f"\n[m1] formula: {m1['formula']}")
        scores, y = get_scores(m1["formula"], test_df, features)
        ci = bootstrap_ci(y, scores, n_bootstrap=args.n_bootstrap) if scores is not None else None
        row = {**m1}
        if ci:
            row.update({
                "AUC_PR_CI_Low": round(ci[0], 4), "AUC_PR_CI_High": round(ci[1], 4),
                "AUC_ROC_CI_Low": round(ci[2], 4), "AUC_ROC_CI_High": round(ci[3], 4),
            })
            print(f"[m1] AUC-PR={m1['auc_pr']:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            print(f"[m1] CI skipped (formula not evaluable or too few bootstrap samples)")
        ci_rows.append(row)

    # ── M2 ────────────────────────────────────────────────────────────────────
    m2 = get_m2_best(disease.name)
    if m2:
        print(f"\n[m2] formula: {m2['formula'][:80]}")
        scores, y = get_scores(m2["formula"], test_df, features)
        ci = bootstrap_ci(y, scores, n_bootstrap=args.n_bootstrap) if scores is not None else None
        row = {**m2}
        if ci:
            row.update({
                "AUC_PR_CI_Low": round(ci[0], 4), "AUC_PR_CI_High": round(ci[1], 4),
                "AUC_ROC_CI_Low": round(ci[2], 4), "AUC_ROC_CI_High": round(ci[3], 4),
            })
            print(f"[m2] AUC-PR={m2['auc_pr']:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            print(f"[m2] CI skipped")
        ci_rows.append(row)

    # ── M3 ────────────────────────────────────────────────────────────────────
    m3_master = RESULTS_DIR / "method3_gp" / "master_gp_summary.csv"
    m3_out_dir = RESULTS_DIR / "method3_gp" / disease.name
    ensure_dir(m3_out_dir)

    if m3_master.exists():
        m3_df = pd.read_csv(m3_master)
        m3_dis = m3_df[m3_df["Disease"] == disease.name].copy().reset_index(drop=True)

        m3_dis["auc_pr"] = m3_dis["Best_GP_AUC_PR"]
        m3_dis["auc_roc"] = m3_dis["Best_GP_AUC_ROC"]
        m3_dis["AUC_PR_CI_Low"] = np.nan
        m3_dis["AUC_PR_CI_High"] = np.nan
        m3_dis["AUC_ROC_CI_Low"] = np.nan
        m3_dis["AUC_ROC_CI_High"] = np.nan

        print(f"\n[m3] {len(m3_dis)} tier row(s) found for {disease.name}")
        best_m3 = None

        for i, row in m3_dis.iterrows():
            formula_s = str(row["Best_GP_Formula"])
            tier = row.get("Config_Used", f"row{i}")
            print(f"  [{tier}] formula length: {len(formula_s)} chars")
            scores, y = get_scores(formula_s, test_df, features)
            if scores is None:
                print(f"  [{tier}] evaluation failed — CI set to NaN")
                continue
            ci = bootstrap_ci(y, scores, n_bootstrap=args.n_bootstrap)
            if ci is None:
                print(f"  [{tier}] bootstrap failed — too few valid samples")
                continue

            # Re-evaluate point estimate with our evaluator for consistency with bootstrap CI.
            # gplearn's program.execute() may differ slightly from our S-expr evaluator for
            # complex formulae, which can put the stored point estimate outside the CI bounds.
            from sklearn.metrics import average_precision_score as _aps, roc_auc_score as _roc
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eval_auc_pr = round(float(_aps(y, scores)), 4)
                    eval_auc_roc = round(float(_roc(y, scores)), 4)
            except Exception:
                eval_auc_pr = round(float(row["Best_GP_AUC_PR"]), 4)
                eval_auc_roc = round(float(row["Best_GP_AUC_ROC"]), 4)

            m3_dis.at[i, "auc_pr"] = eval_auc_pr
            m3_dis.at[i, "auc_roc"] = eval_auc_roc
            m3_dis.at[i, "AUC_PR_CI_Low"] = round(ci[0], 4)
            m3_dis.at[i, "AUC_PR_CI_High"] = round(ci[1], 4)
            m3_dis.at[i, "AUC_ROC_CI_Low"] = round(ci[2], 4)
            m3_dis.at[i, "AUC_ROC_CI_High"] = round(ci[3], 4)
            print(f"  [{tier}] AUC-PR={eval_auc_pr:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

            if best_m3 is None or eval_auc_pr > best_m3["auc_pr"]:
                best_m3 = {
                    "method": "m3", "label": "M3: Genetic Programming",
                    "formula": formula_s,
                    "auc_pr": eval_auc_pr,
                    "auc_roc": eval_auc_roc,
                    "AUC_PR_CI_Low": round(ci[0], 4), "AUC_PR_CI_High": round(ci[1], 4),
                    "AUC_ROC_CI_Low": round(ci[2], 4), "AUC_ROC_CI_High": round(ci[3], 4),
                }

        m3_out = m3_out_dir / "master_summary.csv"
        m3_dis.to_csv(m3_out, index=False)
        print(f"  Wrote M3 per-disease summary: {m3_out}")

        if best_m3:
            ci_rows.append(best_m3)
    else:
        print(f"[m3] master_gp_summary.csv not found")

    # ── M4 ────────────────────────────────────────────────────────────────────
    m4 = get_m4_best(disease.name)
    if m4:
        print(f"\n[m4] formula: {m4['formula'][:80]}")
        scores, y = get_scores(m4["formula"], test_df, features)
        ci = bootstrap_ci(y, scores, n_bootstrap=args.n_bootstrap) if scores is not None else None
        row = {**m4}
        if ci:
            row.update({
                "AUC_PR_CI_Low": round(ci[0], 4), "AUC_PR_CI_High": round(ci[1], 4),
                "AUC_ROC_CI_Low": round(ci[2], 4), "AUC_ROC_CI_High": round(ci[3], 4),
            })
            print(f"[m4] AUC-PR={m4['auc_pr']:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            print(f"[m4] CI skipped")
        ci_rows.append(row)

    # ── LR baseline ───────────────────────────────────────────────────────────
    lr_auc_pr = get_lr_baseline(disease.name)
    if lr_auc_pr is not None:
        print(f"\n[lr] All-features LR baseline AUC-PR: {lr_auc_pr:.4f}")
        ci_rows.append({
            "method": "lr_baseline", "label": "LR Baseline (all features)",
            "formula": "", "auc_pr": round(lr_auc_pr, 4), "auc_roc": float("nan"),
        })

    # ── Write combined CI CSV for forest plot ─────────────────────────────────
    fig_dir = RESULTS_DIR / "figures"
    ensure_dir(fig_dir)
    ci_path = fig_dir / f"{disease.name}_ci_data.csv"
    pd.DataFrame(ci_rows).to_csv(ci_path, index=False)
    print(f"\nWrote combined CI data: {ci_path}")
    print(pd.DataFrame(ci_rows)[
        ["method", "auc_pr", "AUC_PR_CI_Low", "AUC_PR_CI_High"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
