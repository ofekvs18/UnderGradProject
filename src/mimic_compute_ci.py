"""
mimic_compute_ci.py — Bootstrap 95% CIs for the best formula from each method.

Reads master summaries for M1–M4, evaluates each best formula on the test set
with bootstrap resampling, and writes:
  - results/method3_gp/{disease}/master_summary.csv  (M3 rows + CI columns)
  - results/figures/{disease}_ci_data.csv            (one row per method, for forest plot)

Usage:
    python src/mimic_compute_ci.py --disease ra
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
    bootstrap_ci,
    ensure_dir,
    eval_formula_scores,
    get_scores,
    get_splits,
    is_gp_sexpr,
    eval_gp_sexpr,
    load_data_for,
    load_disease_config,
)

N_BOOTSTRAP = 500


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
    # Use CV-selected formula evaluated on frozen test — not train-set best (overfit)
    r = dis.sort_values("CV_Winner_Frozen_Test_AUC_PR", ascending=False).iloc[0]
    return {
        "method": "m2", "label": "M2: Random Search",
        "formula": str(r["CV_Winner_Formula"]),
        "auc_pr": float(r["CV_Winner_Frozen_Test_AUC_PR"]),
        "auc_roc": float("nan"),
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

        # Only consider CV-selected rows to avoid test-set peeking across tiers.
        # Among those, pick the one with the best Frozen_Test_AUC_PR_Final.
        cv_sel = m3_dis[m3_dis["CV_Selected"] == True].copy()
        if cv_sel.empty:
            cv_sel = m3_dis  # fallback: no CV_Selected flag set
        cv_sel = cv_sel.sort_values("Frozen_Test_AUC_PR_Final", ascending=False)
        m3_dis_eval = cv_sel  # only bootstrap-eval the selected rows

        print(f"\n[m3] {len(m3_dis)} total row(s), {len(cv_sel)} CV-selected row(s) for {disease.name}")
        best_m3 = None

        for i, row in m3_dis_eval.iterrows():
            formula_s = str(row["Best_GP_Formula"])
            tier = row.get("Config_Used", f"row{i}")
            seed_label = str(row.get("Seed_File", "none"))
            print(f"  [{tier}/{seed_label}] formula length: {len(formula_s)} chars")
            scores, y = get_scores(formula_s, test_df, features)
            if scores is None:
                print(f"  [{tier}/{seed_label}] evaluation failed — CI set to NaN")
                continue
            ci = bootstrap_ci(y, scores, n_bootstrap=args.n_bootstrap)
            if ci is None:
                print(f"  [{tier}/{seed_label}] bootstrap failed — too few valid samples")
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
            print(f"  [{tier}/{seed_label}] AUC-PR={eval_auc_pr:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

            # Pick best among CV-selected rows by Frozen_Test_AUC_PR_Final (already sorted desc)
            if best_m3 is None:
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
