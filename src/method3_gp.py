"""
Issue #17 / Method 3: Genetic Programming with gplearn.

Uses SymbolicTransformer to evolve formula-based biomarkers from CBC data.
Custom fitness = 0.5*AUC-ROC + 2.0*(AUC-PR/prevalence). The AUC-PR/prevalence
term is a lift score (random baseline = 1.0), making cross-disease fitness comparable.
Hall-of-fame programs are evaluated via program.execute(X) directly —
no formula string parsing.

Research question: can evolutionary search discover CBC combinations that
outperform random search (Method 2, AUC-PR=0.0174) and logistic regression
(AUC-PR=0.017)?
"""

# Standard library
import argparse
from datetime import datetime
import sys
import warnings

# Third-party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from functools import partial

# Local
sys.path.insert(0, "src")
from utils import (
    load_data_for, load_disease_config, load_ml_config, get_splits, compute_binary_metrics,
    find_youden_threshold, precision_at_recall_levels, ensure_dir, RESULTS_DIR,
    get_cv_folds, cv_summary,
)

_ml = load_ml_config()

M2_BEST_AUC_PR = _ml.baselines.m2_best_auc_pr
LR_AUC_PR      = _ml.baselines.lr_auc_pr
LR_AUC_ROC     = _ml.baselines.lr_auc_roc
SEED           = _ml.seed

SMALL_CONFIG  = dict(**_ml.method3.gp_configs.small)
MEDIUM_CONFIG = dict(**_ml.method3.gp_configs.medium)
LARGE_CONFIG  = dict(**_ml.method3.gp_configs.large)

BAD_FRAC                = _ml.method3.bad_frac
HALL_OF_FAME            = _ml.method3.hall_of_fame
N_COMPONENTS            = _ml.method3.n_components


def _combined_auc(y, y_pred, sample_weight, prevalence=None):
    """
    Fitness = roc_w * AUC-ROC + pr_w * (AUC-PR / prevalence).
    AUC-PR / prevalence is a lift score (1.0 = random baseline), making
    cross-disease fitness values directly comparable.
    Use partial() to bind prevalence before passing to make_fitness.
    """
    roc_w = _ml.method3.fitness.roc_weight
    pr_w  = _ml.method3.fitness.pr_weight

    y_pred = np.asarray(y_pred, dtype=float)
    bad = ~np.isfinite(y_pred)
    if bad.mean() > BAD_FRAC:
        return 0.0
    if bad.any():
        y_pred = y_pred.copy()
        y_pred[bad] = np.nanmedian(y_pred[~bad]) if (~bad).any() else 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc_roc = roc_auc_score(y, y_pred)
            auc_pr  = average_precision_score(y, y_pred)
        if auc_roc < 0.5:
            auc_roc = 1.0 - auc_roc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc_pr = average_precision_score(y, -y_pred)

        prev = prevalence if prevalence is not None else float(np.mean(y))
        return float(roc_w * auc_roc + pr_w * (auc_pr / prev))
    except Exception:
        return 0.0



def evaluate_program(program, X_train, y_train, X_test, y_test):
    """
    Evaluate a gplearn program using program.execute(X) directly.
    Returns a metrics dict or None if the program produces invalid output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score_tr = np.asarray(program.execute(X_train), dtype=float)
        score_te = np.asarray(program.execute(X_test),  dtype=float)

    for scores in (score_tr, score_te):
        bad = ~np.isfinite(scores)
        if bad.mean() > BAD_FRAC:
            return None
        if bad.any():
            scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0

    if y_train.sum() < 5 or y_test.sum() < 5:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc_roc = float(roc_auc_score(y_test, score_te))
            auc_pr  = float(average_precision_score(y_test, score_te))
    except Exception:
        return None

    if auc_roc < 0.5:
        score_tr = -score_tr
        score_te = -score_te
        auc_roc  = 1.0 - auc_roc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc_pr = float(average_precision_score(y_test, score_te))

    threshold, _, _ = find_youden_threshold(y_train, score_tr)
    preds = (score_te >= threshold).astype(int)
    m   = compute_binary_metrics(y_test, preds)
    par = precision_at_recall_levels(score_tr, y_train, score_te, y_test)

    return {
        "formula":                str(program),
        "auc_roc":                round(auc_roc, 4),
        "auc_pr":                 round(auc_pr, 4),
        "precision":              round(m["precision"], 4),
        "recall":                 round(m["recall"], 4),
        "f1":                     round(m["f1"], 4),
        "f2":                     round(m["f2"], 4),
        "precision_at_recall_25": par[0.25][0],
        "precision_at_recall_50": par[0.50][0],
        "precision_at_recall_75": par[0.75][0],
    }

def main():
    parser = argparse.ArgumentParser(description="Method 3: GP (Iterative Search)")
    parser.add_argument("--disease", default="ra", help="Disease slug (e.g. ra, dm1)")
    parser.add_argument("--split-salt", default="", help="Labeled split variant (e.g. _seed2)")
    parser.add_argument("--log-every", type=int, default=5, help="Track progress every X gens")
    args = parser.parse_args()

    # 1. Load Configurations
    disease = load_disease_config(args.disease)
    ml = load_ml_config()
    
    # 2. Setup Directories
    OUT_DIR_BASE = RESULTS_DIR / "method3_gp" / disease.name
    ensure_dir(OUT_DIR_BASE)
    
    # 3. Retrieve Dynamic Baseline from Sanity Master
    SANITY_MASTER = RESULTS_DIR / "sanity_check" / "master_sanity_summary.csv"
    baseline_auc_pr = 0.0 
    
    if SANITY_MASTER.exists():
        sanity_df = pd.read_csv(SANITY_MASTER)
        disease_row = sanity_df[sanity_df["Disease"] == disease.name]
        if not disease_row.empty:
            baseline_auc_pr = disease_row["All_Feat_AUC_PR"].values[0]
            print(f"Loaded baseline for {disease.name}: AUC-PR = {baseline_auc_pr:.4f}")
    else:
        print(f"Warning: master_sanity_summary.csv not found. Using 0.0 baseline.")

    # 4. Load Data
    print("Loading data...")
    df, features = load_data_for(disease.name, args.split_salt)
    train_df, test_df = get_splits(df)
    tr_clean = train_df[features + ["is_case"]].dropna()
    te_clean = test_df[features + ["is_case"]].dropna()
    X_train, y_train = tr_clean[features].values, tr_clean["is_case"].values
    X_test, y_test = te_clean[features].values, te_clean["is_case"].values

    # 5. Run Tiers Specified in YAML
    # This allows you to queue multiple runs (e.g. small then huge) without code changes
    cv_folds = get_cv_folds(train_df)
    tier_results = []  # accumulates per-tier dicts including CV scores and program objects

    for tier_name in ml.method3.active_tiers:
        tier_cfg = ml.method3.gp_configs[tier_name]
        patience = tier_cfg.get("patience", 10)

        TIER_DIR = OUT_DIR_BASE / tier_name
        ensure_dir(TIER_DIR)

        print(f"\n{'='*60}")
        print(f"STARTING TIER: {tier_name.upper()} | Target AUC-PR: >{baseline_auc_pr:.4f}")
        print(f"{'='*60}")

        prevalence = float(y_train.mean())
        def _fitness(x1, x2, w):
            return _combined_auc(x1, x2, w, prevalence=prevalence)

        combined_auc_fitness = make_fitness(
                function=_fitness,
                greater_is_better=True
            )
        print(f"  Fitness: roc_w={_ml.method3.fitness.roc_weight} * AUC-ROC + "
              f"pr_w={_ml.method3.fitness.pr_weight} * (AUC-PR / prevalence={prevalence:.4f})")
        gp = SymbolicTransformer(
            population_size=tier_cfg.population_size,
            generations=1,
            warm_start=True,
            hall_of_fame=ml.method3.hall_of_fame,
            n_components=ml.method3.n_components,
            feature_names=features,
            function_set=list(tier_cfg.function_set),
            parsimony_coefficient=tier_cfg.parsimony_coefficient,
            max_samples=tier_cfg.get("max_samples", 1.0),
            metric=combined_auc_fitness,
            random_state=ml.seed,
            n_jobs=1,
            verbose=1
        )

        best_overall_auc_pr = -1.0
        winning_formula = ""
        winning_program = None  # keep program object alive for CV
        winning_gen = 0
        best_overall_roc = 0.0
        patience_counter = 0
        progress_rows = []

        # Iterative Generation Loop
        for gen in range(tier_cfg.generations):
            gp.set_params(generations=gen + 1)
            gp.fit(X_train, y_train)

            current_gen_best_pr = -1.0
            current_gen_best_formula = ""
            current_gen_best_roc = 0.0
            current_gen_best_prog = None

            for prog in gp._best_programs:
                if prog is None: continue
                res = evaluate_program(prog, X_train, y_train, X_test, y_test)
                if res and res["auc_pr"] > current_gen_best_pr:
                    current_gen_best_pr = res["auc_pr"]
                    current_gen_best_formula = res["formula"]
                    current_gen_best_roc = res["auc_roc"]
                    current_gen_best_prog = prog

            improved = False
            if current_gen_best_pr > best_overall_auc_pr:
                best_overall_auc_pr = current_gen_best_pr
                best_overall_roc = current_gen_best_roc
                winning_formula = current_gen_best_formula
                winning_program = current_gen_best_prog  # keep reference alive
                winning_gen = gen
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1

            if gen % args.log_every == 0 or improved:
                progress_rows.append({
                    "Generation": gen,
                    "Current_Best_Formula": winning_formula,
                    "Current_Best_AUC_PR": round(best_overall_auc_pr, 4),
                    "Is_New_Best": improved,
                    "Beats_Baseline": best_overall_auc_pr > baseline_auc_pr
                })
                pd.DataFrame(progress_rows).to_csv(TIER_DIR / "progress_log.csv", index=False)

            if patience_counter >= patience:
                print(f"\n--- STOP: Plateau detected at gen {gen} (no improvement for {patience} gens) ---")
                break

        # ── CV on this tier's winning program (after evolution, not during) ────────
        print(f"\n  Running CV for tier {tier_name}...")
        fold_prs = []
        if winning_program is not None:
            for fold_train_df, fold_val_df in cv_folds:
                ft = fold_train_df[features + ["is_case"]].dropna()
                fv = fold_val_df[features + ["is_case"]].dropna()
                if ft["is_case"].sum() < 5 or fv["is_case"].sum() < 5:
                    print(f"    Fold skipped (too few positives)")
                    continue
                X_ft = ft[features].values
                X_fv = fv[features].values
                y_ft = ft["is_case"].values
                y_fv = fv["is_case"].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sc_ft = np.asarray(winning_program.execute(X_ft), dtype=float)
                    sc_fv = np.asarray(winning_program.execute(X_fv), dtype=float)
                bad_ft = ~np.isfinite(sc_ft)
                bad_fv = ~np.isfinite(sc_fv)
                if bad_ft.mean() > BAD_FRAC or bad_fv.mean() > BAD_FRAC:
                    continue
                sc_ft[bad_ft] = np.nanmedian(sc_ft[~bad_ft]) if (~bad_ft).any() else 0.0
                sc_fv[bad_fv] = np.nanmedian(sc_fv[~bad_fv]) if (~bad_fv).any() else 0.0
                try:
                    from sklearn.metrics import roc_auc_score as _roc, average_precision_score as _aps
                    auc_roc_ft = float(_roc(y_ft, sc_ft))
                    if auc_roc_ft < 0.5:
                        sc_fv = -sc_fv
                    fold_prs.append(float(_aps(y_fv, sc_fv)))
                except Exception:
                    pass

        if len(fold_prs) >= 3:
            tier_cv = cv_summary(fold_prs)
        else:
            print(f"  [WARN] {tier_name}: fewer than 3 valid CV folds — using train AUC-PR fallback")
            tier_cv = {"mean": best_overall_auc_pr, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

        print(f"  Tier {tier_name} CV AUC-PR: mean={tier_cv['mean']:.4f}  std={tier_cv['std']:.4f}")

        tier_results.append({
            "tier_name": tier_name,
            "winning_formula": winning_formula,
            "winning_program": winning_program,
            "best_overall_auc_pr": best_overall_auc_pr,
            "best_overall_roc": best_overall_roc,
            "winning_gen": winning_gen,
            "total_gen": gen,
            "cv_auc_pr_mean": tier_cv["mean"],
            "cv_auc_pr_std": tier_cv["std"],
            "cv_auc_pr_ci95_low": tier_cv["ci95_low"],
            "cv_auc_pr_ci95_high": tier_cv["ci95_high"],
        })

    # ── 6. Pick CV winner across tiers and evaluate on frozen test once ───────────
    if not tier_results:
        print("[WARN] No tier results to summarize.")
        return

    best_tier = max(tier_results, key=lambda r: r["cv_auc_pr_mean"])
    print(f"\nCV-selected tier: {best_tier['tier_name']}  (cv_auc_pr_mean={best_tier['cv_auc_pr_mean']:.4f})")

    # Evaluate CV-selected tier on frozen test exactly once
    if best_tier["winning_program"] is not None:
        frozen_res = evaluate_program(
            best_tier["winning_program"], X_train, y_train, X_test, y_test
        )
        frozen_test_auc_pr_final = frozen_res["auc_pr"] if frozen_res else float("nan")
    else:
        frozen_test_auc_pr_final = float("nan")
    print(f"Frozen test AUC-PR for CV winner: {frozen_test_auc_pr_final:.4f}")

    # 7. Final Aggregate Master Logging (one row per tier)
    MASTER_PATH = RESULTS_DIR / "method3_gp" / "master_gp_summary.csv"
    ensure_dir(MASTER_PATH.parent)

    new_rows = []
    for r in tier_results:
        is_cv_selected = r["tier_name"] == best_tier["tier_name"]
        new_rows.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Disease": disease.name,
            "Split_Salt": args.split_salt,
            "Config_Used": r["tier_name"],
            "Best_GP_Formula": r["winning_formula"],
            "Best_GP_AUC_PR": round(r["best_overall_auc_pr"], 4),
            "Best_GP_AUC_ROC": round(r["best_overall_roc"], 4),
            "Winning_Generation": r["winning_gen"],
            "Total_Generations": r["total_gen"],
            "Beats_Baseline": r["best_overall_auc_pr"] > baseline_auc_pr,
            "CV_AUC_PR_Mean": r["cv_auc_pr_mean"],
            "CV_AUC_PR_Std": r["cv_auc_pr_std"],
            "CV_AUC_PR_CI95_Low": r["cv_auc_pr_ci95_low"],
            "CV_AUC_PR_CI95_High": r["cv_auc_pr_ci95_high"],
            "CV_Selected": is_cv_selected,
            "Frozen_Test_AUC_PR_Final": round(frozen_test_auc_pr_final, 4) if is_cv_selected else None,
        })

    new_df = pd.DataFrame(new_rows)
    if MASTER_PATH.exists():
        master_df = pd.concat([pd.read_csv(MASTER_PATH), new_df], ignore_index=True)
    else:
        master_df = new_df

    master_df.to_csv(MASTER_PATH, index=False)
    print(f"Master summary updated: {disease.name} | CV winner: {best_tier['tier_name']}")

if __name__ == "__main__":
    main()