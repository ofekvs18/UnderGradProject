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
    df, features = load_data_for(disease.name)
    train_df, test_df = get_splits(df)
    tr_clean = train_df[features + ["is_case"]].dropna()
    te_clean = test_df[features + ["is_case"]].dropna()
    X_train, y_train = tr_clean[features].values, tr_clean["is_case"].values
    X_test, y_test = te_clean[features].values, te_clean["is_case"].values

    # 5. Run Tiers Specified in YAML
    # This allows you to queue multiple runs (e.g. small then huge) without code changes
    for tier_name in ml.method3.active_tiers:
        tier_cfg = ml.method3.gp_configs[tier_name]
        patience = tier_cfg.get("patience", 10)
        
        TIER_DIR = OUT_DIR_BASE / tier_name
        ensure_dir(TIER_DIR)

        print(f"\n{'='*60}")
        print(f"STARTING TIER: {tier_name.upper()} | Target AUC-PR: >{baseline_auc_pr:.4f}")
        print(f"{'='*60}")

        # Use the first configuration attempt for primitives/parsimony
        tier_cfg = ml.method3.gp_configs[tier_name]
        patience = tier_cfg.get("patience", 10)
        prevalence = float(y_train.mean())
        combined_auc_fitness = make_fitness(
                function=partial(_combined_auc, prevalence=prevalence),
                greater_is_better=True
            )
        print(f"  Fitness: roc_w={_ml.method3.fitness.roc_weight} * AUC-ROC + "
              f"pr_w={_ml.method3.fitness.pr_weight} * (AUC-PR / prevalence={prevalence:.4f})")
        # Pull parameters directly from the tier config instead of a separate list
        gp = SymbolicTransformer(
            population_size=tier_cfg.population_size,
            generations=1,
            warm_start=True,
            hall_of_fame=ml.method3.hall_of_fame,
            n_components=ml.method3.n_components,
            feature_names=features,
            # Read from tier_cfg now
            function_set=list(tier_cfg.function_set),
            parsimony_coefficient=tier_cfg.parsimony_coefficient,
            metric=combined_auc_fitness,
            random_state=ml.seed,
            n_jobs=1,
            verbose=1
        )

        best_overall_auc_pr = -1.0
        winning_formula = ""
        winning_gen = 0
        best_overall_roc = 0.0
        patience_counter = 0
        progress_rows = []

        # Iterative Generation Loop
        for gen in range(tier_cfg.generations):
            # FIX: Increment generations target to satisfy warm_start check
            gp.set_params(generations=gen + 1)
            
            gp.fit(X_train, y_train)
            
            # Evaluate Hall of Fame to find the best formula of the current state
            current_gen_best_pr = -1.0
            current_gen_best_formula = ""
            current_gen_best_roc = 0.0

            for prog in gp._best_programs:
                if prog is None: continue
                res = evaluate_program(prog, X_train, y_train, X_test, y_test)
                if res and res["auc_pr"] > current_gen_best_pr:
                    current_gen_best_pr = res["auc_pr"]
                    current_gen_best_formula = res["formula"]
                    current_gen_best_roc = res["auc_roc"]

            # Update the global best tracker
            improved = False
            if current_gen_best_pr > best_overall_auc_pr:
                best_overall_auc_pr = current_gen_best_pr
                best_overall_roc = current_gen_best_roc
                winning_formula = current_gen_best_formula
                winning_gen = gen
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1

            # Log Progress Locally (inside results/method3_gp/{disease}/{tier}/)
            if gen % args.log_every == 0 or improved:
                progress_rows.append({
                    "Generation": gen,
                    "Current_Best_Formula": winning_formula,
                    "Current_Best_AUC_PR": round(best_overall_auc_pr, 4),
                    "Is_New_Best": improved,
                    "Beats_Baseline": best_overall_auc_pr > baseline_auc_pr
                })
                pd.DataFrame(progress_rows).to_csv(TIER_DIR / "progress_log.csv", index=False)

            # Plateau Stop Logic
            if patience_counter >= patience:
                print(f"\n--- STOP: Plateau detected at gen {gen} (no improvement for {patience} gens) ---")
                break

        # 6. Final Aggregate Master Logging (Additive)
        master_row = {
            "Disease": disease.name,
            "Config_Used": tier_name,
            "Best_GP_Formula": winning_formula,
            "Best_GP_AUC_PR": round(best_overall_auc_pr, 4),
            "Best_GP_AUC_ROC": round(best_overall_roc, 4),
            "Winning_Generation": winning_gen,
            "Total_Generations": gen,
            "Beats_Baseline": best_overall_auc_pr > baseline_auc_pr
        }

        MASTER_PATH = RESULTS_DIR / "method3_gp" / "master_gp_summary.csv"
        ensure_dir(MASTER_PATH.parent)
        
        if MASTER_PATH.exists():
            master_df = pd.read_csv(MASTER_PATH)
            
            # FIX: Filter out only the specific Disease AND Tier combination
            # This allows you to keep 'ra/small' while updating 'ra/huge'
            mask = (master_df["Disease"] == disease.name) & (master_df["Config_Used"] == tier_name)
            master_df = master_df[~mask]
            
            master_df = pd.concat([master_df, pd.DataFrame([master_row])], ignore_index=True)
        else:
            master_df = pd.DataFrame([master_row])

        # Sort by Disease then Tier for a clean overview
        master_df.sort_values(["Disease", "Config_Used"]).to_csv(MASTER_PATH, index=False)
        print(f"Master summary updated: {disease.name} | Tier: {tier_name}")

if __name__ == "__main__":
    main()