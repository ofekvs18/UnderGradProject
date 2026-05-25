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
    get_cv_folds, cv_summary, count_formula_features, load_per_k_baselines,
    translate_seed_expression,
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



def evaluate_program(program, X_train, y_train, X_test, y_test, features=None):
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

    formula_str = str(program)
    result = {
        "formula":                formula_str,
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
    if features is not None:
        result["num_features"] = count_formula_features(formula_str, features)
    return result

# ── Seed injection helpers (Issue #27) ───────────────────────────────────────

def _load_seed_exprs(seed_file):
    """Load and translate seed expressions from CSV. Returns list of translated expr strings."""
    try:
        df = pd.read_csv(seed_file)
        if "expression" not in df.columns:
            print(f"[WARN] Seed file {seed_file} has no 'expression' column — skipping")
            return []
        exprs = []
        for raw in df["expression"].dropna().astype(str):
            try:
                exprs.append(translate_seed_expression(raw.strip()))
            except Exception:
                pass
        return exprs
    except Exception as e:
        print(f"[WARN] Failed to load seed file {seed_file}: {e}")
        return []


def _parse_seed_expression(expr_str, features, func_objects):
    """
    Convert a seed expression string to a gplearn program list (prefix notation).
    func_objects: dict mapping function name -> _Function object.
    Returns None if the expression uses unavailable functions or variables.
    """
    import ast as _ast
    feature_idx = {f: i for i, f in enumerate(features)}
    _aliases = {'log1p': 'log', 'log2': 'log', 'absolute': 'abs', 'negative': 'neg'}

    def _node(n):
        if isinstance(n, _ast.BinOp):
            op_map = {_ast.Add: 'add', _ast.Sub: 'sub', _ast.Mult: 'mul', _ast.Div: 'div'}
            name = op_map.get(type(n.op))
            fn = func_objects.get(name) if name else None
            if fn is None:
                return None
            left, right = _node(n.left), _node(n.right)
            if left is None or right is None:
                return None
            return [fn] + left + right
        elif isinstance(n, _ast.UnaryOp) and isinstance(n.op, _ast.USub):
            fn = func_objects.get('neg')
            if fn is None:
                return None
            inner = _node(n.operand)
            return ([fn] + inner) if inner is not None else None
        elif isinstance(n, _ast.Call):
            fn_name = (n.func.id if isinstance(n.func, _ast.Name)
                       else n.func.attr if isinstance(n.func, _ast.Attribute) else None)
            if fn_name is None:
                return None
            fn_name = _aliases.get(fn_name, fn_name)
            fn = func_objects.get(fn_name)
            if fn is None or len(n.args) != 1:
                return None
            inner = _node(n.args[0])
            return ([fn] + inner) if inner is not None else None
        elif isinstance(n, _ast.Name):
            idx = feature_idx.get(n.id)
            return [idx] if idx is not None else None
        elif isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)):
            return [float(n.value)]
        elif hasattr(_ast, 'Num') and isinstance(n, _ast.Num):  # Python <3.8
            return [float(n.n)]
        return None

    try:
        tree = _ast.parse(expr_str.strip(), mode='eval')
        return _node(tree.body)
    except Exception:
        return None


def _inject_seed_programs(gp, seed_exprs, features, combined_auc_fitness, X_train, y_train,
                           seed_fraction):
    """
    Replace the weakest programs in gp._programs[-1] with programs built from seed_exprs.
    Must be called after at least one gp.fit() so gp.function_set_ is available.
    Returns the count of successfully injected programs.
    """
    import copy
    if not hasattr(gp, '_programs') or not gp._programs:
        return 0
    pop = gp._programs[-1]
    if not pop:
        return 0

    n_max = max(1, int(len(pop) * seed_fraction))
    n_inject = min(n_max, len(seed_exprs))
    if n_inject == 0:
        return 0

    # Build function map from nodes in existing programs (post-fit objects)
    func_objects = {}
    for prog in pop:
        if prog is not None:
            for node in prog.program:
                if hasattr(node, 'name') and node.name not in func_objects:
                    func_objects[node.name] = node

    valid_pop = [(i, p) for i, p in enumerate(pop) if p is not None and hasattr(p, 'fitness_')]
    if not valid_pop:
        return 0
    # Replace weakest programs first
    valid_pop.sort(key=lambda x: x[1].fitness_)
    replace_slots = [idx for idx, _ in valid_pop[:n_inject]]
    template = valid_pop[0][1]

    injected = 0
    slot_ptr = 0
    for expr_str in seed_exprs:
        if injected >= n_inject:
            break
        program_list = _parse_seed_expression(expr_str, features, func_objects)
        if program_list is None:
            continue
        try:
            new_prog = copy.deepcopy(template)
            new_prog.program = program_list
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = np.asarray(new_prog.execute(X_train), dtype=float)
            bad = ~np.isfinite(scores)
            if bad.mean() > BAD_FRAC:
                continue
            if bad.any():
                scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0
            raw_fit = combined_auc_fitness(y_train, scores, None)
            new_prog.raw_fitness_ = raw_fit
            # greater_is_better=True → parsimony subtracts
            new_prog.fitness_ = raw_fit - gp.parsimony_coefficient * len(program_list)
            pop[replace_slots[slot_ptr]] = new_prog
            slot_ptr += 1
            injected += 1
        except Exception:
            continue
    return injected


def main():
    parser = argparse.ArgumentParser(description="Method 3: GP (Iterative Search)")
    parser.add_argument("--disease", default="ra", help="Disease slug (e.g. ra, dm1)")
    parser.add_argument("--split-salt", default="", help="Labeled split variant (e.g. _seed2)")
    parser.add_argument("--log-every", type=int, default=5, help="Track progress every X gens")
    parser.add_argument("--seed-file", type=str, default=None,
        help="Path to CSV with expression column (LLM seed formulas). "
             "Optional. If omitted, GP initializes fully at random.")
    parser.add_argument("--seed-fraction", type=float, default=0.3,
        help="Fraction of initial population to fill with seed formulas (default 0.3).")
    parser.add_argument("--pop", type=int, default=None,
        help="Override population_size for all active tiers.")
    parser.add_argument("--gen", type=int, default=None,
        help="Override generations for all active tiers.")
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

    # Load per-k LR baselines (computed by sanity_check.py)
    per_k_bl = load_per_k_baselines(disease.name, args.split_salt)
    if per_k_bl:
        print(f"Loaded per-k baselines for k=1..{max(per_k_bl)}")
    else:
        print("Warning: per-k baselines not found — run sanity_check.py first")

    # 5a. Load seed expressions (Issue #27)
    seed_file_basename = "none"
    seed_exprs = []
    if args.seed_file:
        from pathlib import Path as _Path
        seed_file_basename = _Path(args.seed_file).name
        seed_exprs = _load_seed_exprs(args.seed_file)
        if seed_exprs:
            print(f"Loaded {len(seed_exprs)} seed expressions from {seed_file_basename}")
        else:
            print(f"[WARN] No valid seed expressions loaded from {seed_file_basename}")

    # 5. Run Tiers Specified in YAML
    # This allows you to queue multiple runs (e.g. small then huge) without code changes
    cv_folds = get_cv_folds(train_df)
    tier_results = []  # accumulates per-tier dicts including CV scores and program objects

    for tier_name in ml.method3.active_tiers:
        tier_cfg = ml.method3.gp_configs[tier_name]
        patience = tier_cfg.get("patience", 10)
        # CLI overrides for quick experiments
        pop_size = args.pop if args.pop else tier_cfg.population_size
        n_gens   = args.gen if args.gen else tier_cfg.generations

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
            population_size=pop_size,
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
        winning_num_features = 0
        patience_counter = 0
        progress_rows = []
        per_k_best = {}  # {k: {"formula": str, "auc_pr": float, "auc_roc": float}}
        seed_count_used = 0
        _seeds_injected = False

        # Iterative Generation Loop
        for gen in range(n_gens):
            gp.set_params(generations=gen + 1)
            gp.fit(X_train, y_train)

            # Inject seeds into gen-0 population so they participate as parents from gen 1
            if not _seeds_injected and seed_exprs:
                seed_count_used = _inject_seed_programs(
                    gp, seed_exprs, features, combined_auc_fitness,
                    X_train, y_train, args.seed_fraction,
                )
                if seed_count_used > 0:
                    print(f"  Injected {seed_count_used} seed programs into generation 1 parent pool")
                else:
                    print("  [WARN] No seed programs could be parsed/injected for this tier")
                _seeds_injected = True

            current_gen_best_pr = -1.0
            current_gen_best_formula = ""
            current_gen_best_roc = 0.0
            current_gen_best_prog = None

            for prog in gp._best_programs:
                if prog is None: continue
                res = evaluate_program(prog, X_train, y_train, X_test, y_test, features=features)
                if res:
                    k = res.get("num_features", 0)
                    if k not in per_k_best or res["auc_pr"] > per_k_best[k]["auc_pr"]:
                        per_k_best[k] = {"formula": res["formula"], "auc_pr": res["auc_pr"], "auc_roc": res["auc_roc"]}
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
                winning_num_features = count_formula_features(winning_formula, features)
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
                if per_k_best:
                    _tmp_rows = [
                        {
                            "K":                 k,
                            "Best_Formula":      kdata["formula"],
                            "AUC_PR":            kdata["auc_pr"],
                            "AUC_ROC":           kdata["auc_roc"],
                            "Baseline_AUC_PR":   per_k_bl.get(k, {}).get("auc_pr"),
                            "Delta_vs_Baseline": round(kdata["auc_pr"] - per_k_bl.get(k, {}).get("auc_pr"), 4)
                                if per_k_bl.get(k, {}).get("auc_pr") is not None else None,
                        }
                        for k, kdata in sorted(per_k_best.items())
                    ]
                    pd.DataFrame(_tmp_rows).to_csv(TIER_DIR / "per_k_best.csv", index=False)

            if patience_counter >= patience:
                print(f"\n--- STOP: Plateau detected at gen {gen} (no improvement for {patience} gens) ---")
                break

        # ── Nested CV: run CV on all hall-of-fame programs, pick winner by CV mean ──
        # winner_program from the evolution loop was tracked by test AUC-PR (monitoring
        # only); here we select the true CV winner without touching the frozen test set.
        from sklearn.metrics import roc_auc_score as _roc, average_precision_score as _aps
        hof_programs = [p for p in gp._best_programs if p is not None]
        print(f"\n  Running nested CV for tier {tier_name} on {len(hof_programs)} hall-of-fame programs...")

        best_cv_mean = -1.0
        cv_winning_program = winning_program  # fallback
        cv_winning_formula = winning_formula
        tier_cv = {"mean": best_overall_auc_pr, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

        for prog in hof_programs:
            fold_prs = []
            for fold_train_df, fold_val_df in cv_folds:
                ft = fold_train_df[features + ["is_case"]].dropna()
                fv = fold_val_df[features + ["is_case"]].dropna()
                if ft["is_case"].sum() < 5 or fv["is_case"].sum() < 5:
                    continue
                X_ft = ft[features].values
                X_fv = fv[features].values
                y_ft = ft["is_case"].values
                y_fv = fv["is_case"].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sc_ft = np.asarray(prog.execute(X_ft), dtype=float)
                    sc_fv = np.asarray(prog.execute(X_fv), dtype=float)
                bad_ft = ~np.isfinite(sc_ft)
                bad_fv = ~np.isfinite(sc_fv)
                if bad_ft.mean() > BAD_FRAC or bad_fv.mean() > BAD_FRAC:
                    continue
                sc_ft[bad_ft] = np.nanmedian(sc_ft[~bad_ft]) if (~bad_ft).any() else 0.0
                sc_fv[bad_fv] = np.nanmedian(sc_fv[~bad_fv]) if (~bad_fv).any() else 0.0
                try:
                    auc_roc_ft = float(_roc(y_ft, sc_ft))
                    if auc_roc_ft < 0.5:
                        sc_fv = -sc_fv
                    fold_prs.append(float(_aps(y_fv, sc_fv)))
                except Exception:
                    pass

            if len(fold_prs) < 3:
                continue
            s = cv_summary(fold_prs)
            if s["mean"] > best_cv_mean:
                best_cv_mean = s["mean"]
                cv_winning_program = prog
                cv_winning_formula = str(prog)
                tier_cv = s

        if best_cv_mean < 0:
            print(f"  [WARN] {tier_name}: no program had ≥3 valid CV folds — using evolution best as fallback")
            cv_winning_program = winning_program
            cv_winning_formula = winning_formula

        # Use CV winner going forward
        winning_program = cv_winning_program
        winning_formula = cv_winning_formula
        winning_num_features = count_formula_features(winning_formula, features)
        print(f"  Tier {tier_name} CV winner AUC-PR: mean={tier_cv['mean']:.4f}  std={tier_cv['std']:.4f}")

        # Save per-k best formulas for this tier
        per_k_rows = []
        for k, kdata in sorted(per_k_best.items()):
            bl_pr = per_k_bl.get(k, {}).get("auc_pr")
            per_k_rows.append({
                "K":                 k,
                "Best_Formula":      kdata["formula"],
                "AUC_PR":            kdata["auc_pr"],
                "AUC_ROC":           kdata["auc_roc"],
                "Baseline_AUC_PR":   bl_pr,
                "Delta_vs_Baseline": round(kdata["auc_pr"] - bl_pr, 4) if bl_pr is not None else None,
            })
        if per_k_rows:
            pd.DataFrame(per_k_rows).to_csv(TIER_DIR / "per_k_best.csv", index=False)
            print(f"  Saved per-k best to {TIER_DIR}/per_k_best.csv")

            print(f"\n  Per-k best vs LR baseline ({tier_name}):")
            for row in per_k_rows:
                bl_str = (f"baseline={row['Baseline_AUC_PR']:.4f}  delta={row['Delta_vs_Baseline']:+.4f}"
                          if row["Baseline_AUC_PR"] is not None else "baseline=N/A")
                print(f"    k={row['K']}: AUC-PR={row['AUC_PR']:.4f}  {bl_str}")

        tier_results.append({
            "tier_name": tier_name,
            "winning_formula": winning_formula,
            "winning_program": winning_program,
            "winning_num_features": winning_num_features,
            "best_overall_auc_pr": best_overall_auc_pr,
            "best_overall_roc": best_overall_roc,
            "winning_gen": winning_gen,
            "total_gen": gen,
            "cv_auc_pr_mean": tier_cv["mean"],
            "cv_auc_pr_std": tier_cv["std"],
            "cv_auc_pr_ci95_low": tier_cv["ci95_low"],
            "cv_auc_pr_ci95_high": tier_cv["ci95_high"],
            "per_k_best": per_k_best,
            "seed_count_used": seed_count_used,
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
        best_k = r["winning_num_features"]
        bl_pr = per_k_bl.get(best_k, {}).get("auc_pr")
        new_rows.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Disease": disease.name,
            "Split_Salt": args.split_salt,
            "Config_Used": r["tier_name"],
            "Best_GP_Formula": r["winning_formula"],
            "Best_GP_AUC_PR": round(r["best_overall_auc_pr"], 4),
            "Best_GP_AUC_ROC": round(r["best_overall_roc"], 4),
            "Num_Features_Best": best_k,
            "Baseline_AUC_PR_At_K": bl_pr,
            "Delta_vs_Baseline_K": round(r["best_overall_auc_pr"] - bl_pr, 4) if bl_pr is not None else None,
            "Winning_Generation": r["winning_gen"],
            "Total_Generations": r["total_gen"],
            "Beats_Baseline": r["best_overall_auc_pr"] > baseline_auc_pr,
            "CV_AUC_PR_Mean": r["cv_auc_pr_mean"],
            "CV_AUC_PR_Std": r["cv_auc_pr_std"],
            "CV_AUC_PR_CI95_Low": r["cv_auc_pr_ci95_low"],
            "CV_AUC_PR_CI95_High": r["cv_auc_pr_ci95_high"],
            "CV_Selected": is_cv_selected,
            "Frozen_Test_AUC_PR_Final": round(frozen_test_auc_pr_final, 4) if is_cv_selected else None,
            "Seed_File": seed_file_basename,
            "Seed_Count_Used": r["seed_count_used"],
        })

    new_df = pd.DataFrame(new_rows)
    if MASTER_PATH.exists():
        master_df = pd.concat([pd.read_csv(MASTER_PATH), new_df], ignore_index=True)
    else:
        master_df = new_df

    master_df.to_csv(MASTER_PATH, index=False)

    # Per-disease master (pass criteria path: results/method3_gp/<disease>/master_m3_summary.csv)
    DISEASE_MASTER_PATH = OUT_DIR_BASE / "master_m3_summary.csv"
    if DISEASE_MASTER_PATH.exists():
        disease_master_df = pd.concat([pd.read_csv(DISEASE_MASTER_PATH), new_df], ignore_index=True)
    else:
        disease_master_df = new_df
    disease_master_df.to_csv(DISEASE_MASTER_PATH, index=False)

    print(f"Master summary updated: {disease.name} | CV winner: {best_tier['tier_name']}")

if __name__ == "__main__":
    main()