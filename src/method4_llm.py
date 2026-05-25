"""
method4_llm.py — LLM-based biomarker formula generation.

This script uses Med-Gemma 4B IT to generate biomarker formulas.
It integrates performance baselines from the Master Sanity Check CSV and
consolidates all generated formulas into a Master Vault.

Stages:
    generate: Run LLM inference and save raw text
    evaluate: Parse, deduplicate, and evaluate formulas against baselines
    all:      Run both stages sequentially
"""

import argparse
import json
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from utils import (
    CBC_FEATURE_LIST,
    RESULTS_DIR,
    ensure_dir,
    eval_formula_scores,
    evaluate_formula_full,
    get_splits,
    load_data_for,
    load_disease_config,
    load_ml_config,
    load_medgemma,
    load_prompts,
    medgemma_generate,
    get_cv_folds,
    cv_summary,
    count_formula_features,
    load_per_k_baselines,
)

_ml = load_ml_config()

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & PATHS
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR            = None
RAW_FILE           = None
PARSED_FILE        = None
RESULTS_FILE       = None
SUMMARY_FILE       = None

# Global master path (stays outside disease folders)
MASTER_SUMMARY_CSV = RESULTS_DIR / "method4_llm" / "method4_master_summary.csv"
SANITY_CHECK_CSV   = RESULTS_DIR / "sanity_check" / "master_sanity_summary.csv"

# Baseline placeholders updated dynamically
BASelines = {
    "all_feat_pr": 0.0, "all_feat_roc": 0.0,
    "single_feat_pr": 0.0, "single_feat_roc": 0.0,
    "single_feat_name": "None", "prevalence": 0.0
}

def _init_paths(disease: str) -> None:
    """Initialize paths and load baselines. CRITICAL: uses global keyword."""
    global OUT_DIR, RAW_FILE, PARSED_FILE, RESULTS_FILE, SUMMARY_FILE, _CURRENT_DISEASE

    _CURRENT_DISEASE = disease
    OUT_DIR      = RESULTS_DIR / "method4_llm" / disease
    RAW_FILE     = OUT_DIR / "raw_outputs.json"
    PARSED_FILE  = OUT_DIR / "parsed_formulas.json"
    RESULTS_FILE = OUT_DIR / "method4_results.csv"
    SUMMARY_FILE = OUT_DIR / "method4_summary.txt"

    ensure_dir(OUT_DIR)
    _load_baselines_from_sanity(disease)

def _load_baselines_from_sanity(disease_slug: str):
    """Fetch baseline data from the Master Sanity Check CSV."""
    global BASelines
    if not SANITY_CHECK_CSV.exists():
        print(f"[WARN] Master Sanity Check not found at {SANITY_CHECK_CSV}. Using defaults.")
        return

    try:
        df = pd.read_csv(SANITY_CHECK_CSV)
        # Match disease case-insensitively
        row = df[df['Disease'].str.lower() == disease_slug.lower()]
        
        if not row.empty:
            BASelines["all_feat_pr"]     = float(row.iloc[0]['All_Feat_AUC_PR'])
            BASelines["all_feat_roc"]    = float(row.iloc[0]['All_Feat_AUC_ROC'])
            BASelines["single_feat_pr"]  = float(row.iloc[0]['Best_Single_AUC_PR'])
            BASelines["single_feat_roc"] = float(row.iloc[0]['Best_Single_AUC_ROC'])
            BASelines["single_feat_name"] = str(row.iloc[0]['Best_Single_Feat'])
            BASelines["prevalence"]      = float(row.iloc[0]['Total_Prevalence'])
            print(f"Baselines loaded for {disease_slug}: All-Feat PR={BASelines['all_feat_pr']:.4f}")
        else:
            print(f"[WARN] Disease '{disease_slug}' not found in {SANITY_CHECK_CSV}.")
    except Exception as e:
        print(f"[ERROR] Failed to load baselines: {e}")

# Model Settings
MODEL_ID         = _ml.method4.model_id
DEFAULT_REPEATS  = _ml.method4.default_repeats
MAX_NEW_TOKENS   = _ml.method4.max_new_tokens
FUNCTIONAL_CORR  = _ml.method4.functional_corr
FEATURE_VARS     = {
    "hct", "hgb", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc",
    "neut_pct", "lym_pct", "mono_pct", "eos_pct", "baso_pct",
}

# Current disease — set by _init_paths; used by prompt building for seeded strategy
_CURRENT_DISEASE = "ra"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_DATA = load_prompts()
_FORMAT_SPEC = _PROMPT_DATA["method4_llm"]["components"]["format_spec"]["template"]
_COT_INSTRUCTION = _PROMPT_DATA["method4_llm"]["components"]["cot_instruction"]["template"]
_CBC_FEATURES_META = _PROMPT_DATA["method4_llm"]["components"]["cbc_features"]["features"]


def _build_feature_block(disease: str) -> str:
    """
    Build the feature description block for a prompt, using disease-specific
    relevance text from conf/disease/<disease>.yaml when available, falling
    back to the generic ra_relevance entry in prompts.json.
    """
    try:
        dcfg = load_disease_config(disease)
        relevance_map = dict(dcfg.get("feature_relevance", {}))
    except Exception:
        relevance_map = {}

    lines = []
    for var, info in _CBC_FEATURES_META.items():
        relevance = relevance_map.get(var, info.get("ra_relevance", ""))
        lines.append(f"  - {var}: {info['name']} ({info['unit']}) — {relevance}")
    return "\n".join(lines)

def _load_feature_importances(disease: str) -> dict | None:
    """
    Load GP feature importances from results/method3_gp/<disease>/feature_importance.json.
    Returns {feature: score} dict sorted descending, or None if the file doesn't exist.
    """
    fi_path = RESULTS_DIR / "method3_gp" / disease / "feature_importance.json"
    if not fi_path.exists():
        return None
    try:
        with open(fi_path, encoding="utf-8") as f:
            data = json.load(f)
        return dict(sorted(data.items(), key=lambda x: -x[1]))
    except Exception as e:
        print(f"[WARN] Failed to load feature importances from {fi_path}: {e}")
        return None


def build_prompt(strategy: str, n_formulas: int, chain_of_thought: bool) -> str:
    """Builds a prompt string. For seeded strategy, loads runtime GP feature importances."""
    cot_section = f"\n{_COT_INSTRUCTION}\n" if chain_of_thought else ""
    format_spec = _FORMAT_SPEC.format(n_formulas=n_formulas)
    feature_block = _build_feature_block(_CURRENT_DISEASE)

    if strategy == "seeded":
        fi = _load_feature_importances(_CURRENT_DISEASE)
        if fi is None:
            print(f"[WARN] No feature_importance.json for '{_CURRENT_DISEASE}' — falling back to blind prompt")
            strategy = "blind"
        else:
            ranked = list(fi.items())[:10]
            top_features = " > ".join(f"{feat} ({score:.2f})" for feat, score in ranked)

    template = _PROMPT_DATA["method4_llm"]["prompts"][strategy]["template"]

    fmt_kwargs: dict = dict(
        feature_block=feature_block,
        n_formulas=n_formulas,
        cot_section=cot_section,
        format_spec=format_spec,
    )
    if strategy == "seeded":
        fmt_kwargs["top_features"] = top_features

    # Contextual seeding from Master Sanity Check
    if strategy == "seeded" and BASelines["single_feat_name"] != "None":
        baseline_context = (
            f"\nClinical Context Supplement:\n"
            f"- Our statistical analysis shows {BASelines['single_feat_name'].upper()} is the strongest single predictor "
            f"for this cohort (AUC-PR: {BASelines['single_feat_pr']:.4f}).\n"
            f"- Aim to generate formulas that refine this relationship or combine it with other relevant CBC indices.\n"
        )
        fmt_kwargs["feature_block"] = baseline_context + "\n" + feature_block

    return template.format(**fmt_kwargs).strip()

def get_all_prompt_configs() -> list[dict]:
    configs = []
    for strategy in ["blind", "seeded"]:
        for temp in _ml.method4.temperatures:
            cot = temp >= _ml.method4.cot_min_temperature
            n = _ml.method4.n_formulas_per_call
            configs.append({
                "name": f"{strategy}_temp{temp}",
                "strategy": strategy,
                "temperature": temp,
                "n_formulas": n,
                "chain_of_thought": cot,
                "prompt": build_prompt(strategy, n, cot),
            })
    return configs
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GENERATION & PARSING
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(configs: list[dict], repeats: int) -> None:
    """Run inference and save raw results to RAW_FILE."""
    results = []
    if RAW_FILE.exists():
        with open(RAW_FILE) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing entries.")

    model, processor = load_medgemma()
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for cfg in configs:
        for r_idx in range(repeats):
            run_id = f"{cfg['name']}_r{r_idx}"
            if any(res.get("run_id") == run_id for res in results):
                continue

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Inference: {cfg['name']} (Rep {r_idx+1}/{repeats})")
            t0 = time.perf_counter()
            try:
                raw_text = medgemma_generate(
                    model, processor, cfg["prompt"], 
                    temperature=cfg["temperature"], 
                    max_new_tokens=MAX_NEW_TOKENS
                )
                status, error = "ok", None
            except Exception as e:
                raw_text, status, error = "", "error", str(e)
                print(f"  [ERR] {e}")

            results.append({
                "run_id": run_id,
                **cfg,
                "repeat_index": r_idx,
                "raw_text": raw_text,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
                "status": status,
                "error": error,
                "session_id": session_id
            })

            with open(RAW_FILE, "w") as f:
                json.dump(results, f, indent=2)

def normalize_formula(f: str) -> str:
    """Clean and normalize formula string for evaluation."""
    f = f.strip().lower()
    # Strip list markers like "1. ", "2) ", or "Formula: "
    f = re.sub(r"^[0-9\.\s\)\-\:]+", "", f) 
    f = re.sub(r"formula\s*\d*\s*[:=]", "", f)
    f = re.sub(r"score\s*=", "", f)
    
    # Ensure variables match the expected lowercase format
    for v in FEATURE_VARS:
        f = re.sub(rf"\b{v}\b", v, f, flags=re.IGNORECASE)
    
    # Standardize math functions
    f = f.replace("np.log", "log").replace("math.log", "log")
    f = f.replace("np.sqrt", "sqrt").replace("math.sqrt", "sqrt")
    
    return f.strip()

def parse_formulas_from_text(raw_text: str) -> list[str]:
    """Extract formula candidates from LLM output."""
    candidates = []
    
    # Match markdown code blocks (handles ```python, ```text, or just ```)
    # The [a-z]* captures any optional language tag
    code_blocks = re.findall(r"```[a-z]*\n(.*?)\n```", raw_text, re.DOTALL)
    
    for block in code_blocks:
        for line in block.splitlines():
            line = line.strip()
            if any(v in line.lower() for v in FEATURE_VARS):
                # If the line is an assignment (x = y + z), take the right side
                if "=" in line:
                    line = line.split("=")[-1]
                candidates.append(normalize_formula(line))

    # Also look for lines outside of code blocks that look like formulas
    for line in raw_text.splitlines():
        line = line.strip()
        # Look for lines containing features and math operators
        if any(v in line.lower() for v in FEATURE_VARS):
            if re.search(r"[\+\-\*/\(\)]", line):
                # Clean up if line starts with a label
                if ":" in line:
                    line = line.split(":", 1)[1]
                candidates.append(normalize_formula(line))
                
    # Final filter: must have at least one feature and be unique
    refined = [c for c in candidates if any(v in c for v in FEATURE_VARS) and len(c) > 3]
    return list(dict.fromkeys(refined))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION & SUMMARY UPDATES
# ══════════════════════════════════════════════════════════════════════════════

def functional_deduplicate(formula_dicts: list[dict], df: pd.DataFrame) -> list[dict]:
    """
    Remove formulas that produce identical score distributions (r > 0.999).
    This prevents wasting time evaluating mathematically redundant models.
    """
    if not formula_dicts: return []
    
    unique_list = []
    seen_scores = []
    
    print(f"Functional deduplication on {len(formula_dicts)} candidates...")
    
    for item in formula_dicts:
        formula = item["formula"]
        scores = eval_formula_scores(formula, df, list(FEATURE_VARS))
        
        if scores is None or np.std(scores) < 1e-9:
            continue
            
        is_duplicate = False
        for prev_scores in seen_scores:
            correlation = abs(np.corrcoef(scores, prev_scores)[0, 1])
            if correlation > FUNCTIONAL_CORR:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_list.append(item)
            seen_scores.append(scores)
            
    return unique_list

def _update_master_summary(results_df: pd.DataFrame, disease_slug: str, split_salt: str = "",
                           per_k_bl: dict = None, cv_winner=None):
    """
    Updates the global Master Summary by appending the best-performing formula
    for this run (timestamp-stamped; previous entries are never removed).
    cv_winner: optional Series from cv_df with cv_auc_pr_mean/std columns.
    """
    if results_df.empty:
        return

    from datetime import datetime
    best = results_df.iloc[0]
    best_k = int(best["num_features"]) if "num_features" in best.index else 0
    bl_pr = (per_k_bl or {}).get(best_k, {}).get("auc_pr")

    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Disease": disease_slug,
        "Split_Salt": split_salt,
        "Best_LLM_AUC_PR": best["auc_pr"],
        "Best_LLM_AUC_ROC": best["auc_roc"],
        "Best_LLM_Formula": best["formula"],
        "Winning_Strategy": best["strategy"],
        "Winning_Temp": best["temperature"],
        "Total_Yield": len(results_df),
        "Num_Features_Best": best_k,
        "Baseline_AUC_PR_At_K": bl_pr,
        "Delta_vs_Baseline_K": round(float(best["auc_pr"]) - bl_pr, 4) if bl_pr is not None else None,
        "CV_AUC_PR_Mean": round(float(cv_winner["cv_auc_pr_mean"]), 4) if cv_winner is not None else None,
        "CV_AUC_PR_Std": round(float(cv_winner["cv_auc_pr_std"]), 4) if cv_winner is not None else None,
    }])

    if MASTER_SUMMARY_CSV.exists():
        master_df = pd.concat([pd.read_csv(MASTER_SUMMARY_CSV), new_entry], ignore_index=True)
    else:
        ensure_dir(MASTER_SUMMARY_CSV.parent)
        master_df = new_entry

    master_df.to_csv(MASTER_SUMMARY_CSV, index=False)
    print(f"Master Summary updated at: {MASTER_SUMMARY_CSV}")

def _write_performance_summary(df: pd.DataFrame, disease: str, cv_winner=None, frozen_test_auc_pr_final=None):
    """
    Writes a text summary comparing top LLM results to Sanity Check baselines.
    """
    best = df.iloc[0]

    beat_all_pr = "YES" if best['auc_pr'] > BASelines['all_feat_pr'] else "NO"
    beat_single_pr = "YES" if best['auc_pr'] > BASelines['single_feat_pr'] else "NO"

    cv_section = ""
    if cv_winner is not None:
        cv_beat = "YES" if (frozen_test_auc_pr_final or 0) > BASelines['all_feat_pr'] else "NO"
        cv_section = textwrap.dedent(f"""
        CV WINNER (selected by 5-fold CV on train_df)
        ---------------------------------------
        CV_AUC_PR_Mean    : {cv_winner['cv_auc_pr_mean']:.4f}
        CV_AUC_PR_CI95    : [{cv_winner['cv_auc_pr_ci95_low']:.4f}, {cv_winner['cv_auc_pr_ci95_high']:.4f}]
        Frozen_Test_AUC_PR: {frozen_test_auc_pr_final:.4f} (Beat All-Feat LR: {cv_beat})
        CV Winner Formula : {cv_winner['formula']}
        """)

    summary = textwrap.dedent(f"""
        METHOD 4 PERFORMANCE SUMMARY: {disease.upper()}
        {"="*50}
        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        SANITY CHECK BASELINES (Reference: master_sanity_summary.csv)
        ---------------------------------------
        Cohort Prevalence : {BASelines['prevalence']:.4f}
        Best Single Feat  : {BASelines['single_feat_name']} (AUC-PR: {BASelines['single_feat_pr']:.4f})
        All-Feature LR    : AUC-PR: {BASelines['all_feat_pr']:.4f} | AUC-ROC: {BASelines['all_feat_roc']:.4f}

        LLM PERFORMANCE (Method 4)
        ---------------------------------------
        Total Evaluated   : {len(df)}
        Winner Formula    : {best['formula']}
        Winner AUC-PR     : {best['auc_pr']:.4f} (Beat All-Feat LR: {beat_all_pr})
        Winner AUC-ROC    : {best['auc_roc']:.4f}

        HEAD-TO-HEAD
        ---------------------------------------
        Beat Best Single Feat? : {beat_single_pr}
        Beat All-Feature LR?   : {beat_all_pr}
        Strategy of Winner     : {best['strategy']} (Temp: {best['temperature']})
        {cv_section}
        [Master Summary updated at {MASTER_SUMMARY_CSV}]
    """)

    SUMMARY_FILE.write_text(summary.strip())
    print(summary)

def run_evaluate(disease_slug: str, split_salt: str = ""):
    """
    Processes raw LLM outputs, evaluates all formulas locally, 
    and exports the 'champion' to the global Master Summary.
    """
    if not RAW_FILE.exists():
        print(f"[ERROR] No raw outputs found for {disease_slug}. Run generate first.")
        return

    print(f"--- Evaluating Method 4 for {disease_slug} ---")
    with open(RAW_FILE) as f:
        raw_entries = json.load(f)

    df, features = load_data_for(disease_slug, split_salt)
    assert set(CBC_FEATURE_LIST).issubset(set(df.columns)), \
        f"CSV missing features: {set(CBC_FEATURE_LIST) - set(df.columns)}"
    train_df, test_df = get_splits(df)
    per_k_bl = load_per_k_baselines(disease_slug, split_salt)

    all_extracted = []
    seen_strings = set()
    for entry in [r for r in raw_entries if r["status"] == "ok"]:
        raw_fs = parse_formulas_from_text(entry["raw_text"])
        for f_str in raw_fs:
            if f_str not in seen_strings:
                all_extracted.append({
                    "formula": f_str,
                    "strategy": entry["strategy"],
                    "temp": entry["temperature"],
                    "disease": disease_slug,
                    "timestamp": datetime.now().isoformat()
                })
                seen_strings.add(f_str)

    print(f"Extracted {len(all_extracted)} distinct formula strings.")

    unique_formulas = functional_deduplicate(all_extracted, train_df)
    print(f"Functional deduplication: {len(unique_formulas)} unique models remaining.")

    final_results = []
    for i, item in enumerate(unique_formulas):
        if i % 10 == 0: print(f"  Evaluating {i}/{len(unique_formulas)}...")
        metrics = evaluate_formula_full(item["formula"], train_df, test_df, list(FEATURE_VARS))
        if metrics:
            metrics.update({
                "strategy": item["strategy"],
                "temperature": item["temp"],
                "disease": item["disease"],
                "timestamp": item["timestamp"],
                "num_features": count_formula_features(item["formula"], list(FEATURE_VARS)),
            })
            final_results.append(metrics)

    if not final_results:
        print("[WARN] No valid formulas survived evaluation.")
        return

    results_df = pd.DataFrame(final_results).sort_values("auc_pr", ascending=False)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Detailed formula log saved to: {RESULTS_FILE}")

    # ── CV-based winner selection (evaluate stage only) ────────────────────────
    print("\n=== CV-based winner selection (top-20 by train AUC-PR) ===")
    from sklearn.metrics import average_precision_score as _aps, roc_auc_score as _roc

    tr_clean = train_df[list(FEATURE_VARS) + ["is_case"]].dropna()

    # Rank valid formulas by train AUC-PR (single pass, frozen test never used here)
    train_scores = []
    for item in unique_formulas:
        formula = item["formula"]
        sc = eval_formula_scores(formula, tr_clean, list(FEATURE_VARS))
        if sc is None:
            continue
        y_tr_cv = tr_clean["is_case"].values
        if y_tr_cv.sum() < 5:
            continue
        try:
            auc_roc_tr = float(_roc(y_tr_cv, sc))
            if auc_roc_tr < 0.5:
                sc = -sc
            train_scores.append({"formula": formula, "train_auc_pr": float(_aps(y_tr_cv, sc)),
                                  "strategy": item["strategy"], "temp": item["temp"]})
        except Exception:
            pass

    if not train_scores:
        print("[WARN] No train scores available for CV selection — skipping CV block")
        _update_master_summary(results_df, disease_slug, split_salt, per_k_bl)
        _write_performance_summary(results_df, disease_slug)
        return

    train_rank = sorted(train_scores, key=lambda x: x["train_auc_pr"], reverse=True)[:20]
    print(f"  Top-20 pool size: {len(train_rank)}")

    cv_folds = get_cv_folds(train_df)
    cv_rows = []
    for item in train_rank:
        formula = item["formula"]
        fold_prs = []
        for fold_train_df, fold_val_df in cv_folds:
            ft = fold_train_df[list(FEATURE_VARS) + ["is_case"]].dropna()
            fv = fold_val_df[list(FEATURE_VARS) + ["is_case"]].dropna()
            if ft["is_case"].sum() < 5 or fv["is_case"].sum() < 5:
                continue
            sc_ft = eval_formula_scores(formula, ft, list(FEATURE_VARS))
            sc_fv = eval_formula_scores(formula, fv, list(FEATURE_VARS))
            if sc_ft is None or sc_fv is None:
                continue
            try:
                auc_roc_ft = float(_roc(ft["is_case"].values, sc_ft))
                if auc_roc_ft < 0.5:
                    sc_fv = -sc_fv
                fold_prs.append(float(_aps(fv["is_case"].values, sc_fv)))
            except Exception:
                pass

        if len(fold_prs) < 3:
            print(f"  Formula '{formula[:40]}...': fewer than 3 valid CV folds — skipped")
            continue

        s = cv_summary(fold_prs)
        cv_rows.append({
            "formula": formula,
            "strategy": item["strategy"],
            "temperature": item["temp"],
            "train_auc_pr": item["train_auc_pr"],
            "cv_auc_pr_mean": s["mean"],
            "cv_auc_pr_std": s["std"],
            "cv_auc_pr_ci95_low": s["ci95_low"],
            "cv_auc_pr_ci95_high": s["ci95_high"],
            "frozen_test_auc_pr": None,
        })

    if not cv_rows:
        print("[WARN] No valid CV results — falling back to frozen test winner")
        _update_master_summary(results_df, disease_slug, split_salt, per_k_bl)
        _write_performance_summary(results_df, disease_slug)
        return

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_auc_pr_mean", ascending=False).reset_index(drop=True)
    cv_winner = cv_df.iloc[0]

    # Evaluate CV winner on frozen test exactly once
    cv_winner_metrics = evaluate_formula_full(cv_winner["formula"], train_df, test_df, list(FEATURE_VARS))
    frozen_test_auc_pr_final = cv_winner_metrics["auc_pr"] if cv_winner_metrics else float("nan")
    cv_df.loc[0, "frozen_test_auc_pr"] = frozen_test_auc_pr_final

    top_cv_path = OUT_DIR / "method4_top_cv.csv"
    cv_df.to_csv(top_cv_path, index=False)
    print(f"  CV winner: {cv_winner['formula'][:60]}...")
    print(f"  CV AUC-PR mean: {cv_winner['cv_auc_pr_mean']:.4f}  Frozen test: {frozen_test_auc_pr_final:.4f}")
    print(f"  Saved {top_cv_path}")

    # ── Per-k best results ────────────────────────────────────────────────────
    per_k_rows = []
    for k in range(1, len(list(FEATURE_VARS)) + 1):
        k_df = results_df[results_df["num_features"] == k]
        if k_df.empty:
            continue
        best_k_row = k_df.iloc[0]
        bl_pr = per_k_bl.get(k, {}).get("auc_pr")
        per_k_rows.append({
            "K":                 k,
            "Best_Formula":      best_k_row["formula"],
            "AUC_PR":            best_k_row["auc_pr"],
            "AUC_ROC":           best_k_row["auc_roc"],
            "N_Formulas_Tested": len(k_df),
            "Baseline_AUC_PR":   bl_pr,
            "Delta_vs_Baseline": round(float(best_k_row["auc_pr"]) - bl_pr, 4) if bl_pr is not None else None,
        })

    if per_k_rows:
        per_k_path = OUT_DIR / "per_k_best.csv"
        pd.DataFrame(per_k_rows).to_csv(per_k_path, index=False)
        print(f"Saved per-k best formulas to {per_k_path}")
        print("\n=== Per-k best vs LR baseline ===")
        for row in per_k_rows:
            bl_str = (f"baseline={row['Baseline_AUC_PR']:.4f}  delta={row['Delta_vs_Baseline']:+.4f}"
                      if row["Baseline_AUC_PR"] is not None else "baseline=N/A")
            print(f"  k={row['K']}: AUC-PR={row['AUC_PR']:.4f}  {bl_str}")

    _update_master_summary(results_df, disease_slug, split_salt, per_k_bl, cv_winner=cv_winner)
    _write_performance_summary(results_df, disease_slug, cv_winner, frozen_test_auc_pr_final)

# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Method 4: LLM-Based Biomarker Formula Generation (Med-Gemma 4B IT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Execution Examples:
              python src/method4_llm.py generate --disease ra  # Run LLM inference
              python src/method4_llm.py evaluate --disease ra  # Parse and evaluate
              python src/method4_llm.py all --disease ra       # Full pipeline
        """)
    )
    
    parser.add_argument(
        "stage", 
        choices=["generate", "evaluate", "all"], 
        help="Which processing stage to execute"
    )
    
    parser.add_argument("--split-salt", default="", help="Labeled split variant (e.g. _seed2)")
    parser.add_argument(
        "--disease",
        default="ra", 
        help="Disease slug for data loading and baseline lookup (default: ra)"
    )
    
    parser.add_argument(
        "--repeats", 
        type=int, 
        default=DEFAULT_REPEATS, 
        help=f"Number of inference repeats per configuration (default: {DEFAULT_REPEATS})"
    )

    args = parser.parse_args()

    # Initialize paths and load baselines from Master Sanity Check
    _init_paths(args.disease)

    if args.stage == "generate":
        print(f"--- Starting Generation Stage [{args.disease.upper()}] ---")
        configs = get_all_prompt_configs()
        run_inference(configs, args.repeats)

    elif args.stage == "evaluate":
        print(f"--- Starting Evaluation Stage [{args.disease.upper()}] ---")
        run_evaluate(args.disease, args.split_salt)

    elif args.stage == "all":
        print(f"--- Starting Full Pipeline [{args.disease.upper()}] ---")
        
        # Step 1: Inference
        configs = get_all_prompt_configs()
        run_inference(configs, args.repeats)
        
        print("\n" + "="*60)
        
        # Step 2: Evaluation and Vault Update
        run_evaluate(args.disease, args.split_salt)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print(f"Consolidated results available in: {RESULTS_FILE}")
        print(f"Master Summary updated at: {MASTER_SUMMARY_CSV}")

if __name__ == "__main__":
    main()