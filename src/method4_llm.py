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
    RESULTS_DIR,
    ensure_dir,
    eval_formula_scores,
    evaluate_formula_full,
    get_splits,
    load_data_for,
    load_ml_config,
    load_medgemma,
    load_prompts,
    medgemma_generate,
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
    global OUT_DIR, RAW_FILE, PARSED_FILE, RESULTS_FILE, SUMMARY_FILE
    
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
FEATURE_VARS     = {"hct", "hgb", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_DATA = load_prompts()
_FORMAT_SPEC = _PROMPT_DATA["method4_llm"]["components"]["format_spec"]["template"]
_COT_INSTRUCTION = _PROMPT_DATA["method4_llm"]["components"]["cot_instruction"]["template"]

_FEATURE_BLOCK = "\n".join(
    f"  - {var}: {info['name']} ({info['unit']}) — {info['ra_relevance']}"
    for var, info in _PROMPT_DATA["method4_llm"]["components"]["cbc_features"]["features"].items()
)

def build_prompt(strategy: str, n_formulas: int, chain_of_thought: bool) -> str:
    """Builds a prompt string with injected baseline hints for the seeded strategy."""
    template = _PROMPT_DATA["method4_llm"]["prompts"][strategy]["template"]
    cot_section = f"\n{_COT_INSTRUCTION}\n" if chain_of_thought else ""
    format_spec = _FORMAT_SPEC.format(n_formulas=n_formulas)

    # Contextual seeding from Master Sanity Check
    baseline_context = ""
    if strategy == "seeded" and BASelines["single_feat_name"] != "None":
        baseline_context = (
            f"\nClinical Context Supplement:\n"
            f"- Our statistical analysis shows {BASelines['single_feat_name'].upper()} is the strongest single predictor "
            f"for this cohort (AUC-PR: {BASelines['single_feat_pr']:.4f}).\n"
            f"- Aim to generate formulas that refine this relationship or combine it with other relevant CBC indices.\n"
        )

    prompt = template.format(
        feature_block=_FEATURE_BLOCK,
        n_formulas=n_formulas,
        cot_section=cot_section,
        format_spec=format_spec
    )
    
    if baseline_context:
        prompt = prompt.replace("Here are the features available:", baseline_context + "\nHere are the features available:")
        
    return prompt.strip()

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

def _update_master_summary(results_df: pd.DataFrame, disease_slug: str, split_salt: str = ""):
    """
    Updates the global Master Summary by appending the best-performing formula
    for this run (timestamp-stamped; previous entries are never removed).
    """
    if results_df.empty:
        return

    from datetime import datetime
    best = results_df.iloc[0]

    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Disease": disease_slug,
        "Split_Salt": split_salt,
        "Best_LLM_AUC_PR": best["auc_pr"],
        "Best_LLM_AUC_ROC": best["auc_roc"],
        "Best_LLM_Formula": best["formula"],
        "Winning_Strategy": best["strategy"],
        "Winning_Temp": best["temperature"],
        "Total_Yield": len(results_df)
    }])

    if MASTER_SUMMARY_CSV.exists():
        master_df = pd.concat([pd.read_csv(MASTER_SUMMARY_CSV), new_entry], ignore_index=True)
    else:
        ensure_dir(MASTER_SUMMARY_CSV.parent)
        master_df = new_entry

    master_df.to_csv(MASTER_SUMMARY_CSV, index=False)
    print(f"Master Summary updated at: {MASTER_SUMMARY_CSV}")

def _write_performance_summary(df: pd.DataFrame, disease: str):
    """
    Writes a text summary comparing top LLM results to Sanity Check baselines.
    """
    best = df.iloc[0]
    
    beat_all_pr = "YES" if best['auc_pr'] > BASelines['all_feat_pr'] else "NO"
    beat_single_pr = "YES" if best['auc_pr'] > BASelines['single_feat_pr'] else "NO"

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
    train_df, test_df = get_splits(df)

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
                "timestamp": item["timestamp"]
            })
            final_results.append(metrics)

    if not final_results:
        print("[WARN] No valid formulas survived evaluation.")
        return

    results_df = pd.DataFrame(final_results).sort_values("auc_pr", ascending=False)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Detailed formula log saved to: {RESULTS_FILE}")

    _update_master_summary(results_df, disease_slug, split_salt)
    _write_performance_summary(results_df, disease_slug)

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