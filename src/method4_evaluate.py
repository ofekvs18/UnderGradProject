"""
method4_evaluate.py — Parse, validate, deduplicate, and evaluate LLM formulas (issue #22).

Three-stage pipeline:
  1. Parse  — extract formula strings from raw LLM text
  2. Validate — reject formulas that fail on sample data
  3. Deduplicate + Evaluate — remove duplicates, run full metrics

Input  : results/method4_llm/raw_outputs.json  (produced by method4_generate.py)
Outputs: results/method4_llm/
           parsed_formulas.json   — all extracted formulas with metadata
           parsing_report.txt     — funnel counts and diagnostics
           method4_results.csv    — full metrics for every valid unique formula
           method4_summary.txt    — top-10 summary table

Run:
    python src/method4_evaluate.py
"""

import json
import re
import sys
import textwrap
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from utils import (
    RESULTS_DIR,
    ensure_dir,
    eval_formula_scores,
    evaluate_formula_full,
    get_splits,
    load_data,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR        = RESULTS_DIR / "method4_llm"
RAW_FILE       = OUT_DIR / "raw_outputs.json"
PARSED_FILE    = OUT_DIR / "parsed_formulas.json"
REPORT_FILE    = OUT_DIR / "parsing_report.txt"
RESULTS_FILE   = OUT_DIR / "method4_results.csv"
SUMMARY_FILE   = OUT_DIR / "method4_summary.txt"

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_VARS      = {"hct", "hgb", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc"}
BASELINE_LR_AUC_ROC  = 0.658    # all-features logistic regression
BASELINE_LR_AUC_PR   = 0.017
BASELINE_GP_AUC_ROC  = 0.6715   # best genetic programming formula (method 3)
BASELINE_GP_AUC_PR   = 0.0179
FUNCTIONAL_CORR   = 0.999        # correlation threshold for functional dedup


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Parsing
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(raw: str) -> str:
    """
    Normalize a raw formula string so eval_formula_scores() can execute it.

    Transformations applied:
    - Lowercase all CBC variable names (model may use uppercase)
    - Replace np.sqrt/log/abs and other variants with the bare names
      expected by eval_formula_scores (sqrt / log / abs)
    - Strip leading/trailing whitespace
    """
    f = raw.strip()

    # Lowercase known variable names; sort longest-first to avoid partial matches
    # (mchc before mch, etc.)
    for var in sorted(FEATURE_VARS, key=len, reverse=True):
        f = re.sub(rf"\b{re.escape(var)}\b", var, f, flags=re.IGNORECASE)

    # np.xxx() → bare name (eval env only has sqrt / log / abs)
    f = re.sub(r"\bnp\.sqrt\b",  "sqrt",  f)
    f = re.sub(r"\bnp\.log1p\b", "log",   f)
    f = re.sub(r"\bnp\.log\b",   "log",   f)
    f = re.sub(r"\bnp\.abs\b",   "abs",   f)
    f = re.sub(r"\bmath\.sqrt\b","sqrt",  f)
    f = re.sub(r"\bmath\.log\b", "log",   f)
    f = re.sub(r"\bmath\.fabs\b","abs",   f)
    # Bare variants already in correct form; log1p / ln → log
    f = re.sub(r"\blog1p\b",     "log",   f)
    f = re.sub(r"\bln\b",        "log",   f)

    return f


def _contains_feature(text: str) -> bool:
    """True if text contains at least one CBC feature variable."""
    low = text.lower()
    return any(re.search(rf"\b{v}\b", low) for v in FEATURE_VARS)


def _looks_like_expression(text: str) -> bool:
    """
    Heuristic: does this look like a mathematical expression (not prose)?
    Requires at least one operator or function call.
    """
    return bool(re.search(r"[\+\-\*\/\(\)\*\*]|sqrt|log|abs", text))


def _strip_label_prefix(line: str) -> str:
    """Remove labels like 'FORMULA:', '1.', 'Formula 3:' from line start."""
    # FORMULA: ... or FORMULA 1: ...
    line = re.sub(r"^\s*FORMULA\s*\d*\s*[:.)]\s*", "", line, flags=re.IGNORECASE)
    # Numbered list: 1. or 1) or (1)
    line = re.sub(r"^\s*[\(\[]?\d+[\.\)\]]\s*", "", line)
    return line.strip()


def _extract_from_code_block(text: str) -> list[str]:
    """Extract lines from fenced code blocks (``` ... ```)."""
    candidates = []
    for block in re.findall(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL):
        for line in block.splitlines():
            line = line.strip()
            if _contains_feature(line) and _looks_like_expression(line):
                # Strip assignment: score = ... or formula = ...
                line = re.sub(r"^\s*\w+\s*=\s*", "", line)
                candidates.append(line)
    return candidates


def parse_formulas(raw_text: str) -> list[str]:
    """
    Extract candidate formula strings from a single LLM response.

    Strategy (in priority order):
    1. Lines explicitly labelled 'FORMULA:'
    2. Lines inside fenced code blocks that contain features
    3. Any line that contains a feature variable AND an arithmetic operator
       (catch-all for unlabelled outputs)

    Returns a list of normalized formula strings.
    """
    candidates: list[str] = []
    seen_raw: set[str] = set()

    def _add(raw_line: str) -> None:
        norm = _normalize(_strip_label_prefix(raw_line))
        if norm and norm not in seen_raw and _contains_feature(norm):
            seen_raw.add(norm)
            candidates.append(norm)

    lines = raw_text.splitlines()

    # Pass 1: FORMULA: labelled lines
    for line in lines:
        if re.match(r"\s*FORMULA\s*\d*\s*[:.]", line, re.IGNORECASE):
            _add(line)

    # Pass 2: code blocks
    for expr in _extract_from_code_block(raw_text):
        _add(expr)

    # Pass 3: catch-all — any line with a feature and an operator, not already found
    if not candidates:
        for line in lines:
            stripped = _strip_label_prefix(line)
            if _contains_feature(stripped) and _looks_like_expression(stripped):
                # Skip lines that are clearly prose (very long, contain full stops mid-text)
                if len(stripped) < 300:
                    _add(stripped)

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_formula(formula: str, sample_df: pd.DataFrame, features: list[str]) -> str | None:
    """
    Quick-validate a formula on a small sample.

    Returns the (possibly unchanged) formula string if valid, or None with a
    reason string for logging. Uses eval_formula_scores() from utils.py.

    Rejects:
    - Syntax / eval errors
    - Produces None (too many NaN/inf)
    - Constant output (no discriminative signal)
    """
    scores = eval_formula_scores(formula, sample_df, features)
    if scores is None:
        return None
    if np.std(scores) < 1e-10:
        return None  # constant — no discriminative signal
    return formula


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Deduplication
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate(formulas: list[str], sample_df: pd.DataFrame, features: list[str]) -> list[str]:
    """
    Remove exact-string duplicates first, then functional duplicates.

    Functional deduplication: evaluate each formula on sample_df, then
    compute pairwise Pearson correlation; if two formulas correlate above
    FUNCTIONAL_CORR they are considered equivalent — keep the shorter one.
    """
    # Exact dedup
    seen: set[str] = set()
    unique: list[str] = []
    for f in formulas:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    if len(unique) <= 1:
        return unique

    # Evaluate on sample
    score_matrix: list[np.ndarray] = []
    kept: list[str] = []
    for f in unique:
        s = eval_formula_scores(f, sample_df, features)
        if s is not None and np.std(s) > 1e-10:
            score_matrix.append(s)
            kept.append(f)

    if len(kept) <= 1:
        return kept

    # Greedy functional dedup: keep formula unless it correlates ≥ threshold
    # with an already-kept formula
    scores_arr = np.column_stack(score_matrix)  # shape (n_rows, n_formulas)
    n = len(kept)
    keep_mask = [True] * n

    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            x, y = scores_arr[:, i], scores_arr[:, j]
            # Use absolute correlation (sign flip = same formula negated)
            corr = abs(np.corrcoef(x, y)[0, 1])
            if corr >= FUNCTIONAL_CORR:
                # Keep the shorter formula string
                if len(kept[j]) < len(kept[i]):
                    keep_mask[i] = False
                    break
                else:
                    keep_mask[j] = False

    return [f for f, k in zip(kept, keep_mask) if k]


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_top(results_df: pd.DataFrame, n: int = 10) -> str:
    """Return a formatted top-N table string."""
    top = results_df.head(n)
    header = (
        f"{'Rank':<5} {'AUC-ROC':>8} {'AUC-PR':>8} "
        f"{'P@R25':>7} {'P@R50':>7} {'P@R75':>7} "
        f"{'F1':>6} {'F2':>6}  formula"
    )
    sep = "-" * 130
    rows = [header, sep]
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        rows.append(
            f"{rank:<5} {row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
            f"{row['precision_at_recall_25']:>7.4f} {row['precision_at_recall_50']:>7.4f} "
            f"{row['precision_at_recall_75']:>7.4f} "
            f"{row['f1']:>6.4f} {row['f2']:>6.4f}  {row['formula']}"
        )
    return "\n".join(rows)


def _write_report(lines: list[str]) -> None:
    text = "\n".join(lines)
    REPORT_FILE.write_text(text, encoding="utf-8")
    print(text)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ensure_dir(OUT_DIR)

    # ── Load raw outputs ───────────────────────────────────────────────────────
    if not RAW_FILE.exists():
        print(f"[ERROR] Raw outputs not found: {RAW_FILE}")
        print("Run method4_generate.py first (or --dry-run to test prompts).")
        sys.exit(1)

    with open(RAW_FILE, encoding="utf-8") as f:
        raw_outputs: list[dict] = json.load(f)

    ok_outputs = [r for r in raw_outputs if r.get("status") == "ok" and r.get("raw_text")]
    print(f"Loaded {len(raw_outputs)} raw entries  ({len(ok_outputs)} successful)\n")

    # ── Load data (for validation + full eval) ─────────────────────────────────
    print("Loading data...")
    df, features = load_data()
    train_df, test_df = get_splits(df)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}\n")

    # ── Stage 1: Parse ─────────────────────────────────────────────────────────
    print("=== Stage 1: Parsing ===")
    parsed_records: list[dict] = []
    n_raw_total = 0

    for entry in ok_outputs:
        extracted = parse_formulas(entry["raw_text"])
        n_raw_total += len(extracted)
        for formula in extracted:
            parsed_records.append({
                "formula":       formula,
                "source_run_id": entry.get("run_id", ""),
                "config_name":   entry.get("config_name", ""),
                "strategy":      entry.get("strategy", ""),
                "temperature":   entry.get("temperature"),
            })

    print(f"Extracted {n_raw_total} candidate formula(s) from {len(ok_outputs)} LLM responses")

    # Distinct formula strings (cross-response exact dedup for accurate reporting)
    distinct_formulas = list(dict.fromkeys(r["formula"] for r in parsed_records))
    print(f"Distinct formula strings  : {len(distinct_formulas)}")

    # Save all parsed formulas (with source metadata, may include duplicates)
    with open(PARSED_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed_records, f, indent=2)
    print(f"Saved {PARSED_FILE}")

    # ── Stage 2: Validate (on distinct strings only) ───────────────────────────
    print("\n=== Stage 2: Validation ===")
    valid_formulas: list[str] = []
    invalid_count = 0

    for formula in distinct_formulas:
        result = validate_formula(formula, train_df, features)
        if result is not None:
            valid_formulas.append(result)
        else:
            invalid_count += 1

    print(f"Valid: {len(valid_formulas)}  |  Rejected: {invalid_count}")

    # ── Stage 3: Deduplicate ───────────────────────────────────────────────────
    print("\n=== Stage 3: Deduplication ===")
    before_dedup = len(valid_formulas)
    unique_formulas = deduplicate(valid_formulas, train_df, features)
    print(f"Before dedup: {before_dedup}  |  After dedup: {len(unique_formulas)}")

    # ── Stage 4: Full evaluation ───────────────────────────────────────────────
    print(f"\n=== Stage 4: Full evaluation ({len(unique_formulas)} formulas) ===")
    results = []
    skipped = 0

    for i, formula in enumerate(unique_formulas, 1):
        row = evaluate_formula_full(formula, train_df, test_df, features)
        if row is None:
            skipped += 1
        else:
            results.append(row)
        if i % 10 == 0 or i == len(unique_formulas):
            print(f"  {i}/{len(unique_formulas)}  valid={len(results)}  skipped={skipped}")

    if not results:
        print("\n[WARN] No formulas survived full evaluation. Check raw_outputs.json quality.")
        _write_report([
            "PARSING REPORT",
            "=" * 60,
            f"Raw LLM responses processed : {len(ok_outputs)}",
            f"Candidate formulas extracted: {n_raw_total}",
            f"Valid after pre-validation  : {len(valid_formulas)}",
            f"Unique after deduplication  : {len(unique_formulas)}",
            f"Survived full evaluation    : 0",
            "",
            "[WARN] No usable formulas found.",
        ])
        return

    results_df = (
        pd.DataFrame(results)
        .sort_values("auc_pr", ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {RESULTS_FILE}")

    # ── Reports ────────────────────────────────────────────────────────────────
    best = results_df.iloc[0]
    beats_lr_roc = best["auc_roc"] > BASELINE_LR_AUC_ROC
    beats_lr_pr  = best["auc_pr"]  > BASELINE_LR_AUC_PR
    beats_gp_roc = best["auc_roc"] > BASELINE_GP_AUC_ROC
    beats_gp_pr  = best["auc_pr"]  > BASELINE_GP_AUC_PR

    top_table = _print_top(results_df, n=10)

    summary_lines = [
        "METHOD 4 — LLM-GENERATED BIOMARKER FORMULAS",
        "=" * 70,
        "",
        "PARSING FUNNEL",
        "-" * 40,
        f"  Raw LLM responses processed : {len(ok_outputs)}",
        f"  Candidate formulas extracted: {n_raw_total}",
        f"  Valid after pre-validation  : {len(valid_formulas)}",
        f"  Unique after deduplication  : {len(unique_formulas)}",
        f"  Survived full evaluation    : {len(results_df)}",
        "",
        "BASELINES",
        "-" * 40,
        f"  All-features LR  AUC-ROC={BASELINE_LR_AUC_ROC:.4f}  AUC-PR={BASELINE_LR_AUC_PR:.4f}",
        f"  Best GP formula  AUC-ROC={BASELINE_GP_AUC_ROC:.4f}  AUC-PR={BASELINE_GP_AUC_PR:.4f}",
        "",
        "BEST FORMULA",
        "-" * 40,
        f"  {best['formula']}",
        f"  AUC-ROC={best['auc_roc']:.4f}  AUC-PR={best['auc_pr']:.4f}",
        f"  Beats LR   AUC-ROC: {'YES' if beats_lr_roc else 'NO'}  AUC-PR: {'YES' if beats_lr_pr else 'NO'}",
        f"  Beats GP   AUC-ROC: {'YES' if beats_gp_roc else 'NO'}  AUC-PR: {'YES' if beats_gp_pr else 'NO'}",
        "",
        "AUC-PR DISTRIBUTION",
        "-" * 40,
        f"  max    : {results_df['auc_pr'].max():.4f}",
        f"  99th   : {results_df['auc_pr'].quantile(0.99):.4f}",
        f"  median : {results_df['auc_pr'].median():.4f}",
        f"  min    : {results_df['auc_pr'].min():.4f}",
        f"  # beating LR  ({BASELINE_LR_AUC_PR}): "
        f"{(results_df['auc_pr'] > BASELINE_LR_AUC_PR).sum()}",
        f"  # beating GP  ({BASELINE_GP_AUC_PR}): "
        f"{(results_df['auc_pr'] > BASELINE_GP_AUC_PR).sum()}",
        "",
        "TOP 10 BY AUC-PR",
        "-" * 40,
        top_table,
    ]

    summary_text = "\n".join(summary_lines)
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")
    print(f"\nSaved {SUMMARY_FILE}")

    # Parsing report (separate file for thesis documentation)
    report_lines = [
        "PARSING REPORT — method4_evaluate.py",
        "=" * 60,
        "",
        "INPUT",
        f"  File           : {RAW_FILE}",
        f"  Total entries  : {len(raw_outputs)}",
        f"  Successful runs: {len(ok_outputs)}",
        "",
        "STAGE 1 — PARSING",
        f"  Candidates extracted (incl. cross-run duplicates): {n_raw_total}",
        f"  Distinct formula strings                         : {len(distinct_formulas)}",
        "",
        f"STAGE 2 — VALIDATION (full train set, {len(train_df):,} rows)",
        f"  Valid   : {len(valid_formulas)}",
        f"  Rejected: {invalid_count}",
        "",
        f"STAGE 3 — DEDUPLICATION (corr_threshold={FUNCTIONAL_CORR}, full train set)",
        f"  Before: {before_dedup}",
        f"  After : {len(unique_formulas)}",
        f"  Removed: {before_dedup - len(unique_formulas)}",
        "",
        "STAGE 4 — FULL EVALUATION",
        f"  Evaluated : {len(unique_formulas)}",
        f"  Valid     : {len(results_df)}",
        f"  Skipped   : {skipped}",
        "",
        "NORMALIZATION RULES APPLIED",
        textwrap.dedent("""\
          - CBC variable names lowercased (e.g. RDW → rdw)
          - np.sqrt() → sqrt()
          - np.log1p() / np.log() / ln() / log1p() → log()
          - np.abs() → abs()
          - Assignment stripped (score = ...) from code-block lines
          - FORMULA: / numbered-list prefixes stripped
        """),
    ]
    _write_report(report_lines)

    print(f"\nMethod 4 evaluation complete. {len(results_df)} formulas evaluated.")


if __name__ == "__main__":
    main()
