"""
run_full_pipeline.py — End-to-end pipeline runner.

Runs, in order:
  1. BigQuery cohort extraction  (run_pipeline.py via Hydra)
  2. Sanity check                (sanity_check.py)
  3. Method 1 — Threshold        (method_threshold.py)
  4. Method 2 — Random Formula   (method2_random_formula.py)
  5. Method 3 — Genetic Program  (method3_gp.py)
  6. Method 4 — LLM              (method4_llm.py)

Usage:
    python src/run_full_pipeline.py --disease ra
    python src/run_full_pipeline.py --disease t1d --skip-bq
    python src/run_full_pipeline.py --disease ra --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
PYTHON = str(Path(sys.executable))


def run_step(label, cmd, dry_run=False):
    print()
    print("=" * 70)
    print(f"STEP: {label}")
    print("=" * 70)
    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR: step '{label}' exited with code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"\n  Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run the full biomarker pipeline end-to-end")
    parser.add_argument("--disease", default="ra", help="Disease slug matching a conf/disease/<slug>.yaml (e.g. ra, t1d)")
    parser.add_argument("--skip-bq", action="store_true", help="Skip BigQuery extraction (data CSV already present)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    disease = args.disease
    dry_run = args.dry_run

    print("=" * 70)
    print(f"Full Biomarker Pipeline — disease: {disease}")
    if args.skip_bq:
        print("  BigQuery extraction: SKIPPED")
    if dry_run:
        print("  Mode: DRY RUN")
    print("=" * 70)

    # Step 1: BigQuery extraction
    if not args.skip_bq:
        run_step(
            "BigQuery cohort extraction",
            [PYTHON, str(SRC_DIR / "run_pipeline.py"), f"disease={disease}"],
            dry_run=dry_run,
        )

    # Step 2: Sanity check
    run_step(
        "Sanity check (leakage test)",
        [PYTHON, str(SRC_DIR / "sanity_check.py"), "--disease", disease],
        dry_run=dry_run,
    )

    # Step 3: Method 1 — Threshold
    run_step(
        "Method 1 — Threshold optimization",
        [PYTHON, str(SRC_DIR / "method_threshold.py"), "--disease", disease],
        dry_run=dry_run,
    )

    # Step 4: Method 2 — Random formula
    run_step(
        "Method 2 — Random formula search",
        [PYTHON, str(SRC_DIR / "method2_random_formula.py"), "--disease", disease],
        dry_run=dry_run,
    )

    # Step 5: Method 3 — Genetic programming
    run_step(
        "Method 3 — Genetic programming",
        [PYTHON, str(SRC_DIR / "method3_gp.py"), "--disease", disease],
        dry_run=dry_run,
    )

    # Step 6: Method 4 — LLM
    run_step(
        "Method 4 — LLM-guided search",
        [PYTHON, str(SRC_DIR / "method4_llm.py"), "--disease", disease],
        dry_run=dry_run,
    )

    # Step 7: Cross-method correlation
    run_step(
        "Cross-method score vector correlation",
        [PYTHON, str(SRC_DIR / "cross_method_correlation.py"), "--disease", disease],
        dry_run=dry_run,
    )

    print()
    print("=" * 70)
    print(f"All steps complete for disease: {disease}")
    print("=" * 70)


if __name__ == "__main__":
    main()
