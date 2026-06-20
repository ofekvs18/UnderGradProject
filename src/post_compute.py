"""
post_compute.py — Orchestrate all post-training steps for one disease.

Steps (in order):
  1. mimic_compute_ci    — Bootstrap CIs on MIMIC-IV test set (M1–M5 + LR)
  2. nhanes_data         — Build NHANES modeling CSV from XPT files  [--skip-nhanes]
  3. nhanes_evaluate     — Evaluate best formulas on NHANES external set  [--skip-nhanes]
  4. nhanes_compute_ci   — Bootstrap CIs on NHANES (M1–M5)  [--skip-nhanes]
  5. ehrshot_bq_data     — Extract EHRSHOT cohort from BigQuery  [--skip-ehrshot]
  6. ehrshot_evaluate    — Evaluate best formulas on EHRSHOT  [--skip-ehrshot]
  7. ehrshot_compute_ci  — Bootstrap CIs on EHRSHOT (M1–M5)  [--skip-ehrshot]
  8. build_dashboard_data — Aggregate master summaries into dashboard CSV
  9. plot_ci_forest      — Forest plot of AUC-PR CIs

Usage:
    python src/post_compute.py --disease ra
    python src/post_compute.py --disease ra --n-bootstrap 1000 --skip-nhanes
    python src/post_compute.py --disease ra --skip-ehrshot
    python src/post_compute.py --disease ra --ehrshot-key-file .secrets/bq_sa.json
    python src/post_compute.py --disease ra --steps 1 8 9   # run specific steps only

Designed to be called from run_all.sh or an sbatch job array.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# All paths relative to project root (where this script is invoked from)
PYTHON = sys.executable

STEPS = [
    {
        "id":      1,
        "name":    "mimic_compute_ci",
        "script":  "src/mimic_compute_ci.py",
        "args":    lambda d, ns: ["--disease", d, "--n-bootstrap", str(ns.n_bootstrap)],
        "nhanes":  False,
        "ehrshot": False,
    },
    {
        "id":      2,
        "name":    "nhanes_data",
        "script":  "src/nhanes_data.py",
        "args":    lambda d, ns: ["--disease", d, "--nhanes-dir", ns.nhanes_dir],
        "nhanes":  True,
        "ehrshot": False,
    },
    {
        "id":      3,
        "name":    "nhanes_evaluate",
        "script":  "src/nhanes_evaluate.py",
        "args":    lambda d, ns: ["--disease", d],
        "nhanes":  True,
        "ehrshot": False,
    },
    {
        "id":      4,
        "name":    "nhanes_compute_ci",
        "script":  "src/nhanes_compute_ci.py",
        "args":    lambda d, ns: ["--disease", d, "--n-bootstrap", str(ns.n_bootstrap)],
        "nhanes":  True,
        "ehrshot": False,
    },
    {
        "id":      5,
        "name":    "ehrshot_bq_data",
        "script":  "src/ehrshot_bq_data.py",
        "args":    lambda d, ns: (
            ["--disease", d, "--key-file", ns.ehrshot_key_file]
            if ns.ehrshot_key_file else ["--disease", d]
        ),
        "nhanes":  False,
        "ehrshot": True,
    },
    {
        "id":      6,
        "name":    "ehrshot_evaluate",
        "script":  "src/ehrshot_evaluate.py",
        "args":    lambda d, ns: ["--disease", d],
        "nhanes":  False,
        "ehrshot": True,
    },
    {
        "id":      7,
        "name":    "ehrshot_compute_ci",
        "script":  "src/ehrshot_compute_ci.py",
        "args":    lambda d, ns: ["--disease", d, "--n-bootstrap", str(ns.n_bootstrap)],
        "nhanes":  False,
        "ehrshot": True,
    },
    {
        "id":      8,
        "name":    "build_dashboard_data",
        "script":  "src/build_dashboard_data.py",
        "args":    lambda d, ns: ["--disease", d],
        "nhanes":  False,
        "ehrshot": False,
    },
    {
        "id":      9,
        "name":    "plot_ci_forest",
        "script":  "src/plot_ci_forest.py",
        "args":    lambda d, ns: ["--disease", d],
        "nhanes":  False,
        "ehrshot": False,
    },
]


def run_step(step, disease, ns):
    cmd = [PYTHON, step["script"]] + step["args"](disease, ns)
    print(f"\n{'='*70}")
    print(f"  Step {step['id']}: {step['name']}  [{disease}]")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n  --> {status}  ({elapsed:.1f}s)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Run all post-compute steps for a single disease"
    )
    parser.add_argument("--disease", required=True,
                        help="Disease slug, e.g. ra, lup, t1d, t2d, psr, crhn")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap iterations for CI steps (default: 500)")
    parser.add_argument("--skip-nhanes", action="store_true",
                        help="Skip NHANES steps (2–4) — useful when NHANES data is absent")
    parser.add_argument("--nhanes-dir", default="data/nhanes",
                        help="Root directory containing NHANES XPT files (default: data/nhanes)")
    parser.add_argument("--skip-ehrshot", action="store_true",
                        help="Skip EHRSHOT steps (5–6) — useful when BQ credentials are absent")
    parser.add_argument("--ehrshot-key-file", default="",
                        help="Path to GCP service-account JSON for BigQuery (default: ADC)")
    parser.add_argument("--steps", type=int, nargs="+",
                        help="Run only these step IDs, e.g. --steps 1 7 8")
    ns = parser.parse_args()

    active_steps = [
        s for s in STEPS
        if (ns.steps is None or s["id"] in ns.steps)
        and not (ns.skip_nhanes and s["nhanes"])
        and not (ns.skip_ehrshot and s["ehrshot"])
    ]

    print(f"\npost_compute: disease={ns.disease}  n_bootstrap={ns.n_bootstrap}")
    print(f"skip_nhanes={ns.skip_nhanes}  skip_ehrshot={ns.skip_ehrshot}")
    print(f"Steps to run: {[s['id'] for s in active_steps]}")

    results = {}
    for step in active_steps:
        ok = run_step(step, ns.disease, ns)
        results[step["name"]] = ok

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    all_ok = True
    for name, ok in results.items():
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}]  {name}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
