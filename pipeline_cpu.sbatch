#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-8:00:00
#SBATCH --job-name biomarker_cpu
#SBATCH --output cluster_results/pipeline_biomarker_%x_%j.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# CPU-only pipeline: data extraction, sanity, M1, M2, M3, M4-evaluate, post-processing
# M4 generate (GPU) must have already run — raw_outputs.json must exist.
# Launched by run_all.sh with --dependency=afterok:<gpu_job_id>.

: "${DISEASE:?DISEASE not set}"
SPLIT_SALT=${SPLIT_SALT:-""}

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Disease:    $DISEASE"
echo "Split salt: ${SPLIT_SALT:-(default)}"
echo "Start:      $(date)"
echo "=========================================="

set -e

cd ~/UnderGradProject
mkdir -p results cluster_results

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python
echo "Python: $($PYTHON --version)"

PIPELINE_START=$SECONDS

run_step() {
    local label=$1; shift
    echo ""
    echo "-- [$label] start: $(date) --"
    local t0=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - t0 ))
    echo "-- [$label] done in $(( elapsed / 60 ))m $(( elapsed % 60 ))s --"
}

run_step "extract cohort"    $PYTHON -u src/run_pipeline.py          disease="$DISEASE" split_salt="$SPLIT_SALT"
run_step "sanity check"      $PYTHON -u src/sanity_check.py          --disease "$DISEASE" --split-salt "$SPLIT_SALT"
run_step "method1 threshold" $PYTHON -u src/method_threshold.py      --disease "$DISEASE" --split-salt "$SPLIT_SALT"
run_step "method2 random"    $PYTHON -u src/method2_random_formula.py --disease "$DISEASE" --split-salt "$SPLIT_SALT"
run_step "method3 GP"        $PYTHON -u src/method3_gp.py             --disease "$DISEASE" --split-salt "$SPLIT_SALT"
run_step "method4 evaluate"  $PYTHON -u src/method4_llm.py evaluate   --disease "$DISEASE" --split-salt "$SPLIT_SALT"
run_step "cross-method corr" $PYTHON -u src/cross_method_correlation.py --disease "$DISEASE"
run_step "compare methods"   $PYTHON -u src/compare_methods.py

TOTAL=$(( SECONDS - PIPELINE_START ))
echo ""
echo "=========================================="
echo "End:           $(date)"
echo "Total runtime: $(( TOTAL / 3600 ))h $(( (TOTAL % 3600) / 60 ))m $(( TOTAL % 60 ))s"
echo "=========================================="
