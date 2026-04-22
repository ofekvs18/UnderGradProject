#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-14:00:00
#SBATCH --job-name biomarker
#SBATCH --output results/pipeline_%x_%j.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

: "${DISEASE:?DISEASE not set — use launch_pipeline.sh}"
SKIP_BQ=${SKIP_BQ:-""}
START_FROM=${START_FROM:-"bq"}  # bq | sanity | method1 | method2 | method3 | method4

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Disease: $DISEASE"
echo "Start: $(date)"
echo "START_FROM: $START_FROM"
echo "=========================================="

set -e

cd ~/UnderGradProject
mkdir -p results

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

STEPS=(bq sanity method1 method2 method3 method4)
REACHED=0
for s in "${STEPS[@]}"; do
    [[ "$s" == "$START_FROM" ]] && REACHED=1
    [[ "$REACHED" == "0" ]] && continue

    case "$s" in
        bq)
            if [ "$SKIP_BQ" != "--skip-bq" ]; then
                run_step "BQ extraction" $PYTHON -u src/run_pipeline.py disease="$DISEASE"
            else
                echo "Skipping BigQuery extraction."
            fi
            ;;
        sanity)  run_step "sanity check"      $PYTHON -u src/sanity_check.py --disease "$DISEASE" ;;
        method1) run_step "method1 threshold" $PYTHON -u src/method_threshold.py --disease "$DISEASE" ;;
        method2) run_step "method2 random"    $PYTHON -u src/method2_random_formula.py --disease "$DISEASE" ;;
        method3) run_step "method3 GP"        $PYTHON -u src/method3_gp.py --disease "$DISEASE" ;;
        method4) run_step "method4 LLM"       $PYTHON -u src/method4_llm.py all --disease "$DISEASE" ;;
    esac
done

TOTAL=$(( SECONDS - PIPELINE_START ))
echo ""
echo "=========================================="
echo "End: $(date)"
echo "Total runtime: $(( TOTAL / 3600 ))h $(( (TOTAL % 3600) / 60 ))m $(( TOTAL % 60 ))s"
echo "=========================================="
