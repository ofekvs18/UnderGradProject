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
echo "=========================================="

set -e  # exit immediately if any command fails

cd ~/UnderGradProject
mkdir -p results

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python
echo "Python: $($PYTHON --version)"

run_from() { [[ "$START_FROM" == "$1" || "$_RUNNING" == "1" ]] && _RUNNING=1 && return 0 || return 1; }
_RUNNING=""

if run_from bq; then
    if [ "$SKIP_BQ" != "--skip-bq" ]; then
        $PYTHON -u src/run_pipeline.py disease="$DISEASE"
    else
        echo "Skipping BigQuery extraction."
    fi
fi

run_from sanity  && $PYTHON -u src/sanity_check.py --disease "$DISEASE"
run_from method1 && $PYTHON -u src/method_threshold.py --disease "$DISEASE"
run_from method2 && $PYTHON -u src/method2_random_formula.py --disease "$DISEASE"
run_from method3 && $PYTHON -u src/method3_gp.py --disease "$DISEASE"
run_from method4 && $PYTHON -u src/method4_llm.py all --disease "$DISEASE"

echo "=========================================="
echo "End: $(date)"
