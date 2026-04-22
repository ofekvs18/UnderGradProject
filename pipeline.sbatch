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

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Disease: $DISEASE"
echo "Start: $(date)"
echo "=========================================="

cd ~/UnderGradProject
mkdir -p results

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python
echo "Python: $($PYTHON --version)"

if [ "$SKIP_BQ" != "--skip-bq" ]; then
    $PYTHON -u src/run_pipeline.py disease="$DISEASE"
else
    echo "Skipping BigQuery extraction."
fi

$PYTHON -u src/sanity_check.py --disease "$DISEASE"

$PYTHON -u src/method_threshold.py --disease "$DISEASE"

$PYTHON -u src/method2_random_formula.py --disease "$DISEASE"

$PYTHON -u src/method3_gp.py --disease "$DISEASE"

$PYTHON -u src/method4_llm.py --disease "$DISEASE"

echo "=========================================="
echo "End: $(date)"
