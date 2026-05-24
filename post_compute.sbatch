#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-2:00:00
#SBATCH --job-name post_compute
#SBATCH --output cluster_results/post_compute_%x_%j.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Post-compute pipeline: bootstrap CIs (MIMIC-IV + NHANES), dashboard, forest plot.
# Launched by run_all.sh with --dependency=afterok:<cpu_job_id>.
# Set SKIP_NHANES=1 for diseases without NHANES data (t1d, t2d).

: "${DISEASE:?DISEASE not set}"
SKIP_NHANES=${SKIP_NHANES:-0}

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Disease:     $DISEASE"
echo "Skip NHANES: $SKIP_NHANES"
echo "Start:       $(date)"
echo "=========================================="

set -e

cd ~/UnderGradProject
mkdir -p cluster_results

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python
echo "Python: $($PYTHON --version)"

EXTRA_FLAGS=""
[[ "$SKIP_NHANES" == "1" ]] && EXTRA_FLAGS="--skip-nhanes"

$PYTHON -u src/post_compute.py \
    --disease "$DISEASE" \
    --n-bootstrap 1000 \
    $EXTRA_FLAGS

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
