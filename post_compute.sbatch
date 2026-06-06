#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-2:00:00
#SBATCH --job-name post_compute
#SBATCH --output cluster_results/post_compute_%x_%j.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Post-compute pipeline: bootstrap CIs (MIMIC-IV + NHANES), NHANES + EHRSHOT eval, dashboard, forest plot.
# Launched by run_all.sh with --dependency=afterok:<cpu_job_id>.
# Set SKIP_NHANES=1 for diseases without NHANES data (t1d, t2d).
# Set SKIP_EHRSHOT=1 to skip BigQuery EHRSHOT steps (e.g. no BQ credentials).
# Set EHRSHOT_KEY_FILE to a GCP service-account JSON path; leave unset to use ADC.
# Set NHANES_DIR to the directory holding NHANES XPT files (default: data/nhanes).

: "${DISEASE:?DISEASE not set}"
SKIP_NHANES=${SKIP_NHANES:-0}
SKIP_EHRSHOT=${SKIP_EHRSHOT:-0}
NHANES_DIR=${NHANES_DIR:-data/nhanes}
EHRSHOT_KEY_FILE=${EHRSHOT_KEY_FILE:-""}

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Disease:          $DISEASE"
echo "Skip NHANES:      $SKIP_NHANES"
echo "Skip EHRSHOT:     $SKIP_EHRSHOT"
echo "NHANES dir:       $NHANES_DIR"
echo "EHRSHOT key file: ${EHRSHOT_KEY_FILE:-(ADC)}"
echo "Start:            $(date)"
echo "=========================================="

set -e

cd ~/UnderGradProject
mkdir -p cluster_results

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python
echo "Python: $($PYTHON --version)"

EXTRA_FLAGS=""
[[ "$SKIP_NHANES"  == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --skip-nhanes"
[[ "$SKIP_EHRSHOT" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --skip-ehrshot"
[[ -n "$EHRSHOT_KEY_FILE" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --ehrshot-key-file $EHRSHOT_KEY_FILE"

$PYTHON -u src/post_compute.py \
    --disease "$DISEASE" \
    --n-bootstrap 1000 \
    --nhanes-dir "$NHANES_DIR" \
    $EXTRA_FLAGS

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
