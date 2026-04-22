#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name bp_cpu
#SBATCH --output results/pipeline_%x_%j.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

: "${DISEASE:?DISEASE not set — use launch_pipeline.sh}"
SKIP_BQ=${SKIP_BQ:-""}

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Disease: $DISEASE"
echo "Start: $(date)"
echo "=========================================="


module load anaconda
source activate biomarkers

which python
python --version
pip list | grep -E "google|omegaconf|gplearn"

cd ~/UnderGradProject
mkdir -p results

if [ "$SKIP_BQ" != "--skip-bq" ]; then
    python -u src/run_pipeline.py disease="$DISEASE"
else
    echo "Skipping BigQuery extraction."
fi

python -u src/sanity_check.py --disease "$DISEASE"

python -u src/method_threshold.py --disease "$DISEASE"

python -u src/method2_random_formula.py --disease "$DISEASE"

python -u src/method3_gp.py --disease "$DISEASE"

echo "=========================================="
echo "End: $(date)"
