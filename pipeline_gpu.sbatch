#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name bp_gpu
#SBATCH --output results/pipeline_%x_%j.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

DISEASE=${1:?Usage: sbatch pipeline_gpu.sbatch <disease>}

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Disease: $DISEASE"
echo "Start: $(date)"
echo "=========================================="

module load anaconda
source activate biomarkers

cd ~/UnderGradProject

python -u src/method4_llm.py --disease "$DISEASE"

echo "=========================================="
echo "End: $(date)"
