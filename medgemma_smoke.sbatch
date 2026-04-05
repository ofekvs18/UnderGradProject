#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-00:30:00
#SBATCH --job-name mg_smoke
#SBATCH --output mg_smoke-%J.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Start: $(date)"

module load anaconda
source activate medgemma

# If you set a custom cache dir, uncomment:
# export HF_HOME=/storage/ofekvi/hf_cache
# export TRANSFORMERS_CACHE=/storage/ofekvi/hf_cache

cd ~/UnderGradProject
python -u src/medgemma_smoke_test.py

echo "End: $(date)"