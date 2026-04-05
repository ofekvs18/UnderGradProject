#!/bin/bash
### GP Experiment — LARGE_CONFIG (pop=500, gen=100)
### Estimated runtime: 4-8 hours depending on node load

#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name gp_large
#SBATCH --output gp_large-%J.out
#SBATCH --mail-user=ofek@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

### Print job info ###
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Config: LARGE (pop=500, gen=100)"
echo "=========================================="

### Setup environment ###
module load anaconda
source activate biomarkers-discovery-env

### Run GP with LARGE config ###
cd ~/UnderGradProject
python -u src/method3_gp_large.py

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
