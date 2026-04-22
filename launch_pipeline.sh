#!/bin/bash
# Usage: bash launch_pipeline.sh <disease> [--skip-bq]
# Submits the CPU job, then chains the GPU job to run after it succeeds.

DISEASE=${1:?Usage: bash launch_pipeline.sh <disease> [--skip-bq]}
SKIP_BQ=${2:-""}

CPU_JOB=$(sbatch --parsable --export=DISEASE="$DISEASE",SKIP_BQ="$SKIP_BQ" pipeline_cpu.sbatch)
echo "Submitted CPU job $CPU_JOB for disease=$DISEASE"

GPU_JOB=$(sbatch --parsable --dependency=afterok:"$CPU_JOB" --export=DISEASE="$DISEASE" pipeline_gpu.sbatch)
echo "Submitted GPU job $GPU_JOB for disease=$DISEASE (depends on $CPU_JOB)"
