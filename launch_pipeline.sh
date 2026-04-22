#!/bin/bash
# Usage: bash launch_pipeline.sh <disease> [--skip-bq]
# Submits the CPU job, then chains the GPU job to run after it succeeds.

DISEASE=${1:?Usage: bash launch_pipeline.sh <disease> [--skip-bq]}
SKIP_BQ=${2:-""}

CPU_JOB=$(sbatch --parsable pipeline_cpu.sbatch "$DISEASE" "$SKIP_BQ")
echo "Submitted CPU job $CPU_JOB for disease=$DISEASE"

GPU_JOB=$(sbatch --parsable --dependency=afterok:"$CPU_JOB" pipeline_gpu.sbatch "$DISEASE")
echo "Submitted GPU job $GPU_JOB for disease=$DISEASE (depends on $CPU_JOB)"
