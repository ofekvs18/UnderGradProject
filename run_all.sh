#!/bin/bash
# run_all.sh — submit one GPU + one CPU SLURM job per disease.
# CPU job depends on GPU finishing (afterok). NHANES evaluate is submitted
# as an additional CPU job for diseases where data/<disease>_nhanes_data.csv exists.
#
# Usage:
#   bash run_all.sh            # live submission
#   bash run_all.sh --dry-run  # print sbatch commands, no submission

CONF_DIR="conf/disease"
SPLIT_SALT=${SPLIT_SALT:-""}
DRY_RUN=0

# t1d and t2d are intentionally excluded — NHANES cannot distinguish them.
NHANES_DISEASES="ra crhn psr lup"

for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python

# Fake ID counter stored in a temp file so subshells share it.
_ID_FILE=$(mktemp)
echo "10000" > "$_ID_FILE"
trap 'rm -f "$_ID_FILE"' EXIT

# _submit <label> [sbatch args...]
# Prints the submitted job ID to stdout; prints a status line to stderr.
_submit() {
    local label=$1; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        local id=$(( $(cat "$_ID_FILE") + 1 ))
        echo "$id" > "$_ID_FILE"
        echo "[dry-run] sbatch $*" >&2
        echo "$id"
    else
        local jid
        jid=$(sbatch --parsable "$@")
        echo "  [$label] job $jid" >&2
        echo "$jid"
    fi
}

for file in "$CONF_DIR"/*.yaml; do
    disease=$(basename "$file" .yaml)
    echo ""
    echo "=== $disease ==="

    # 1. GPU job — MedGemma generate
    gpu_id=$(_submit "GPU $disease" \
        --export=DISEASE="$disease",SPLIT_SALT="$SPLIT_SALT" \
        medgemma_generate.sbatch)
    [[ "$DRY_RUN" == "1" ]] && echo "  → GPU job ID: $gpu_id"

    # 2. CPU job — depends on GPU finishing
    cpu_id=$(_submit "CPU $disease" \
        --dependency=afterok:"$gpu_id" \
        --export=DISEASE="$disease",SPLIT_SALT="$SPLIT_SALT" \
        pipeline_cpu.sbatch)
    [[ "$DRY_RUN" == "1" ]] && echo "  → CPU job ID: $cpu_id (dependency: afterok:$gpu_id)"

    # 3. NHANES evaluate — only ra, crhn, psr, lup (t1d/t2d excluded by config)
    if [[ " $NHANES_DISEASES " == *" $disease "* ]]; then
        nhanes_id=$(_submit "NHANES $disease" \
            --dependency=afterok:"$cpu_id" \
            --partition=main --time=0-01:00:00 \
            --job-name="nhanes_${disease}" \
            --output="results/nhanes_${disease}_%j.out" \
            --cpus-per-task=2 --mem=8G \
            --wrap="cd ~/UnderGradProject && $PYTHON -u src/nhanes_evaluate.py --disease $disease")
        [[ "$DRY_RUN" == "1" ]] && echo "  → NHANES job ID: $nhanes_id (dependency: afterok:$cpu_id)"
    fi
done

echo ""
echo "Done submitting all jobs."
