#!/bin/bash
# run_all.sh — submit one GPU + one CPU + one post-compute SLURM job per disease.
# CPU depends on GPU (afterok); post-compute depends on CPU (afterok).
# Post-compute runs CIs, NHANES eval, dashboard, and forest plot.
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

    # 3. Post-compute — CIs, NHANES eval, dashboard, forest plot
    #    NHANES steps are skipped for diseases without NHANES data (t1d, t2d).
    skip_nhanes=0
    [[ " $NHANES_DISEASES " != *" $disease "* ]] && skip_nhanes=1

    post_id=$(_submit "POST $disease" \
        --dependency=afterok:"$cpu_id" \
        --export=DISEASE="$disease",SKIP_NHANES="$skip_nhanes" \
        post_compute.sbatch)
    [[ "$DRY_RUN" == "1" ]] && echo "  → POST job ID: $post_id (dependency: afterok:$cpu_id, skip_nhanes=$skip_nhanes)"
done

echo ""
echo "Done submitting all jobs."
