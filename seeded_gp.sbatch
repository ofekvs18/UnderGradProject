#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-4:00:00
#SBATCH --job-name seeded_gp
#SBATCH --output cluster_results/seeded_gp_%x_%j.out
#SBATCH --mail-user=ofekvi@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

: "${DISEASE:?DISEASE not set}"
SPLIT_SALT=${SPLIT_SALT:-""}

echo "SLURM_JOBID=$SLURM_JOBID"
echo "Disease:    $DISEASE"
echo "Start:      $(date)"
echo "=========================================="

set -e

cd ~/UnderGradProject
mkdir -p cluster_results

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python

run_step() {
    local label=$1; shift
    echo ""
    echo "-- [$label] start: $(date) --"
    local t0=$SECONDS
    "$@"
    echo "-- [$label] done in $(( (SECONDS - t0) / 60 ))m $(( (SECONDS - t0) % 60 ))s --"
}

SEED_DIR="data/llm_seeds/$DISEASE"
if [[ ! -d "$SEED_DIR" ]]; then
    echo "No seed dir at $SEED_DIR — nothing to do."
    exit 0
fi

for seed_file in "$SEED_DIR"/*.csv; do
    [[ -f "$seed_file" ]] || continue
    seed_name=$(basename "$seed_file" .csv)
    run_step "seeded GP ($seed_name)" \
        $PYTHON -u src/method3_gp.py \
        --disease "$DISEASE" --split-salt "$SPLIT_SALT" \
        --seed-file "$seed_file"
done

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
