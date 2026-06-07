#!/bin/bash
# run_seeded_gp.sh — submit one CPU job per disease to run seeded GP only.
# Skips diseases with no seed files. No GPU or POST dependencies.
#
# Usage:
#   bash run_seeded_gp.sh               # live submission
#   bash run_seeded_gp.sh --dry-run     # print sbatch commands, no submission

DRY_RUN=0
[[ "$1" == "--dry-run" ]] && DRY_RUN=1

PYTHON=/home/ofekvi/.conda/envs/biomarkers/bin/python
SPLIT_SALT=${SPLIT_SALT:-""}

for disease in ra crhn psr lup t1d t2d; do
    SEED_DIR="data/llm_seeds/$disease"
    if [[ ! -d "$SEED_DIR" ]]; then
        echo "[$disease] no seed dir — skip"
        continue
    fi
    shopt -s nullglob
    seeds=("$SEED_DIR"/*.csv)
    shopt -u nullglob
    if [[ ${#seeds[@]} -eq 0 ]]; then
        echo "[$disease] seed dir empty — skip"
        continue
    fi

    echo "[$disease] submitting seeded GP (${#seeds[@]} seed files) …"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] sbatch --export=DISEASE=$disease,SPLIT_SALT=$SPLIT_SALT seeded_gp.sbatch"
    else
        jid=$(sbatch --parsable \
            --export=DISEASE="$disease",SPLIT_SALT="$SPLIT_SALT" \
            seeded_gp.sbatch)
        echo "  job $jid"
    fi
done

echo ""
echo "Done."
