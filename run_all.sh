#!/bin/bash

# Folder containing your disease yaml files
CONF_DIR="conf/disease"

# Pipeline control (can override from environment)
START_FROM=${START_FROM:-"bq"}    # bq | sanity | method1 | method2 | method3 | method4
END_AT=${END_AT:-"method4"}       # bq | sanity | method1 | method2 | method3 | method4
SPLIT_SALT=${SPLIT_SALT:-""}      # e.g. _seed2 — labels the split table and CSV

# Create a logs directory if it doesn't exist
mkdir -p logs

for file in "$CONF_DIR"/*.yaml
do
    # Extract name (e.g., ra.yaml -> ra)
    disease_slug=$(basename "$file" .yaml)

    echo "Submitting: $disease_slug (START_FROM=$START_FROM, END_AT=$END_AT, SPLIT_SALT=${SPLIT_SALT:-(default)})"

    sbatch --job-name="biomarker_${disease_slug}" \
           --export=DISEASE="$disease_slug",START_FROM="$START_FROM",END_AT="$END_AT",SPLIT_SALT="$SPLIT_SALT" \
           pipeline.sbatch
done
