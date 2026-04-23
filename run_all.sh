#!/bin/bash

# Folder containing your disease yaml files
CONF_DIR="conf/disease"

# Create a logs directory if it doesn't exist
mkdir -p logs

for file in "$CONF_DIR"/*.yaml
do
    # Extract name (e.g., ra.yaml -> ra)
    disease_slug=$(basename "$file" .yaml)
    
    echo "Submitting: $disease_slug"
    
    # Passing the variables into your sbatch template
    sbatch --job-name="biomarker_${disease_slug}" \
           --export=DISEASE="$disease_slug" \
           pipeline.sbatch
done