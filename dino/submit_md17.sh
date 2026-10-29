#!/bin/bash
# Submit all MD17 model variants as independent SLURM jobs.
# Usage: bash submit_md17.sh [molecule]   (default: aspirin)

MOLECULE=${1:-aspirin}

echo "Submitting MD17 jobs for molecule: $MOLECULE"

declare -A VARIANTS=(
    [baseline]="baseline:"
    [concat]="ect:concat"
    [film-final]="ect:film-final"
    [film-layer]="ect:film-layer"
    [adaln-layer]="ect:adaln-layer"
)

for NAME in "${!VARIANTS[@]}"; do
    SPEC="${VARIANTS[$NAME]}"
    MODEL="${SPEC%%:*}"
    COND="${SPEC##*:}"

    JID=$(sbatch --parsable \
        --export=ALL,MODEL="$MODEL",COND_TYPE="$COND",MOLECULE="$MOLECULE" \
        --output="logs/md17_${NAME}_${MOLECULE}_%j.txt" \
        run_md17.job)
    echo "  $NAME -> job $JID  (logs/md17_${NAME}_${MOLECULE}_${JID}.txt)"
done
