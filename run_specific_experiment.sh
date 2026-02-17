#!/bin/bash

# First argument is the range of cores (e.g., 0-3)
CORE_RANGE=$1
# The remaining arguments are the datasets (e.g., concrete energy)
shift # Remove the first argument from the list to leave only the datasets
DATASETS=("$@")

# Safety check
if [ -z "$CORE_RANGE" ] || [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Usage: ./lancar.sh <cores> <dataset1> <dataset2> ..."
    echo "Example: ./lancar.sh 0-3 concrete energy"
    exit 1
fi

# Environment Settings (JAX/Torch/CatBoost)
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4

for DATASET in "${DATASETS[@]}"; do
    echo "Launching $DATASET on cores $CORE_RANGE..."
    
    # -u forces immediate print to the log
    taskset -c $CORE_RANGE python -u comparisons/credo_comparisons.py \
        -n_rep 30 \
        -dataset "$DATASET" \
        -outlier_analysis True \
        -outlier_same_time True > "log_$DATASET.txt" 2>&1 &
    
    # Captures the PID for control
    PID=$!
    echo "Process for $DATASET started (PID: $PID)"
done

echo "-------------------------------------------------------"
echo "All processes have been sent. To see the progress:"
echo "tail -f log_<dataset>.txt"