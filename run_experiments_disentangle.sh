#!/bin/bash

# List of datasets or configurations you want to run
# Capture all arguments passed by the user
DATASETS=("$@")

# Check if the user passed any arguments
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: You need to pass at least one dataset."
    echo "Example: ./run_experiments.sh concrete energy wine"
    exit 1
fi

# Common configurations
N_REP=30
OUTLIER_ANALYSIS=True
OUTLIER_SAME_TIME=True

# 8 blocks of 4 cores
CORES=("0-3" "4-7" "8-11" "12-15" "16-19" "20-23" "24-27" "28-31")

for i in "${!DATASETS[@]}"; do
    CORE_IDX=$((i % 8))
    CORE_RANGE=${CORES[$CORE_IDX]}
    DATASET=${DATASETS[$i]}
    
    echo "Starting experiments on dataset $DATASET on cores $CORE_RANGE..."
    
    # JAX and Backend configurations
    # OMP_NUM_THREADS limits the internal parallelism of linear algebra libraries
    export JAX_PLATFORMS=cpu
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export VECLIB_MAX_THREADS=4  # For some versions of Torch/NumPy
    export NUMEXPR_NUM_THREADS=4
    
    # taskset -c defines CPU affinity
    # The '&' at the end sends the process to the background to run in parallel
    taskset -c $CORE_RANGE python comparisons/decomp_comparison.py \
        -n_rep $N_REP \
        -dataset "$DATASET" > "log_$DATASET.txt" 2>&1 &
    
    echo "Process for $DATASET sent to the background (see log_$DATASET.txt)"
done

echo "All processes have been started. Use 'htop' to monitor the cores."