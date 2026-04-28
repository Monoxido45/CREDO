#!/bin/bash

N_REP=30
OUTLIER_DETECTOR="lof"
OUTLIER_CONTAMINATION=0.05
INLIER_SIZE=0.2
IFOREST_N_ESTIMATORS=200
IFOREST_MAX_SAMPLES="auto"
IFOREST_MAX_FEATURES=1.0
DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-rep|-n_rep|-n)
            N_REP="$2"
            shift 2
            ;;
        --outlier-detector|--detector)
            OUTLIER_DETECTOR="$2"
            shift 2
            ;;
        --outlier-contamination|--contamination)
            OUTLIER_CONTAMINATION="$2"
            shift 2
            ;;
        --inlier-size)
            INLIER_SIZE="$2"
            shift 2
            ;;
        --iforest-n-estimators)
            IFOREST_N_ESTIMATORS="$2"
            shift 2
            ;;
        --iforest-max-samples)
            IFOREST_MAX_SAMPLES="$2"
            shift 2
            ;;
        --iforest-max-features)
            IFOREST_MAX_FEATURES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options] dataset1 [dataset2 ...]"
            echo "Options:"
            echo "  --n-rep N                         Number of repetitions (default: 30)"
            echo "  --outlier-detector METHOD         lof or isolation_forest (default: lof)"
            echo "  --outlier-contamination VALUE     Detector contamination (default: 0.05)"
            echo "  --inlier-size VALUE               Fraction of non-outliers used as inliers (default: 0.2)"
            echo "  --iforest-n-estimators N          Isolation Forest trees (default: 200)"
            echo "  --iforest-max-samples VALUE       Isolation Forest max_samples (default: auto)"
            echo "  --iforest-max-features VALUE      Isolation Forest max_features (default: 1.0)"
            echo "Example:"
            echo "  $0 --outlier-detector isolation_forest concrete airfoil winewhite"
            exit 0
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ && ${#DATASETS[@]} -eq 0 ]]; then
                N_REP="$1"
            else
                DATASETS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: You need to pass at least one dataset."
    echo "Usage: $0 [options] dataset1 [dataset2 ...]"
    echo "Example: $0 --outlier-detector isolation_forest concrete airfoil winewhite"
    exit 1
fi

# 8 blocks of 4 cores
CORES=("0-3" "4-7" "8-11" "12-15" "16-19" "20-23" "24-27" "28-31")

for i in "${!DATASETS[@]}"; do
    CORE_IDX=$((i % 8))
    CORE_RANGE=${CORES[$CORE_IDX]}
    DATASET=${DATASETS[$i]}
    
    LOG_FILE="log_disentangle_${DATASET}_${OUTLIER_DETECTOR}.txt"

    echo "Starting disentanglement on dataset $DATASET with $OUTLIER_DETECTOR on cores $CORE_RANGE..."
    
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
        -dataset "$DATASET" \
        -outlier_detector "$OUTLIER_DETECTOR" \
        -outlier_contamination "$OUTLIER_CONTAMINATION" \
        -inlier_size "$INLIER_SIZE" \
        -iforest_n_estimators "$IFOREST_N_ESTIMATORS" \
        -iforest_max_samples "$IFOREST_MAX_SAMPLES" \
        -iforest_max_features "$IFOREST_MAX_FEATURES" > "$LOG_FILE" 2>&1 &
    
    echo "Process for $DATASET sent to the background (see $LOG_FILE)"
done

echo "All processes have been started. Use 'htop' to monitor the cores."
