#!/bin/bash

N_REP=50
BASE_MODEL="qnn"
GAMMA=0.2
GAMMA_MIN=0.1
GAMMA_MAX=0.9
TAU_GAMMA=1.0
CREDO_DROPOUT=0.1
RESULTS_TAG=""
SCARCITY_METHOD="both"
SCARCITY_IFOREST_N_ESTIMATORS=200
SCARCITY_IFOREST_MAX_SAMPLES="auto"
SCARCITY_IFOREST_MAX_FEATURES=1.0
SCARCITY_IFOREST_BOOTSTRAP=""
CREDO_DROPOUT_TRAINING=False
CREDO_FIRST_QNN_BACKBONE=""
OUTLIER_ANALYSIS=True
OUTLIER_SAME_TIME=True
RUN_LABEL="same_time"
OUTLIER_DETECTOR="both"
OUTLIER_CONTAMINATION=0.05
INLIER_SIZE=0.2
IFOREST_N_ESTIMATORS=200
IFOREST_MAX_SAMPLES="auto"
IFOREST_MAX_FEATURES=1.0
QNN_HIDDEN_LAYERS=(64 64 32)
DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-rep|-n_rep|-n)
            N_REP="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --gamma-min|--gamma_min)
            GAMMA_MIN="$2"
            shift 2
            ;;
        --gamma-max|--gamma_max)
            GAMMA_MAX="$2"
            shift 2
            ;;
        --tau-gamma|--tau_gamma)
            TAU_GAMMA="$2"
            shift 2
            ;;
        --credo-dropout|--credo_dropout)
            CREDO_DROPOUT="$2"
            shift 2
            ;;
        --qnn-hidden-layers|--qnn_hidden_layers)
            QNN_HIDDEN_LAYERS=("$2" "$3" "$4")
            shift 4
            ;;
        --results-tag|--results_tag)
            RESULTS_TAG="$2"
            shift 2
            ;;
        --scarcity-method|--scarcity_method)
            SCARCITY_METHOD="$2"
            shift 2
            ;;
        --scarcity-iforest-n-estimators|--scarcity_iforest_n_estimators)
            SCARCITY_IFOREST_N_ESTIMATORS="$2"
            shift 2
            ;;
        --scarcity-iforest-max-samples|--scarcity_iforest_max_samples)
            SCARCITY_IFOREST_MAX_SAMPLES="$2"
            shift 2
            ;;
        --scarcity-iforest-max-features|--scarcity_iforest_max_features)
            SCARCITY_IFOREST_MAX_FEATURES="$2"
            shift 2
            ;;
        --scarcity-iforest-bootstrap|--scarcity_iforest_bootstrap)
            SCARCITY_IFOREST_BOOTSTRAP="--scarcity_iforest_bootstrap"
            shift
            ;;
        --credo-dropout-training|--credo_dropout_training)
            CREDO_DROPOUT_TRAINING="$2"
            shift 2
            ;;
        --credo-first-qnn-backbone|--credo_first_qnn_backbone)
            CREDO_FIRST_QNN_BACKBONE="--credo-first-qnn-backbone"
            shift
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
        --same-time)
            OUTLIER_ANALYSIS=True
            OUTLIER_SAME_TIME=True
            RUN_LABEL="same_time"
            shift
            ;;
        --outlier-only)
            OUTLIER_ANALYSIS=True
            OUTLIER_SAME_TIME=False
            RUN_LABEL="outlier_only"
            shift
            ;;
        --full)
            OUTLIER_ANALYSIS=False
            OUTLIER_SAME_TIME=False
            RUN_LABEL="full"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options] dataset1 [dataset2 ...]"
            echo "Options:"
            echo "  --n-rep N                         Number of repetitions (default: 50)"
            echo "  --base-model MODEL                Base model for competitors: qnn, qnn_mc, catboost, rfqr (default: qnn)"
            echo "  --gamma VALUE                     Fixed CREDO gamma (default: 0.2)"
        echo "  --gamma-min VALUE                 Adaptive CREDO gamma_min (default: 0.1)"
            echo "  --gamma-max VALUE                 Adaptive CREDO gamma_max (default: 0.9)"
            echo "  --tau-gamma VALUE                 Adaptive CREDO tau_gamma (default: 1.0)"
            echo "  --credo-dropout VALUE             MC-dropout rate for the CREDO envelope (default: 0.2)"
            echo "  --qnn-hidden-layers H1 H2 H3      Shared QNN widths (default: 128 128 64)"
            echo "  --results-tag TAG                 Optional suffix for result files/folders"
            echo "  --scarcity-method METHOD          knn, isolation_forest, or both (default: both)"
            echo "  --scarcity-iforest-n-estimators N Isolation Forest trees for scarcity score (default: 200)"
            echo "  --scarcity-iforest-max-samples V  Isolation Forest max_samples for scarcity score (default: auto)"
            echo "  --scarcity-iforest-max-features V Isolation Forest max_features for scarcity score (default: 1.0)"
            echo "  --scarcity-iforest-bootstrap       Bootstrap samples for scarcity Isolation Forest"
            echo "  --credo-dropout-training BOOL      Keep CREDO dropout active during training (default: False)"
            echo "  --credo-first-qnn-backbone         Fit the original CREDO QNN first and inherit it in baselines"
            echo "  --outlier-only                    Run only outlier coverage/ratio metrics"
            echo "  --same-time                       Run full experiment plus both outlier detectors (default)"
            echo "  --full                            Run full experiment only, including SMIS and coverage"
            echo "  --outlier-detector METHOD         lof, isolation_forest, or both (default: both)"
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
            DATASETS+=("$1")
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

RESULTS_TAG_ARGS=()
if [ -n "$RESULTS_TAG" ]; then
    RESULTS_TAG_ARGS=(-results_tag "$RESULTS_TAG")
fi

# 8 blocks of 4 cores
CORES=("0-3" "4-7" "8-11" "12-15" "16-19" "20-23" "24-27" "28-31")

for i in "${!DATASETS[@]}"; do
    CORE_IDX=$((i % 8))
    CORE_RANGE=${CORES[$CORE_IDX]}
    DATASET=${DATASETS[$i]}
    
    LOG_TAG=""
    if [ -n "$RESULTS_TAG" ]; then
        LOG_TAG="_${RESULTS_TAG}"
    fi
    SCARCITY_LOG_TAG=""
    if [ "$SCARCITY_METHOD" = "isolation_forest" ]; then
        SCARCITY_LOG_TAG="_scarcity_iforest"
    elif [ "$SCARCITY_METHOD" = "both" ]; then
        SCARCITY_LOG_TAG="_scarcity_both"
    fi
    LOG_FILE="log_${DATASET}_${BASE_MODEL}${LOG_TAG}${SCARCITY_LOG_TAG}_${RUN_LABEL}_${OUTLIER_DETECTOR}.txt"

    echo "Starting $RUN_LABEL experiment on dataset $DATASET with $OUTLIER_DETECTOR on cores $CORE_RANGE..."
    
    # JAX and Backend configurations
    # OMP_NUM_THREADS limits the internal parallelism of linear algebra libraries
    export JAX_PLATFORMS=cpu
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export VECLIB_MAX_THREADS=4  # For some versions of Torch/NumPy
    export NUMEXPR_NUM_THREADS=4
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
    mkdir -p "$MPLCONFIGDIR"
    
    # taskset -c defines CPU affinity
    # The '&' at the end sends the process to the background to run in parallel
    nohup setsid taskset -c $CORE_RANGE python comparisons/credo_comparisons.py \
        -n_rep $N_REP \
        -dataset "$DATASET" \
        -base_model "$BASE_MODEL" \
        -gamma "$GAMMA" \
        -gamma_min "$GAMMA_MIN" \
        -gamma_max "$GAMMA_MAX" \
            -tau_gamma "$TAU_GAMMA" \
		-credo_dropout "$CREDO_DROPOUT" \
		-qnn_hidden_layers "${QNN_HIDDEN_LAYERS[@]}" \
        "${RESULTS_TAG_ARGS[@]}" \
        -scarcity_method "$SCARCITY_METHOD" \
        -scarcity_iforest_n_estimators "$SCARCITY_IFOREST_N_ESTIMATORS" \
        -scarcity_iforest_max_samples "$SCARCITY_IFOREST_MAX_SAMPLES" \
        -scarcity_iforest_max_features "$SCARCITY_IFOREST_MAX_FEATURES" \
        $SCARCITY_IFOREST_BOOTSTRAP \
        -credo_dropout_training "$CREDO_DROPOUT_TRAINING" \
        $CREDO_FIRST_QNN_BACKBONE \
        -outlier_analysis $OUTLIER_ANALYSIS \
        -outlier_same_time $OUTLIER_SAME_TIME \
        -outlier_detector "$OUTLIER_DETECTOR" \
        -outlier_contamination "$OUTLIER_CONTAMINATION" \
        -inlier_size "$INLIER_SIZE" \
        -iforest_n_estimators "$IFOREST_N_ESTIMATORS" \
        -iforest_max_samples "$IFOREST_MAX_SAMPLES" \
        -iforest_max_features "$IFOREST_MAX_FEATURES" > "$LOG_FILE" 2>&1 &
    
    echo "Process for $DATASET sent to the background (see $LOG_FILE)"
done

echo "All processes have been started. Use 'htop' to monitor the cores."
