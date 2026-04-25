#!/bin/bash

MODE=${MODE:-both}
N_REP=${N_REP:-10}
N_MCMC=${N_MCMC:-1000}
EPOCHS=${EPOCHS:-2000}
DATASETS=("$@")

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("airfoil" "concrete" "winered" "winewhite")
fi

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib

poetry run python comparisons/gamma_ablation.py \
    --datasets "${DATASETS[@]}" \
    --mode "$MODE" \
    --n_rep "$N_REP" \
    --n_mcmc "$N_MCMC" \
    --epochs "$EPOCHS"

poetry run python comparisons/plot_gamma_ablation.py \
    --datasets "${DATASETS[@]}" \
    --study "$MODE"
