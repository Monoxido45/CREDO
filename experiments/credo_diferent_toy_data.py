import sys
import os

#project_root = os.path.abspath("..")
#sys.path.append(project_root)

from .uacqr import uacqr
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# importing our package
from credo.credal_cp import CredalCPRegressor
import numpy as np

# For reproducibility
np.random.seed(125)
torch.manual_seed(125)

device = torch.device("cpu")  # change to "cuda" if desired

def make_variable_data_v3(n=800, std_dev=1/5, seed=42):

    np.random.seed(seed)

    # --------------------------
    # scarce region
    # --------------------------
    p_scarce = 0.02
    n_scarce = int(n * p_scarce)
    n_dense = n - n_scarce

    # dense region outside [0, 0.4]
    x_dense = np.random.uniform(low=-1, high=1, size=n_dense)

    x_dense = x_dense[
        (x_dense < 0) | (x_dense > 0.4)
    ]

    while len(x_dense) < n_dense:

        new_samples = np.random.uniform(
            low=-1,
            high=1,
            size=n_dense
        )

        new_samples = new_samples[
            (new_samples < 0) | (new_samples > 0.4)
        ]

        x_dense = np.concatenate([x_dense, new_samples])

    x_dense = x_dense[:n_dense]

    # sparse region
    x_scarce = np.random.uniform(
        low=0,
        high=0.4,
        size=n_scarce
    )

    # join
    x = np.concatenate([x_dense, x_scarce])
    np.random.shuffle(x)

    # --------------------------
    # true mean
    # --------------------------
    mu = (
        x**3
        + 2 * np.exp(-6 * (x - 0.3)**2)
    )

    # --------------------------
    # heteroscedastic variance
    # --------------------------
    sigma = np.zeros_like(x)

    # left region
    mask_low = (x <= -0.3)

    sigma[mask_low] = 0.1

    # middle-left
    mask_mid = (x > -0.3) & (x < 0)

    sigma[mask_mid] = (
        0.2
        + 0.15 * np.abs(np.sin(10 * x[mask_mid]))
    )

    # sparse region
    mask_sparse = (x >= 0) & (x <= 0.4)

    sigma[mask_sparse] = (
        0.05
        + 0.1 * np.abs(np.sin(12 * x[mask_sparse]))
    )

    # right region with increasing variability
    mask_right = (x > 0.4)

    sigma[mask_right] = (
        0.25
        + 0.5 * (x[mask_right] - 0.4)
        + 0.15 * np.abs(np.sin(8 * x[mask_right]))
    )

    # optional scaling
    sigma *= std_dev / (1/5)

    # --------------------------
    # generate observations
    # --------------------------
    y = mu + np.random.normal(
        scale=sigma,
        size=len(x)
    )

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "mu": mu,
        "sigma": sigma
    })

    return df


def main():
    # Generate data
    data = make_variable_data_v3(n=1500)

    # Train / cal / test split
    train, rest = train_test_split(data, test_size=0.5, random_state=42)
    cal, test = train_test_split(rest, test_size=0.5, random_state=42)

    # ============================================
    # Training, calibration, and prediction
    # ============================================
    X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
    Y_train = train["y"].values.astype(np.float32)

    X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
    Y_cal = cal["y"].values.astype(np.float32)

    X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
    Y_test = test["y"].values.astype(np.float32)

    # Train and calibrate credal CP regions
    credal_CP_bart = CredalCPRegressor(
        nc_type='Quantile',
        base_model="BART",
        alpha=0.1,
        adaptive_gamma=True,
        gamma=0.05,
    )

    # starting fitting
    credal_CP_bart.fit(
        X_train,
        Y_train,
        progressbar=True,
        n_cores=4,
        n_MCMC=1000,
        alpha_bart=0.985,
    )

    # calibration of the credal CPs
    bart_cutoff = credal_CP_bart.calibrate(X_cal, Y_cal, N_samples_MC=1000)

    # CQR and CQR-r based on Random Forests
    alpha = 0.1
    seed = 123

    # random forest parameters
    uacqr_params = {
        "model_type": "rfqr",
        "B": 100,
        "uacqrs_agg": "std",
        "base_model_type": "Quantile",
    }

    rfqr_params = {
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
    }

    # fitting base estimator and UACQR
    uacqr_results = uacqr(
        rfqr_params,
        bootstrapping_for_uacqrp=False,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        alpha=alpha,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=seed,
        uacqrs_bagging=False,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
    )

    uacqr_results.fit(X_train, Y_train)
    uacqr_results.calibrate(X_cal, Y_cal)
    uacqr_pred_test = uacqr_results.predict_uacqr(X_test)

    X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
    pred_cqr = uacqr_results.predict_uacqr(X_test_grid)
    y_pred_bart, aleatoric, epistemic = credal_CP_bart.predict(
        X_test_grid, disentangle=True
    )

    # ============================================
    # Predictions BART (intervals + uncertainties)
    # ============================================
    pred_bart = np.asarray(y_pred_bart)
    lower_bart = pred_bart[:, 0]
    upper_bart = pred_bart[:, 1]
    center_bart = 0.5 * (lower_bart + upper_bart)

    # Normalized decomposition
    total = aleatoric + epistemic
    new_aleatoric = aleatoric / total
    new_epistemic = epistemic / total

    xx = X_test_grid.ravel()

    # ============================================
    # Figure 2x2
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fontsize = 24
    plt.rcParams.update({
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    })

    # =====================================================
    # (0,0) - BART Prediction Interval (CREDO)
    # =====================================================
    ax = axes[0, 0]

    ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, zorder=3)
    ax.fill_between(xx, lower_bart, upper_bart, color="C0", alpha=0.25)
    ax.plot(xx, center_bart, color="C0", lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CREDO", fontweight="bold")
    ax.grid(True)
    ax.set_ylim(-1.35, 3)

    # =====================================================
    # (0,1) - CQR Prediction Interval
    # =====================================================
    ax = axes[0, 1]

    lower_cqr = pred_cqr["CQR"]["lower"]
    upper_cqr = pred_cqr["CQR"]["upper"]

    ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, zorder=3)
    ax.fill_between(xx, lower_cqr, upper_cqr, color="C1", alpha=0.25)
    ax.plot(xx, 0.5 * (upper_cqr + lower_cqr), color="C1", lw=1)
    ax.set_xlabel("x")
    ax.set_title("CQR")
    ax.grid(True)
    ax.set_ylim(-1.35, 3)

    # =====================================================
    # (1,0) - BART Uncertainty Decomposition
    # =====================================================
    ax = axes[1, 0]

    ax.plot(xx, aleatoric, label="Aleatoric", color="C2")
    ax.plot(xx, epistemic, label="Epistemic", color="C1")
    ax.set_title("Uncertainty Decomposition (Raw)")
    ax.set_ylabel("Epistemic/Aleatoric length")
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # =====================================================
    # (1,1) - Empty
    # =====================================================
    axes[1, 1].axis("off")

    _ = bart_cutoff
    _ = uacqr_pred_test
    _ = new_aleatoric
    _ = new_epistemic

    plt.tight_layout()
    plt.savefig("CREDO_CQR_decomposition_new_data.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
