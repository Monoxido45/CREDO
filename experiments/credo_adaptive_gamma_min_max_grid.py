import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from credo.credal_cp import CredalCPRegressor


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(125)
torch.manual_seed(125)


def make_gap_epistemic_few_middle(n, noise_std=0.1, p_middle=0.01):
    n_mid = max(1, int(n * p_middle))
    n_side = (n - n_mid) // 2

    x_left = np.random.uniform(-1.0, -0.2, size=n_side)
    x_mid = np.random.uniform(-0.2, 0.2, size=n_mid)
    x_right = np.random.uniform(0.2, 1.0, size=n_side)
    x = np.concatenate([x_left, x_mid, x_right])

    mu = np.zeros_like(x)

    mask_left = x < -0.2
    mu[mask_left] = -0.5 + 1.5 * (x[mask_left] + 0.8) - 1.0 * (x[mask_left] + 0.8) ** 2

    mask_mid = (x >= -0.2) & (x <= 0.2)
    mu[mask_mid] = 0.2 * np.sin(8 * x[mask_mid])

    mask_right = x > 0.2
    mu[mask_right] = 0.7 + 0.8 * (x[mask_right] - 0.5) ** 2 - 0.3 * (x[mask_right] - 0.5) ** 3

    eps = np.random.normal(scale=noise_std, size=len(x))
    y = mu + eps

    df = pd.DataFrame({"x": x, "y": y})
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


def fit_credo_model(
    X_train,
    y_train,
    X_cal,
    y_cal,
    alpha,
    gamma_min,
    gamma_max,
    tau=1.0,
    k=50,
    n_mcmc=1000,
):
    model = CredalCPRegressor(
        nc_type="Quantile",
        base_model="BART",
        alpha=alpha,
        adaptive_gamma=True,
        gamma=gamma_min,
    )

    model.fit(
        X_train,
        y_train,
        progressbar=True,
        n_cores=4,
        n_MCMC=n_mcmc,
        alpha_bart=0.985,
        k=k,
    )

    model.calibrate(
        X_cal,
        y_cal,
        N_samples_MC=n_mcmc,
        gamma_max=gamma_max,
        gamma_min=gamma_min,
        tau=tau,
    )

    return model


def plot_interval_grid(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    y_test,
    gamma_mins,
    gamma_maxs,
    alpha=0.1,
    tau=1.0,
    k=50,
    n_mcmc=1000,
):
    X_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
    xx = X_grid.ravel()

    fig, axes = plt.subplots(
        len(gamma_mins),
        len(gamma_maxs),
        figsize=(5 * len(gamma_maxs), 3.8 * len(gamma_mins)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fitted_models = {}

    for row, gamma_min in enumerate(gamma_mins):
        for col, gamma_max in enumerate(gamma_maxs):
            if gamma_max <= gamma_min:
                raise ValueError(
                    f"gamma_max must be greater than gamma_min, got gamma_min={gamma_min} and gamma_max={gamma_max}"
                )

            model = fit_credo_model(
                X_train=X_train,
                y_train=y_train,
                X_cal=X_cal,
                y_cal=y_cal,
                alpha=alpha,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                tau=tau,
                k=k,
                n_mcmc=n_mcmc,
            )
            fitted_models[(gamma_min, gamma_max)] = model

            pred_grid = np.asarray(model.predict(X_grid))
            lower = pred_grid[:, 0]
            upper = pred_grid[:, 1]
            center = 0.5 * (lower + upper)

            ax = axes[row, col]
            ax.scatter(X_test.ravel(), y_test.ravel(), s=12, color="k", alpha=0.35, zorder=3)
            ax.fill_between(xx, lower, upper, color="C0", alpha=0.25)
            ax.plot(xx, center, color="C0", lw=1.2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.35, 1.25)
            ax.set_title(rf"$\gamma_{{min}}={gamma_min:.3f}$, $\gamma_{{max}}={gamma_max:.2f}$")

            if row == len(gamma_mins) - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")

    fig.suptitle(
        rf"CREDO-BART adaptive gamma intervals ($\alpha={alpha:.2f}$, $\tau={tau:.2f}$, $k={k}$)",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    interval_path = os.path.join(RESULTS_DIR, "credo_adaptive_gamma_min_max_interval_grid.png")
    plt.savefig(interval_path, dpi=300, bbox_inches="tight")
    plt.show()

    return fitted_models, X_grid, interval_path


def plot_gamma_grid(
    fitted_models,
    X_grid,
    gamma_mins,
    gamma_maxs,
    tau,
):
    xx = X_grid.ravel()

    fig, axes = plt.subplots(
        len(gamma_mins),
        len(gamma_maxs),
        figsize=(5 * len(gamma_maxs), 3.0 * len(gamma_mins)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row, gamma_min in enumerate(gamma_mins):
        for col, gamma_max in enumerate(gamma_maxs):
            model = fitted_models[(gamma_min, gamma_max)]
            gamma_grid = model.compute_gamma(
                X_grid,
                gamma_max=gamma_max,
                gamma_min=gamma_min,
                tau=tau,
            )

            ax = axes[row, col]
            ax.plot(xx, gamma_grid, color="C4", lw=2.0)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(rf"$\gamma_{{min}}={gamma_min:.3f}$, $\gamma_{{max}}={gamma_max:.2f}$")

            if row == len(gamma_mins) - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel(r"$\gamma(x)$")

    fig.suptitle(
        rf"Adaptive gamma schedules ($\tau={tau:.2f}$)",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    gamma_path = os.path.join(RESULTS_DIR, "credo_adaptive_gamma_min_max_schedule_grid.png")
    plt.savefig(gamma_path, dpi=300, bbox_inches="tight")
    plt.show()

    return gamma_path


def main():
    data = make_gap_epistemic_few_middle(n=1500, noise_std=0.1, p_middle=0.01)

    train, rest = train_test_split(data, test_size=0.5, random_state=42)
    cal, test = train_test_split(rest, test_size=0.5, random_state=42)

    X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
    y_train = train["y"].values.astype(np.float32)
    X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
    y_cal = cal["y"].values.astype(np.float32)
    X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
    y_test = test["y"].values.astype(np.float32)

    alpha = 0.10
    tau = 1.0
    k = 50
    gamma_mins = [0.01, 0.05, 0.15]
    gamma_maxs = [0.75, 0.85, 0.95]

    fitted_models, X_grid, interval_path = plot_interval_grid(
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        gamma_mins=gamma_mins,
        gamma_maxs=gamma_maxs,
        alpha=alpha,
        tau=tau,
        k=k,
        n_mcmc=1000,
    )

    gamma_path = plot_gamma_grid(
        fitted_models=fitted_models,
        X_grid=X_grid,
        gamma_mins=gamma_mins,
        gamma_maxs=gamma_maxs,
        tau=tau,
    )

    print(f"Saved interval grid to {interval_path}")
    print(f"Saved gamma grid to {gamma_path}")


if __name__ == "__main__":
    main()
