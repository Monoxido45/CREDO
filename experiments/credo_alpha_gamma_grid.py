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
    gamma,
    n_mcmc=1000,
):
    model = CredalCPRegressor(
        nc_type="Quantile",
        base_model="BART",
        alpha=alpha,
        adaptive_gamma=False,
        gamma=gamma,
    )

    model.fit(
        X_train,
        y_train,
        progressbar=True,
        n_cores=4,
        n_MCMC=n_mcmc,
        alpha_bart=0.985,
    )

    model.calibrate(
        X_cal,
        y_cal,
        N_samples_MC=n_mcmc,
    )

    return model


def plot_alpha_gamma_grid(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    y_test,
    alphas,
    gammas,
    n_mcmc=1000,
):
    X_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
    xx = X_grid.ravel()

    fig, axes = plt.subplots(
        len(gammas),
        len(alphas),
        figsize=(5 * len(alphas), 3.8 * len(gammas)),
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

    for row, gamma in enumerate(gammas):
        for col, alpha in enumerate(alphas):
            model = fit_credo_model(
                X_train=X_train,
                y_train=y_train,
                X_cal=X_cal,
                y_cal=y_cal,
                alpha=alpha,
                gamma=gamma,
                n_mcmc=n_mcmc,
            )

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
            ax.set_title(rf"$\alpha={alpha:.2f}$, $\gamma={gamma:.3f}$")

            if row == len(gammas) - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")

    fig.suptitle(
        "CREDO-BART fixed gamma grid",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = os.path.join(RESULTS_DIR, "credo_alpha_gamma_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return output_path


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

    alphas = [0.05, 0.15, 0.30]
    gammas = [0.025, 0.1, 0.20]

    output_path = plot_alpha_gamma_grid(
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        alphas=alphas,
        gammas=gammas,
        n_mcmc=1000,
    )

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
