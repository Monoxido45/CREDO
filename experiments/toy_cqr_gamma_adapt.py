import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credo.utils import AdaptiveGammaCQR, average_coverage, compute_interval_length
from .uacqr import uacqr


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(125)

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


def summarize_intervals(name, intervals, y_true):
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    coverage = average_coverage(upper, lower, y_true)
    avg_length = np.mean(compute_interval_length(upper, lower))
    print(f"{name}: coverage={coverage:.3f}, avg_length={avg_length:.3f}")


def main():
    alpha = 0.1
    gamma = 0.1
    seed = 123

    data = make_gap_epistemic_few_middle(n=1500, noise_std=0.1, p_middle=0.01)
    train, rest = train_test_split(data, test_size=0.5, random_state=42)
    cal, test = train_test_split(rest, test_size=0.5, random_state=42)

    X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
    y_train = train["y"].values.astype(np.float32)
    X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
    y_cal = cal["y"].values.astype(np.float32)
    X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
    y_test = test["y"].values.astype(np.float32)
    X_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)

    rfqr_params = {
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
    }

    uacqr_results = uacqr(
        rfqr_params,
        bootstrapping_for_uacqrp=False,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        alpha=alpha,
        model_type="rfqr",
        B=100,
        random_state=seed,
        uacqrs_bagging=False,
        uacqrs_agg="std",
    )
    uacqr_results.fit(X_train, y_train)
    uacqr_results.calibrate(X_cal, y_cal)

    adaptive_cqr = AdaptiveGammaCQR(
        base_model=uacqr_results.cqr_base_model,
        type_model="rfqr",
        is_fitted=True,
        alpha=alpha,
        gamma=gamma,
        k=7,
    )
    adaptive_cqr.fit(X_train, y_train)
    adaptive_cqr.calibrate(
        X_cal,
        y_cal,
        gamma_max=0.95,
        gamma_min=0.025,
        tau=1.0,
    )

    uacqr_pred_test = uacqr_results.predict_uacqr(X_test)
    pred_cqr = uacqr_results.predict_uacqr(X_grid)
    pred_cqr_test_df = uacqr_results.predict_uacqr(X_test)
    pred_cqr_test = np.column_stack(
        (pred_cqr_test_df["CQR"]["lower"].to_numpy(), pred_cqr_test_df["CQR"]["upper"].to_numpy())
    )
    pred_adaptive_test = adaptive_cqr.predict(X_test)
    pred_adaptive_grid = adaptive_cqr.predict(X_grid)
    gamma_grid = adaptive_cqr.compute_gamma(X_grid, gamma_max=0.95, gamma_min=0.025, tau=1.0)

    summarize_intervals("CQR", pred_cqr_test, y_test)
    summarize_intervals("Adaptive-gamma CQR", pred_adaptive_test, y_test)

    lower_cqr = pred_cqr["CQR"]["lower"]
    upper_cqr = pred_cqr["CQR"]["upper"]
    center_cqr = 0.5 * (upper_cqr + lower_cqr)
    xx = X_grid.ravel()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, height_ratios=[3, 1])

    ax = axes[0]
    ax.scatter(X_test.ravel(), y_test.ravel(), s=30, color="k", alpha=0.8, zorder=3)
    ax.fill_between(xx, lower_cqr, upper_cqr, color="C1", alpha=0.25, label="CQR")
    ax.plot(xx, center_cqr, color="C1", lw=1)
    ax.fill_between(
        xx,
        pred_adaptive_grid[:, 0],
        pred_adaptive_grid[:, 1],
        color="C0",
        alpha=0.22,
        label="Adaptive-gamma CQR",
    )
    ax.plot(xx, 0.5 * (pred_adaptive_grid[:, 1] + pred_adaptive_grid[:, 0]), color="C0", lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CQR vs Adaptive-gamma CQR")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-1.35, 1.25)

    ax = axes[1]
    ax.plot(xx, gamma_grid, color="C4", lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel("gamma(x)")
    ax.set_title("Adaptive gamma schedule")
    ax.grid(True)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "toy_cqr_adaptive_gamma_rfqr.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    _ = X_train
    _ = y_train
    _ = X_cal
    _ = y_cal
    _ = uacqr_pred_test

if __name__ == "__main__":
    main()