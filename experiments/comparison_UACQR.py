from experiments.uacqr import uacqr
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from credo.credal_cp import CredalCPRegressor
import numpy as np
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(125)
torch.manual_seed(125)

device = torch.device("cpu") 

def make_gap_epistemic_few_middle(n, noise_std=0.1, p_middle=0.01):
    n_mid = max(1, int(n * p_middle))
    n_side = (n - n_mid) // 2

    # Regions in x
    x_left  = np.random.uniform(-1.0, -0.2, size=n_side)
    x_mid   = np.random.uniform(-0.2,  0.2, size=n_mid)
    x_right = np.random.uniform( 0.2,  1.0, size=n_side)
    x = np.concatenate([x_left, x_mid, x_right])

    # True mean function (piecewise, with gap)
    mu = np.zeros_like(x)

    # left: smooth non-linear
    mask_left = x < -0.2
    mu[mask_left] = -0.5 + 1.5*(x[mask_left] + 0.8) - 1.0*(x[mask_left] + 0.8)**2

    # middle: wiggly but under-sampled
    mask_mid = (x >= -0.2) & (x <= 0.2)
    mu[mask_mid] = 0.2 * np.sin(8 * x[mask_mid])

    # right: different non-linear behavior
    mask_right = x > 0.2
    mu[mask_right] = 0.7 + 0.8*(x[mask_right] - 0.5)**2 - 0.3*(x[mask_right] - 0.5)**3

    # Small, constant noise
    eps = np.random.normal(scale=noise_std, size=len(x))
    y = mu + eps

    df = pd.DataFrame({"x": x, "y": y})
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return df

# Generate data
data = make_gap_epistemic_few_middle(n=1500, noise_std=0.1, p_middle=0.01)

# Train / cal / test split
train, rest = train_test_split(data, test_size=0.5, random_state=42)
cal, test   = train_test_split(rest,  test_size=0.5, random_state=42)

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
    nc_type = 'Quantile',
    base_model = "BART",
    alpha = 0.1,
    adaptive_gamma = True,
    gamma = 0.05,
)

# starting fitting
credal_CP_bart.fit(
    X_train, 
    Y_train,
    progressbar = True,
    n_cores = 4,
    n_MCMC = 1000,
    alpha_bart = 0.985,
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
    "max_features" : "sqrt",
    "min_samples_leaf": 5,
}


# fitting base estimator and UACQR
uacqr_results = uacqr(
    rfqr_params,
    bootstrapping_for_uacqrp=False,
    q_lower=alpha / 2 * 100,
    q_upper=(1 - alpha / 2) * 100,
    alpha = alpha,
    model_type=uacqr_params["model_type"],
    B=uacqr_params["B"],
    random_state=seed,
    uacqrs_bagging=False,
    uacqrs_agg=uacqr_params["uacqrs_agg"],
)

uacqr_results.fit(X_train, Y_train)
uacqr_results.calibrate(X_cal, Y_cal)
uacqr_pred_test = uacqr_results.predict(X_test)

# ============================================
# Plotting and predicting
# ============================================
X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
pred_bart = credal_CP_bart.predict(X_test_grid)
pred_cqr = uacqr_results.predict(X_test_grid)

# Simple concise plotting assuming predict() returns (N,2) arrays [lower, upper]
pred_bart = np.asarray(pred_bart)
lower_bart, upper_bart = pred_bart[:, 0], pred_bart[:, 1]
center_bart = 0.5 * (lower_bart + upper_bart)

xx = X_test_grid.ravel()
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
# Global sizes (will affect legends and any subsequent text)
fontsize = 16
plt.rcParams.update({
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    })

# =========================
# BART (top-left)
# =========================
ax = axes[0, 0]

ax.scatter(X_test.ravel(), Y_test.ravel(),
           s=30, color="k", alpha=0.8, zorder=3)

ax.fill_between(xx, lower_bart, upper_bart,
                color="C0", alpha=0.25)

ax.plot(xx, center_bart,
        color="C0", lw=1)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("CREDO", fontweight="bold")
ax.grid(True)

# =========================
# CQR (top-right)
# =========================
ax = axes[0, 1]

lower_cqr = pred_cqr["CQR"]["lower"]
upper_cqr = pred_cqr["CQR"]["upper"]

ax.scatter(X_test.ravel(), Y_test.ravel(),
           s=30, color="k", alpha=0.8, zorder=3)

ax.fill_between(xx, lower_cqr, upper_cqr,
                color="C1", alpha=0.25)

ax.plot(xx, 0.5 * (upper_cqr + lower_cqr),
        color="C1", lw=1)

ax.set_xlabel("x")
ax.set_title("CQR")
ax.grid(True)

# =========================
# CQR-r (bottom-left)
# =========================
ax = axes[1, 0]

lower_cqrr = pred_cqr["CQR-r"]["lower"]
upper_cqrr = pred_cqr["CQR-r"]["upper"]

ax.scatter(X_test.ravel(), Y_test.ravel(),
           s=30, color="k", alpha=0.8, zorder=3)

ax.fill_between(xx, lower_cqrr, upper_cqrr,
                color="C2", alpha=0.25)

ax.plot(xx, 0.5 * (upper_cqrr + lower_cqrr),
        color="C2", lw=1)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("CQR-r")
ax.grid(True)

# =========================
# UACQR (bottom-right)
# =========================
ax = axes[1, 1]

lower_uacqrs = pred_cqr["UACQR-S"]["lower"]
upper_uacqrs = pred_cqr["UACQR-S"]["upper"]

ax.scatter(X_test.ravel(), Y_test.ravel(),
           s=30, color="k", alpha=0.8, zorder=3)

ax.fill_between(xx, lower_uacqrs, upper_uacqrs,
                color="C3", alpha=0.25)

ax.plot(xx, 0.5 * (upper_uacqrs + lower_uacqrs),
        color="C3", lw=1)

ax.set_xlabel("x")
ax.set_title("UACQR-S")
ax.grid(True)

# =========================
# Save
# =========================
plt.tight_layout()
plt.show()
output_path = os.path.join(RESULTS_DIR, "bart_CQR_CQRr_UACQR_test.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()





