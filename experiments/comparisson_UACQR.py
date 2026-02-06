from uacqr import uacqr
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# importing our package
from credal_cp.credal_cp import CredalCPRegressor
import numpy as np

###### PASTA
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# For reproducibility
np.random.seed(125)
torch.manual_seed(125)

device = torch.device("cpu")  # change to "cuda" if desired

def make_gap_epistemic_few_middle(n, noise_std=0.1, p_middle=0.01):
    """
    Example where:
      - left and right regions are well sampled
      - middle region has very few points (high epistemic uncertainty)
      - noise is small and constant (mostly aleatoric)
      - mean is different on each side, forcing extrapolation
    """
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
    base_gamma = 0.05,
)

# starting fitting
credal_CP_bart.fit(
    X_train, 
    Y_train,
    progressbar = True,
    n_cores = 4,
    n_MCMC = 1000,
    alpha_bart = 0.98,
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

catboost_params = {
    "iterations": 1000,
    "learning_rate": 1e-3,
    "depth": 6,  # default value
    "l2_leaf_reg": 3,  # default value
    "random_strength": 1,  # default value
    "bagging_temperature": 1,  # default value
    "od_type": "Iter",
    "od_wait": 50,
    "use_best_model": False,
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
uacqr_pred_test = uacqr_results.predict_uacqr(X_test)

#Average_coverage_cqrr = average_coverage(
     #uacqr_pred_test["CQR-r"]["upper"], uacqr_pred_test["CQR-r"]["lower"], Y_test)
#Average_coverage_cqr = average_coverage(
    #uacqr_pred_test["CQR"]["upper"], uacqr_pred_test["CQR"]["lower"], Y_test)


# ============================================
# Plotting and predicting
# ============================================
X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)
pred_bart = credal_CP_bart.predict(X_test_grid)
pred_cqr = uacqr_results.predict_uacqr(X_test_grid)

# Simple concise plotting assuming predict() returns (N,2) arrays [lower, upper]
pred_bart = np.asarray(pred_bart)
lower_bart, upper_bart = pred_bart[:, 0], pred_bart[:, 1]
center_bart = 0.5 * (lower_bart + upper_bart)

xx = X_test_grid.ravel()
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# =========================
# BART (top-left)
# =========================
ax = axes[0, 0]

ax.scatter(X_test.ravel(), Y_test.ravel(),
           s=30, color="k", alpha=0.8, label="Test data", zorder=3)

ax.fill_between(xx, lower_bart, upper_bart,
                color="C0", alpha=0.25, label="BART interval")

ax.plot(xx, center_bart,
        color="C0", lw=1, label="BART center")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("BART")
ax.legend()
ax.grid(True)

# =========================
# CQR (top-right)
# =========================
ax = axes[0, 1]

x_test_plot = X_test.ravel()
idx = np.argsort(x_test_plot)

x_sorted = x_test_plot[idx]
lower_sorted = uacqr_pred_test["CQR"]["lower"][idx]
upper_sorted = uacqr_pred_test["CQR"]["upper"][idx]

ax.scatter(x_test_plot, Y_test.ravel(),
           s=30, color="k", alpha=0.8, zorder=3)

ax.fill_between(x_sorted, lower_sorted, upper_sorted,
                color="C1", alpha=0.25, label="CQR interval")

ax.plot(x_sorted, 0.5 * (upper_sorted + lower_sorted),
        color="C1", lw=1, label="CQR center")

ax.set_xlabel("x")
ax.set_title("CQR")
ax.legend()
ax.grid(True)

# =========================
# CQR-r (bottom-left)
# =========================
ax = axes[1, 0]

lower_sorted = uacqr_pred_test["CQR-r"]["lower"][idx]
upper_sorted = uacqr_pred_test["CQR-r"]["upper"][idx]

ax.scatter(x_test_plot, Y_test.ravel(),
           s=30, color="k", alpha=0.8, zorder=3)

ax.fill_between(x_sorted, lower_sorted, upper_sorted,
                color="C2", alpha=0.25, label="CQR-r interval")

ax.plot(x_sorted, 0.5 * (upper_sorted + lower_sorted),
        color="C2", lw=1, label="CQR-r center")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("CQR-r")
ax.legend()
ax.grid(True)

# =========================
# UACQR (bottom-right)
# =========================
ax = axes[1, 1]

lower_sorted = uacqr_pred_test["UACQR-S"]["lower"][idx]
upper_sorted = uacqr_pred_test["UACQR-S"]["upper"][idx]

ax.scatter(x_test_plot, Y_test.ravel(),
           s=30, color="k", alpha=0.8, zorder=3)

ax.fill_between(x_sorted, lower_sorted, upper_sorted,
                color="C3", alpha=0.25, label="UACQR-S interval")

ax.plot(x_sorted, 0.5 * (upper_sorted + lower_sorted),
        color="C3", lw=1, label="UACQR-S center")

ax.set_xlabel("x")
ax.set_title("UACQR-S")
ax.legend()
ax.grid(True)

# =========================
# Save
# =========================
plt.tight_layout()

output_path = os.path.join(RESULTS_DIR, "bart_CQR_CQRr_UACQR_teste.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()





