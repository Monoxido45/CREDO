# Repeating code from credal sets testing but for 
# BART and GP
# showcasing the flexibility of our framework and the ability to use different underlying models for the credal CPs, 
# as well as the ability to handle heteroscedasticity with GPs.

# ============================================
# 0. Imports & data generation
# ============================================
from jaxtyping import install_import_hook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  
from sklearn.model_selection import train_test_split
import torch

# importing our package
from credo.credal_cp import CredalCPRegressor
import numpy as np
import copy

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# For reproducibility
np.random.seed(125)
torch.manual_seed(125)

device = torch.device("cpu")  # change to "cuda" if desired

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
# 1. Training, calibration, and prediction
# ============================================
X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
Y_train = train["y"].values.astype(np.float32)

X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
Y_cal = cal["y"].values.astype(np.float32)

X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
Y_test = test["y"].values.astype(np.float32)

# ============================================
# 2. Train and calibrate credal CPs
# ============================================
kernel = (
    gpx.kernels.RBF() + 
    gpx.kernels.Matern52()
)
kernel_noise = (
    gpx.kernels.RBF()
)

credal_CP_gp = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "GP",
    alpha = 0.1, # for simplicity in this example
    gamma = 0.1,
)

credal_CP_gp_hetero = copy.deepcopy(credal_CP_gp)

# fitting GP-based credal CP
credal_CP_gp.fit(
    X_train,
    Y_train,
    scale = True,
    heteroscedastic = False,
    kernel = kernel,
)

# fitting also the heteroscedastic version
credal_CP_gp_hetero.fit(
    X_train,
    Y_train,
    scale = True,
    heteroscedastic = True,
    kernel = kernel,
    kernel_noise = kernel_noise,
    )

# BaRT-based credal CP
credal_CP_bart = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "BART",
    alpha = 0.1,
    gamma = 0.1,
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
bart_cutoff = credal_CP_bart.calibrate(X_cal, Y_cal,
                                               N_samples_MC=1000)

gp_cutoff = credal_CP_gp.calibrate(X_cal, Y_cal,
                                             N_samples_MC=1000)

gp_hetero_cutoff = credal_CP_gp_hetero.calibrate(X_cal, Y_cal,
                                             N_samples_MC=1000)

# ============================================
# 3. Plotting and predicting
# ============================================
# Prediction on the test set grid
X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
pred_bart = credal_CP_bart.predict(X_test_grid)
pred_gp = credal_CP_gp.predict(X_test_grid)
pred_gp_hetero = credal_CP_gp_hetero.predict(X_test_grid)

# Simple concise plotting assuming predict() returns (N,2) arrays [lower, upper]
pred_bart = np.asarray(pred_bart)
pred_gp  = np.asarray(pred_gp)
pred_gp_hetero = np.asarray(pred_gp_hetero)

lower_bart, upper_bart = pred_bart[:, 0], pred_bart[:, 1]
lower_gp,  upper_gp  = pred_gp[:, 0],  pred_gp[:, 1]
lower_gp_hetero, upper_gp_hetero = pred_gp_hetero[:, 0], pred_gp_hetero[:, 1]

center_bart = 0.5 * (lower_bart + upper_bart)
center_gp  = 0.5 * (lower_gp  + upper_gp)
center_gp_hetero = 0.5 * (lower_gp_hetero + upper_gp_hetero)

xx = X_test_grid.ravel()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# BART plot (left)
ax = axes[0]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_bart, upper_bart, color="C0", alpha=0.25, label="BART interval")
ax.plot(xx, center_bart, color="C0", lw=1, label="BART center")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("BART")
ax.legend()
ax.grid(True)

# GP plot (middle)
ax = axes[1]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_gp, upper_gp, color="C1", alpha=0.25, label="GP interval")
ax.plot(xx, center_gp, color="C1", lw=1, label="GP center")
ax.set_xlabel("x")
ax.set_title("GP")
ax.legend()
ax.grid(True)

# heteroscedastic GP plot (right)
ax = axes[2]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_gp_hetero, upper_gp_hetero, color="C2", alpha=0.25, label="Het GP interval")
ax.plot(xx, center_gp_hetero, color="C2", lw=1, label="Het GP center")
ax.set_xlabel("x")
ax.set_title("Heteroscedastic GP")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


############## just between GP's for better visibility ############## 
X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
pred_gp = credal_CP_gp.predict(X_test_grid)
pred_gp_hetero = credal_CP_gp_hetero.predict(X_test_grid)
lower_gp,  upper_gp  = pred_gp[:, 0],  pred_gp[:, 1]
lower_gp_hetero, upper_gp_hetero = pred_gp_hetero[:, 0], pred_gp_hetero[:, 1]

center_gp  = 0.5 * (lower_gp  + upper_gp)
center_gp_hetero = 0.5 * (lower_gp_hetero + upper_gp_hetero)
xx = X_test_grid.ravel()

fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

# GP plot (middle)
ax = axes[0]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_gp, upper_gp, color="C1", alpha=0.25, label="GP interval")
ax.plot(xx, center_gp, color="C1", lw=1, label="GP center")
ax.set_xlabel("x")
ax.set_title("GP")
ax.legend()
ax.grid(True)

# heteroscedastic GP plot (right)
ax = axes[1]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_gp_hetero, upper_gp_hetero, color="C2", alpha=0.25, label="Het GP interval")
ax.plot(xx, center_gp_hetero, color="C2", lw=1, label="Het GP center")
ax.set_xlabel("x")
ax.set_title("Heteroscedastic GP")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# 4. Marginal coverage computation
# ============================================
# Compute marginal coverage on the test set
pred_bart_test = np.asarray(credal_CP_bart.predict(X_test))
pred_gp_test  = np.asarray(credal_CP_gp.predict(X_test))

l_bart, u_bart = pred_bart_test[:, 0], pred_bart_test[:, 1]
l_gp,  u_gp  = pred_gp_test[:, 0],  pred_gp_test[:, 1]
y_true = Y_test.ravel()

inside_bart = (y_true >= l_bart) & (y_true <= u_bart)
inside_gp  = (y_true >= l_gp)  & (y_true <= u_gp)

coverage_bart = float(np.mean(inside_bart))
coverage_gp  = float(np.mean(inside_gp))
avg_len_bart = float(np.mean(u_bart - l_bart))
avg_len_gp  = float(np.mean(u_gp - l_gp))

print(f"BART marginal coverage: {coverage_bart:.3f}, avg interval length: {avg_len_bart:.3f}")
print(f"GP marginal coverage:  {coverage_gp:.3f}, avg interval length: {avg_len_gp:.3f}")


# ============================================
# Testing another example (with empty spaces in the middle)
# ============================================

def make_variable_data(n, std_dev=1/5):
    # proportion of points in the scarce region
    p_scarce = 0.02      # few points in [0, 0.4]
    n_scarce = int(n * p_scarce)
    n_dense = n - n_scarce

    # dense region: outside [0, 0.4]
    x_dense = np.random.uniform(low=-1, high=1, size=n_dense)
    x_dense = x_dense[(x_dense < 0) | (x_dense > 0.4)]
    while len(x_dense) < n_dense:
        new_samples = np.random.uniform(low=-1, high=1, size=n_dense)
        new_samples = new_samples[(new_samples < 0) | (new_samples > 0.4)]
        x_dense = np.concatenate([x_dense, new_samples])
    x_dense = x_dense[:n_dense]

    # scarce region: [0, 0.4]
    x_scarce = np.random.uniform(low=0, high=0.4, size=n_scarce)

    # join everything
    x = np.concatenate([x_dense, x_scarce])
    np.random.shuffle(x)

    # true mean
    mu = (x**3) + 2 * np.exp(-6 * (x - 0.3)**2)

    # --------------------------
    # variance is quite complex
    # --------------------------
    sigma = np.zeros_like(x)

    # left region [-1, -0.3]: low variance (easy)
    mask_low = (x <= -0.3)
    sigma[mask_low] = 0.1

    # intermediate region (-0.3, 0): moderate variance + oscillation
    mask_mid = (x > -0.3) & (x < 0)
    sigma[mask_mid] = 0.2 + 0.15 * np.abs(np.sin(10 * x[mask_mid]))

    # sparse region [0, 0.4]: high variance (hard, few points)
    mask_sparse = (x >= 0) & (x <= 0.4)
    sigma[mask_sparse] = 0.7 + 0.3 * np.abs(np.sin(12 * x[mask_sparse]))

    # right region (0.4, 1]: back to moderate variance
    mask_right = (x > 0.4)
    sigma[mask_right] = 0.3 + 0.1 * (x[mask_right] > 0.7)

    # optional global scaling
    sigma *= std_dev / (1/5)

    # generate y
    y = mu + np.random.normal(scale=sigma, size=len(x))

    return pd.DataFrame({'x': x, 'y': y})

np.random.seed(125)
torch.manual_seed(125)
n= 1500
data = make_variable_data(n)
train, rest = train_test_split(data, test_size=0.5, random_state=125)
cal, test = train_test_split(rest, test_size=0.5, random_state=125)


# ============================================
# 1. Training, calibration, and prediction
# ============================================
X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
Y_train = train["y"].values.astype(np.float32)

X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
Y_cal = cal["y"].values.astype(np.float32)

X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
Y_test = test["y"].values.astype(np.float32)

# ============================================
# 2. Train and calibrate credal CPs
# ============================================
kernel = (
    gpx.kernels.RBF() + 
    gpx.kernels.Matern52()
)
kernel_noise = (
    gpx.kernels.RBF()
)

credal_CP_gp = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "GP",
    alpha = 0.1, # for simplicity in this example
    gamma = 0.1,
)

credal_CP_gp_hetero = copy.deepcopy(credal_CP_gp)

# fitting GP-based credal CP
credal_CP_gp.fit(
    X_train,
    Y_train,
    scale = True,
    heteroscedastic = False,
    kernel = kernel,
)

# fitting also the heteroscedastic version
credal_CP_gp_hetero.fit(
    X_train,
    Y_train,
    scale = True,
    heteroscedastic = True,
    kernel = kernel,
    kernel_noise = kernel_noise,
    activation_sigma="lognormal",
    )


# calibration of the credal CPs
gp_cutoff = credal_CP_gp.calibrate(X_cal, Y_cal,
                                             N_samples_MC=1000)

gp_hetero_cutoff = credal_CP_gp_hetero.calibrate(X_cal, Y_cal,
                                             N_samples_MC=1000)

# ============================================
# 3. Plotting and predicting
# ============================================
# Prediction on the test set grid
X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
pred_gp = credal_CP_gp.predict(X_test_grid)
pred_gp_hetero = credal_CP_gp_hetero.predict(X_test_grid)

# Simple concise plotting assuming predict() returns (N,2) arrays [lower, upper]
pred_gp  = np.asarray(pred_gp)
pred_gp_hetero = np.asarray(pred_gp_hetero)

lower_gp,  upper_gp  = pred_gp[:, 0],  pred_gp[:, 1]
lower_gp_hetero, upper_gp_hetero = pred_gp_hetero[:, 0], pred_gp_hetero[:, 1]

center_gp  = 0.5 * (lower_gp  + upper_gp)
center_gp_hetero = 0.5 * (lower_gp_hetero + upper_gp_hetero)

xx = X_test_grid.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# BART plot (left)
ax = axes[0]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_gp, upper_gp, color="C0", alpha=0.25, label="GP interval")
ax.plot(xx, center_gp, color="C0", lw=1, label="GP center")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("GP")
ax.legend()
ax.grid(True)

# GP plot (right)
ax = axes[1]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_gp_hetero, upper_gp_hetero, color="C1", alpha=0.25, label="Hetero GP interval")
ax.plot(xx, center_gp_hetero, color="C1", lw=1, label="Hetero GP center")
ax.set_xlabel("x")
ax.set_title("Heteroscedastic GP")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# 4. Marginal coverage computation
# ============================================
# Compute marginal coverage on the test set
pred_hetero_gp = np.asarray(credal_CP_gp_hetero.predict(X_test))
pred_gp_test  = np.asarray(credal_CP_gp.predict(X_test))

l_hetero_gp, u_hetero_gp = pred_hetero_gp[:, 0], pred_hetero_gp[:, 1]
l_gp,  u_gp  = pred_gp_test[:, 0],  pred_gp_test[:, 1]
y_true = Y_test.ravel()

inside_hetero_gp = (y_true >= l_hetero_gp) & (y_true <= u_hetero_gp)
inside_gp  = (y_true >= l_gp)  & (y_true <= u_gp)

coverage_hetero_gp = float(np.mean(inside_hetero_gp))
coverage_gp  = float(np.mean(inside_gp))
avg_len_hetero_gp = float(np.mean(u_hetero_gp - l_hetero_gp))
avg_len_gp  = float(np.mean(u_gp - l_gp))

print(f"Hetero GP marginal coverage: {coverage_hetero_gp:.3f}, avg interval length: {avg_len_hetero_gp:.3f}")
print(f"GP marginal coverage:  {coverage_gp:.3f}, avg interval length: {avg_len_gp:.3f}")


