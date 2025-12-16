# ============================================
# 0. Imports & data generation
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  
from sklearn.model_selection import train_test_split
import torch

# importing our package
from credal_cp.credal_cp import CredalCPRegressor
import numpy as np

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
data = make_gap_epistemic_few_middle(n=1000, noise_std=0.1, p_middle=0.01)

# Train / cal / test split
train, rest = train_test_split(data, test_size=0.5, random_state=42)
cal, test   = train_test_split(rest,  test_size=0.5, random_state=42)

# Quick sanity plot of the data
plt.figure(figsize=(6, 4))
plt.scatter(train["x"], train["y"], s=10, alpha=0.5, label="Train")
plt.scatter(cal["x"],   cal["y"],   s=10, alpha=0.5, label="Cal")
plt.scatter(test["x"],  test["y"],  s=10, alpha=0.5, label="Test")
plt.legend()
plt.title("Data with gap in the middle")
plt.show()

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
credal_CP_ensemble = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "MDN",
    alpha = 0.1,
)

credal_CP_dropout = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "MDN",
    alpha = 0.1,
)

# starting fitting
credal_CP_ensemble.fit(
    X_train, 
    Y_train,
    nn_type = "Ensemble",
    n_models = 10,
    epochs = 300,
    num_components = 5,
    hidden_layers=[128, 64],
    batch_size = 48,
    lr = 0.001,
    weight_decay=1e-4,
    scale=True,
    patience = 15,
)

credal_CP_dropout.fit(
    X_train,
    Y_train,
    nn_type = "MC_Dropout",
    num_components = 5,
    hidden_layers=[128, 64],
    dropout_rate = 0.3,
    epochs = 500,
    batch_size = 48,
    weight_decay=1e-4,
    lr = 0.001,
    scale=True,
    patience = 35,
)

# calibration of the credal CPs
ensemble_cutoff = credal_CP_ensemble.calibrate(X_cal, Y_cal, beta = 0.1,
                                               N_samples_MC=1000)
dropout_cutoff = credal_CP_dropout.calibrate(X_cal, Y_cal, beta = 0.1, 
                                             N_samples_MC=1000)

# ============================================
# 4. Plot results
# ============================================
# Prediction on the test set grid
X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)
pred_ens = credal_CP_ensemble.predict(X_test_grid)
pred_do = credal_CP_dropout.predict(X_test_grid)


# Simple concise plotting assuming predict() returns (N,2) arrays [lower, upper]
pred_ens = np.asarray(pred_ens)
pred_do  = np.asarray(pred_do)

lower_ens, upper_ens = pred_ens[:, 0], pred_ens[:, 1]
lower_do,  upper_do  = pred_do[:, 0],  pred_do[:, 1]

center_ens = 0.5 * (lower_ens + upper_ens)
center_do  = 0.5 * (lower_do  + upper_do)

xx = X_test_grid.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Ensemble plot (left)
ax = axes[0]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_ens, upper_ens, color="C0", alpha=0.25, label="Ensemble interval")
ax.plot(xx, center_ens, color="C0", lw=1, label="Ensemble center")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Ensemble")
ax.legend()
ax.grid(True)

# Dropout plot (right)
ax = axes[1]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_do, upper_do, color="C1", alpha=0.25, label="Dropout interval")
ax.plot(xx, center_do, color="C1", lw=1, label="Dropout center")
ax.set_xlabel("x")
ax.set_title("MC Dropout")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# 5. Marginal coverage computation
# ============================================
# Compute marginal coverage on the test set
pred_ens_test = np.asarray(credal_CP_ensemble.predict(X_test))
pred_do_test  = np.asarray(credal_CP_dropout.predict(X_test))

l_ens, u_ens = pred_ens_test[:, 0], pred_ens_test[:, 1]
l_do,  u_do  = pred_do_test[:, 0],  pred_do_test[:, 1]

y_true = Y_test.ravel()

inside_ens = (y_true >= l_ens) & (y_true <= u_ens)
inside_do  = (y_true >= l_do)  & (y_true <= u_do)

coverage_ens = float(np.mean(inside_ens))
coverage_do  = float(np.mean(inside_do))

avg_len_ens = float(np.mean(u_ens - l_ens))
avg_len_do  = float(np.mean(u_do - l_do))

print(f"Ensemble marginal coverage: {coverage_ens:.3f}, avg interval length: {avg_len_ens:.3f}")
print(f"Dropout marginal coverage:  {coverage_do:.3f}, avg interval length: {avg_len_do:.3f}")

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

np.random.seed(42)
n= 1500
data = make_variable_data(n)
train, rest = train_test_split(data, test_size=0.3, random_state=42)
cal, test = train_test_split(rest, test_size=0.5, random_state=42)


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
# 3. Train and calibrate credal CPs
# ============================================
credal_CP_ensemble = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "MDN",
    alpha = 0.1,
)

credal_CP_dropout = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "MDN",
    alpha = 0.1,
)

# starting fitting
credal_CP_ensemble.fit(
    X_train, 
    Y_train,
    nn_type = "Ensemble",
    n_models = 10,
    epochs = 300,
    num_components = 5,
    hidden_layers=[128, 64],
    batch_size = 48,
    lr = 0.001,
    weight_decay=1e-4,
    scale=True,
    patience = 15,
)

credal_CP_dropout.fit(
    X_train,
    Y_train,
    nn_type = "MC_Dropout",
    num_components = 5,
    hidden_layers=[128, 64],
    dropout_rate = 0.3,
    epochs = 500,
    batch_size = 48,
    weight_decay=1e-4,
    lr = 0.001,
    scale=True,
    patience = 35,
)

# calibration of the credal CPs
ensemble_cutoff = credal_CP_ensemble.calibrate(X_cal, Y_cal, beta = 0.1,
                                               N_samples_MC=1000)
dropout_cutoff = credal_CP_dropout.calibrate(X_cal, Y_cal, beta = 0.1, 
                                             N_samples_MC=1000)

# ============================================
# 4. Plot results
# ============================================
# Prediction on the test set grid
X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)
pred_ens = credal_CP_ensemble.predict(X_test_grid)
pred_do = credal_CP_dropout.predict(X_test_grid)


# Simple concise plotting assuming predict() returns (N,2) arrays [lower, upper]
pred_ens = np.asarray(pred_ens)
pred_do  = np.asarray(pred_do)

lower_ens, upper_ens = pred_ens[:, 0], pred_ens[:, 1]
lower_do,  upper_do  = pred_do[:, 0],  pred_do[:, 1]

center_ens = 0.5 * (lower_ens + upper_ens)
center_do  = 0.5 * (lower_do  + upper_do)

xx = X_test_grid.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Ensemble plot (left)
ax = axes[0]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_ens, upper_ens, color="C0", alpha=0.25, label="Ensemble interval")
ax.plot(xx, center_ens, color="C0", lw=1, label="Ensemble center")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Ensemble")
ax.legend()
ax.grid(True)

# Dropout plot (right)
ax = axes[1]
ax.scatter(X_test.ravel(), Y_test.ravel(), s=30, color="k", alpha=0.8, label="Test data", zorder=3)
ax.fill_between(xx, lower_do, upper_do, color="C1", alpha=0.25, label="Dropout interval")
ax.plot(xx, center_do, color="C1", lw=1, label="Dropout center")
ax.set_xlabel("x")
ax.set_title("MC Dropout")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


# ============================================
# 5. Marginal coverage computation
# ============================================
# Compute marginal coverage on the test set
pred_ens_test = np.asarray(credal_CP_ensemble.predict(X_test))
pred_do_test  = np.asarray(credal_CP_dropout.predict(X_test))

l_ens, u_ens = pred_ens_test[:, 0], pred_ens_test[:, 1]
l_do,  u_do  = pred_do_test[:, 0],  pred_do_test[:, 1]

y_true = Y_test.ravel()

inside_ens = (y_true >= l_ens) & (y_true <= u_ens)
inside_do  = (y_true >= l_do)  & (y_true <= u_do)

coverage_ens = float(np.mean(inside_ens))
coverage_do  = float(np.mean(inside_do))

avg_len_ens = float(np.mean(u_ens - l_ens))
avg_len_do  = float(np.mean(u_do - l_do))

print(f"Ensemble marginal coverage: {coverage_ens:.3f}, avg interval length: {avg_len_ens:.3f}")
print(f"Dropout marginal coverage:  {coverage_do:.3f}, avg interval length: {avg_len_do:.3f}")
