# ============================================
# Imports & data generation
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
from credal_cp.utils import CQR
import numpy as np
from matplotlib.patches import Patch

# For reproducibility
np.random.seed(125)
torch.manual_seed(125)

device = torch.device("cpu")  # change to "cuda" if desired

# first type of data generation
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

# second type
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

# Generate data
data = make_gap_epistemic_few_middle(n=2500, noise_std=0.1)

# Train / cal / test split
train, rest = train_test_split(data, test_size=0.5, random_state=125)
cal, test   = train_test_split(rest,  test_size=0.5, random_state=125)

# Visualize data
plt.figure(figsize=(6, 4))
plt.scatter(train["x"], train["y"], s=10, alpha=0.5, label="Train")
plt.scatter(cal["x"],   cal["y"],   s=10, alpha=0.5, label="Cal")
plt.scatter(test["x"],  test["y"],  s=10, alpha=0.5, label="Test")
plt.legend()
plt.title("Data with gap in the middle")
plt.show()

# plotting aleatoric vs epistemic uncertainty for MC-dropout and Deep Ensembles
def plot_uncertainty_decomposition(
    credal_CP_ensemble,
    credal_CP_dropout,
    X_test_grid,
    X_test,
    Y_test,
    normalize=True,
):
    # Getting uncertainties from Credal CP (Deep Ensemble)
    y_pred_ens, aleatoric_ens, epistemic_ens = credal_CP_ensemble.predict(X_test_grid, disentangle=True)

    # Getting uncertainties from Credal CP (Dropout)
    y_pred_do, aleatoric_do, epistemic_do = credal_CP_dropout.predict(X_test_grid, disentangle=True)

    # normalizing the uncertainties for better visualization
    if normalize:
        total_ens = aleatoric_ens + epistemic_ens
        aleatoric_ens /= total_ens
        epistemic_ens /= total_ens

        total_do = aleatoric_do + epistemic_do
        aleatoric_do /= total_do
        epistemic_do /= total_do

    # Also get full prediction intervals (conformalized) for both methods to plot below
    pred_ens_intervals = np.asarray(y_pred_do)
    pred_do_intervals  = np.asarray(y_pred_ens)

    lower_ens, upper_ens = pred_ens_intervals[:, 0], pred_ens_intervals[:, 1]
    lower_do,  upper_do  = pred_do_intervals[:, 0],  pred_do_intervals[:, 1]

    center_ens = 0.5 * (lower_ens + upper_ens)
    center_do  = 0.5 * (lower_do  + upper_do)

    xx = X_test_grid.ravel()

    # Create a 2x2 grid: top row = uncertainty decomposition, bottom row = prediction intervals
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Top-left: Deep Ensemble uncertainty decomposition
    ax = axes[0, 0]
    ax.plot(xx, aleatoric_ens, label="Aleatoric", color="C2")
    ax.plot(xx, epistemic_ens, label="Epistemic", color="C1")
    ax.set_title("Normalized Uncertainty Decomposition (Deep Ensemble)")
    ax.set_ylabel("Uncertainty")
    ax.legend()
    ax.grid(True)

    # Top-right: Dropout uncertainty decomposition
    ax = axes[0, 1]
    ax.plot(xx, aleatoric_do, label="Aleatoric", color="C2")
    ax.plot(xx, epistemic_do, label="Epistemic", color="C1")
    ax.set_title("Normalized Uncertainty Decomposition (Dropout)")
    ax.legend()
    ax.grid(True)

    # Bottom-left: Deep Ensemble prediction intervals
    ax = axes[1, 0]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha=0.4, zorder=3)
    ax.fill_between(xx, lower_ens, upper_ens, color="C0", alpha=0.25, label="Prediction Interval")
    ax.plot(xx, center_ens, color="C0", lw=1, label="Interval Center")
    ax.set_title("Prediction Intervals (Deep Ensemble)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

    # Bottom-right: Dropout prediction intervals
    ax = axes[1, 1]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha=0.4, zorder=3)
    ax.fill_between(xx, lower_do, upper_do, color="C0", alpha=0.25, label="Prediction Interval")
    ax.plot(xx, center_do, color="C0", lw=1, label="Interval Center")
    ax.set_title("Prediction Intervals (Dropout)")
    ax.set_xlabel("x")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Function to fit all methods to the data
def fit_all_methods(train, cal, test):
    X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
    Y_train = train["y"].values.astype(np.float32)

    X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
    Y_cal = cal["y"].values.astype(np.float32)

    X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
    Y_test = test["y"].values.astype(np.float32)

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
        num_components = 1,
        hidden_layers=[64],
        batch_size = 16,
        lr = 0.001,
        weight_decay=1e-6,
        scale=True,
        patience = 15,
    )

    credal_CP_dropout.fit(
        X_train,
        Y_train,
        nn_type = "MC_Dropout",
        num_components = 1,
        hidden_layers=[64],
        dropout_rate = 0.5,
        epochs = 1000,
        batch_size = 16,
        weight_decay=1e-6,
        lr = 0.001,
        scale=True,
        patience = 30,
    )

    # Using their base models to fit standard CQR and CQR-r
    dropout_base_model = credal_CP_dropout.base_model

    # fitting CQR and CQR-r
    cqr_dropout = CQR(base_model=dropout_base_model, alpha=0.1, variation="standard", is_fitted=True)
    cqr_r_dropout = CQR(base_model=dropout_base_model, alpha=0.1, variation="cqr-r", is_fitted=True)

    cqr_dropout.fit(X_train, Y_train)
    cqr_r_dropout.fit(X_train, Y_train)

    # calibration of CQR and CQR-r
    cqr_dropout.calibrate(X_cal, Y_cal)
    cqr_r_dropout.calibrate(X_cal, Y_cal)


    # calibration of the credal CPs
    credal_CP_ensemble.calibrate(X_cal, Y_cal, beta = 0.1,
                                                N_samples_MC=1000)

    credal_CP_dropout.calibrate(X_cal, Y_cal, beta = 0.1, 
                                                N_samples_MC=1000)

    return credal_CP_ensemble, credal_CP_dropout, \
                cqr_dropout, cqr_r_dropout, \
                    X_test, Y_test

def fit_bart(train, cal, test):
    X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
    Y_train = train["y"].values.astype(np.float32)

    X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
    Y_cal = cal["y"].values.astype(np.float32)

    X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
    Y_test = test["y"].values.astype(np.float32)

    credal_CP_bart = CredalCPRegressor(
        nc_type = 'Quantile',
        base_model = "BART",
        alpha = 0.1,
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

    bart_cutoff = credal_CP_bart.calibrate(
        X_cal, 
        Y_cal, 
        beta = 0.1,
        N_samples_MC=1000
        )
    
    return credal_CP_bart, X_test, Y_test

def plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test,
    Y_test,
    normalize=True,
):
    y_pred, aleatoric, epistemic = credal_CP_bart.predict(X_test_grid, disentangle=True)

    # normalizing the uncertainties for better visualization
    if normalize:
        total = aleatoric + epistemic
        new_aleatoric = aleatoric/total
        new_epistemic = epistemic/total
    else:
        new_aleatoric = aleatoric
        new_epistemic = epistemic
    
     # Also get full prediction intervals (conformalized) for both methods to plot below
    pred_intervals = np.asarray(y_pred)

    lower, upper = pred_intervals[:, 0], pred_intervals[:, 1]
    center_ens = 0.5 * (lower + upper)

    xx = X_test_grid.ravel()

    # Create a 2x2 grid: top row = uncertainty decomposition, bottom row = prediction intervals
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

    # Top-left: BART uncertainty decomposition
    ax = axes[0]
    ax.plot(xx, new_aleatoric, label="Aleatoric", color="C2")
    ax.plot(xx, new_epistemic, label="Epistemic", color="C1")
    ax.set_title("Normalized Uncertainty Decomposition (BART)")
    ax.set_ylabel("Uncertainty percentage")
    ax.legend()
    ax.grid(True)

    # Bottom-right: BART prediction intervals
    ax = axes[1]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha=0.4, zorder=3)
    ax.fill_between(xx, lower, upper, color="C0", alpha=0.25, label="Prediction Interval")
    ax.plot(xx, center_ens, color="C0", lw=1, label="Interval Center")
    ax.set_title("Prediction Intervals (BART)")
    ax.set_xlabel("x")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# function to plot all results along with empirical coverage & avg length
def plot_results(
        credal_CP_ensemble, 
        credal_CP_dropout,
        cqr_dropout, 
        cqr_r_dropout,
        X_test, 
        Y_test,
        ):

    # Prediction on the test set grid
    X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)
    pred_ens = np.asarray(credal_CP_ensemble.predict(X_test_grid))
    pred_do  = np.asarray(credal_CP_dropout.predict(X_test_grid))
    pred_ens_vanilla = np.asarray(credal_CP_ensemble.predict(X_test_grid, conformalize=False))
    pred_do_vanilla  = np.asarray(credal_CP_dropout.predict(X_test_grid, conformalize=False))
    pred_cqr_do    = np.asarray(cqr_dropout.predict(X_test_grid))
    pred_cqr_r_do  = np.asarray(cqr_r_dropout.predict(X_test_grid))

    lower_ens, upper_ens = pred_ens[:, 0], pred_ens[:, 1]
    lower_do,  upper_do  = pred_do[:, 0],  pred_do[:, 1]
    lower_ens_vanilla, upper_ens_vanilla = pred_ens_vanilla[:, 0], pred_ens_vanilla[:, 1]
    lower_do_vanilla,  upper_do_vanilla  = pred_do_vanilla[:, 0],  pred_do_vanilla[:, 1]
    lower_cqr_do, upper_cqr_do = pred_cqr_do[:, 0], pred_cqr_do[:, 1]
    lower_cqr_r_do, upper_cqr_r_do = pred_cqr_r_do[:, 0], pred_cqr_r_do[:, 1]

    center_ens = 0.5 * (lower_ens + upper_ens)
    center_do  = 0.5 * (lower_do  + upper_do)
    center_ens_vanilla = 0.5 * (lower_ens_vanilla + upper_ens_vanilla)
    center_do_vanilla  = 0.5 * (lower_do_vanilla  + upper_do_vanilla)
    center_cqr_do = 0.5 * (lower_cqr_do + upper_cqr_do)
    center_cqr_r_do = 0.5 * (lower_cqr_r_do + upper_cqr_r_do)

    xx = X_test_grid.ravel()

    # Helper to compute empirical coverage & avg length on the actual test set
    def stats_on_test(predict_fn, X_test_pts, Y_test_pts, vanilla_credal = False):
        if vanilla_credal:
            pred = np.asarray(predict_fn(X_test_pts, conformalize=False))
        else:
            pred = np.asarray(predict_fn(X_test_pts))
        l, u = pred[:, 0], pred[:, 1]
        inside = (Y_test_pts.ravel() >= l) & (Y_test_pts.ravel() <= u)
        cov = float(np.mean(inside))
        avg_len = float(np.mean(u - l))
        return cov, avg_len

    # Compute stats for each candidate (using the fitted predict methods)
    stats = {}
    stats["Credal CP (Deep Ensemble)"] = stats_on_test(credal_CP_ensemble.predict, X_test, Y_test)
    stats["Credal CP (Dropout)"] = stats_on_test(credal_CP_dropout.predict, X_test, Y_test)
    stats["Credal Vanilla (Deep Ensemble)"] = stats_on_test(credal_CP_ensemble.predict, X_test, Y_test, vanilla_credal=True)
    stats["Credal Vanilla (Dropout)"] = stats_on_test(credal_CP_dropout.predict, X_test, Y_test, vanilla_credal=True)
    stats["CQR"]    = stats_on_test(cqr_dropout.predict, X_test, Y_test)
    stats["CQR-r"]  = stats_on_test(cqr_r_dropout.predict, X_test, Y_test)

    # Determine common y-limits using predictions + actual Y_test
    all_lowers = np.hstack([lower_ens,
                            lower_do, lower_cqr_do, lower_cqr_r_do, Y_test.ravel()])
    all_uppers = np.hstack([upper_ens,
                            upper_do, upper_cqr_do, upper_cqr_r_do, Y_test.ravel()])
    ymin = float(np.min(all_lowers)) - 0.1
    ymax = float(np.max(all_uppers)) + 0.1

    # Plot grid: 2 rows (ensemble, dropout) x 3 cols (credal, cqr, cqr-r)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

    # Row 0: Ensemble-based methods
    ax = axes[0, 0]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha = 0.4, zorder=3)
    ax.fill_between(xx, lower_ens, upper_ens, color="C0", alpha=0.25)
    ax.plot(xx, center_ens, color="C0", lw=1)
    cov, avg_len = stats["Credal CP (Deep Ensemble)"]
    ax.set_title(f"Credal CP (Deep Ensemble)\ncoverage={cov:.3f}, len={avg_len:.3f}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    ax = axes[0, 1]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha = 0.4, zorder=3)
    ax.fill_between(xx, lower_ens_vanilla, upper_ens_vanilla, color="C3", alpha=0.25)
    ax.plot(xx, center_ens_vanilla, color="C3", lw=1)
    cov, avg_len = stats["Credal Vanilla (Deep Ensemble)"]
    ax.set_title(f"Credal Vanilla (Deep Ensemble)\ncoverage={cov:.3f}, len={avg_len:.3f}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    ax = axes[0, 2]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha = 0.4, zorder=3)
    ax.fill_between(xx, lower_do_vanilla, upper_do_vanilla, color="C3", alpha=0.25)
    ax.plot(xx, center_do_vanilla, color="C3", lw=1)
    cov, avg_len = stats["Credal Vanilla (Dropout)"]
    ax.set_title(f"Credal Vanilla (Dropout)\ncoverage={cov:.3f}, len={avg_len:.3f}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    # Row 1: Dropout-based methods
    ax = axes[1, 0]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha = 0.4, zorder=3)
    ax.fill_between(xx, lower_do, upper_do, color="C0", alpha=0.25)
    ax.plot(xx, center_do, color="C0", lw=1)
    cov, avg_len = stats["Credal CP (Dropout)"]
    ax.set_title(f"Credal CP (Dropout)\ncoverage={cov:.3f}, len={avg_len:.3f}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.grid(True)

    ax = axes[1, 1]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha=0.4, zorder=3)
    ax.fill_between(xx, lower_cqr_do, upper_cqr_do, color="C1", alpha=0.25)
    ax.plot(xx, center_cqr_do, color="C1", lw=1)
    cov, avg_len = stats["CQR"]
    ax.set_title(f"CQR\ncoverage={cov:.3f}, len={avg_len:.3f}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.grid(True)

    ax = axes[1, 2]
    ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha = 0.4, zorder=3)
    ax.fill_between(xx, lower_cqr_r_do, upper_cqr_r_do, color="C2", alpha=0.25)
    ax.plot(xx, center_cqr_r_do, color="C2", lw=1)
    cov, avg_len = stats["CQR-r"]
    ax.set_title(f"CQR-r\ncoverage={cov:.3f}, len={avg_len:.3f}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.grid(True)

    # Shared legend (create one representative patch per method)
    legend_patches = [
        Patch(facecolor="C0", alpha=0.25, label="Credal CP"),
        Patch(facecolor="C3", alpha=0.25, label="Credal Vanilla"),
        Patch(facecolor="C1", alpha=0.25, label="CQR"),
        Patch(facecolor="C2", alpha=0.25, label="CQR-r"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

####### Fitting neural network based methods #######
credal_CP_ensemble, credal_CP_dropout, \
        cqr_dropout, cqr_r_dropout, \
            X_test, Y_test = fit_all_methods(train, cal, test)

# plotting uncertainty decomposition
X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)
plot_uncertainty_decomposition(
    credal_CP_ensemble,
    credal_CP_dropout,
    X_test_grid,
    X_test,
    Y_test,
)

# looking at SD
sd_dropout = credal_CP_dropout.sigma_list
# stack sigma tensors as rows (assumes all entries have same number of elements)
sigma_rows = torch.stack([s.flatten().detach().cpu() for s in sd_dropout], dim=0)

# mean across columns (per-row mean)
sigma_row_means = sigma_rows.mean(dim=1)

# optional: convert to numpy for printing/plotting
sigma_row_means_np = sigma_row_means.numpy()

# Plot sigma (MC samples) as a function of X_test_grid
xx = X_test_grid.ravel()

plt.figure(figsize=(8, 4))

plt.plot(xx, sigma_row_means_np, color="C1", lw=2)
plt.xlabel("x")
plt.ylabel("sigma")
plt.title("Predicted sigma vs x")
plt.legend()
plt.grid(True)
plt.show()

# plotting all results
plot_results(
    credal_CP_ensemble, 
    credal_CP_dropout,
    cqr_dropout, 
    cqr_r_dropout,
    X_test, 
    Y_test,
)

####### Fitting bart-based method #######
credal_CP_bart, X_test_bart, Y_test_bart = fit_bart(train, cal, test)

X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)
# plotting uncertainty decomposition for BART
plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test_bart,
    Y_test_bart,
)


############ Testing the other example with variable noise ############
np.random.seed(42)
n= 2500
data = make_variable_data(n)
train, rest = train_test_split(data, test_size=0.5, random_state=42)
cal, test = train_test_split(rest, test_size=0.5, random_state=42)

credal_CP_ensemble, credal_CP_dropout, \
        cqr_dropout, cqr_r_dropout, \
            X_test, Y_test = fit_all_methods(train, cal, test)

X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)

plot_uncertainty_decomposition(
    credal_CP_ensemble,
    credal_CP_dropout,
    X_test_grid,
    X_test,
    Y_test,
    normalize=False,
)

plot_results(
    credal_CP_ensemble,
    credal_CP_dropout,
    cqr_dropout,
    cqr_r_dropout,
    X_test,
    Y_test,
)

####### Fitting bart-based method #######
credal_CP_bart, X_test_bart, Y_test_bart = fit_bart(train, cal, test)

X_test_grid = np.linspace(-1.0, 1.0, 500).astype(np.float32).reshape(-1, 1)
# plotting uncertainty decomposition for BART
plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test_bart,
    Y_test_bart,
    normalize=False,
)

