# Imports
import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('Qt5Agg')  
from sklearn.model_selection import train_test_split
import torch

# importing our package
from credal_cp.credal_cp import CredalCPRegressor
from credal_cp.epistemic_models import MDN_model, DE_MDN_model
import numpy as np

import pymc as pm
import pymc_bart as pmb
from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule
import arviz as az

from scipy.stats import norm

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)
rng = np.random.default_rng(42)
alpha = 0.1
beta = 0.1

device = torch.device("cpu")  # change to "cuda" if desired

# Epistemic types of data generation
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

def make_epistemic_mixture_gaps(rng, n):
    """
    Example where:
      - several gaps in the input space
      - noise is small and constant (mostly aleatoric)
      - mean is non-linear
    """
    n = 140

    # Mixture to create dense regions, sparse regions, and a few tail points (extrapolation pressure)
    mix = rng.choice([0, 1, 2, 3], size=n, p=[0.55, 0.30, 0.10, 0.05])

    x = np.empty(n)
    x[mix == 0] = rng.normal(loc=-1.8, scale=0.35, size=(mix == 0).sum())   # dense left cluster
    x[mix == 1] = rng.normal(loc=1.6,  scale=0.45, size=(mix == 1).sum())   # dense right cluster
    x[mix == 2] = rng.uniform(-0.4, 0.4, size=(mix == 2).sum())            # sparse middle band
    x[mix == 3] = rng.choice([-3.8, 3.8], size=(mix == 3).sum()) + rng.normal(0, 0.15, size=(mix == 3).sum())

    x = np.clip(x, -4.2, 4.2)
    x = np.sort(x)
    X = x.reshape(-1, 1)


    def f(x_):
        """Smooth latent function."""
        return 0.8*np.sin(1.1*x_) + 0.25*x_ + 0.45*np.cos(0.6*x_)


    def sigma_base(x_):
        """Heteroscedastic baseline scale."""
        return (
            0.12
            + 0.10*np.exp(-0.5*((x_+1.7)/0.5)**2)
            + 0.25/(1+np.exp(-(x_-1.2)))
            + 0.06*(x_/3.0)**2
        )


    def sample_noise(x_):
        """
        Adds conditional heterogeneity beyond Gaussian:
        A 'shock' component (shifted, high-variance) occurs with probability increasing in right tail and far tails.
        """
        x_ = np.asarray(x_)

        p_shock = 0.05 + 0.35/(1+np.exp(-(x_-1.3))) + 0.15*(np.abs(x_) > 3.2)
        p_shock = np.clip(p_shock, 0.05, 0.65)

        shock = rng.binomial(1, p_shock, size=x_.shape[0])

        e_main  = rng.normal(0, 1, size=x_.shape[0])
        e_shock = rng.normal(loc=1.8, scale=2.2, size=x_.shape[0])  # shifted & heavier

        return (1-shock)*e_main + shock*e_shock


    y = f(x) + sigma_base(x) * sample_noise(x)
    df = pd.DataFrame({"x": x, "y": y})
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)

# generating heteroscedastic but more regular data for testing
def make_heteroscedastic_sin(n=3000, noise_base=0.05):
    """
    X ~ Uniform(-1, 1)
    Y ~ Normal(mean = sin-like function, sd = heteroscedastic function of x)
    """
    x = np.random.uniform(-1.0, 1.0, size=n)
    mu = np.sin(2.5 * x)            # sin-like mean
    sigma = noise_base + 0.4 * (x ** 2)  # small near 0, larger towards edges
    y = mu + np.random.normal(scale=sigma, size=n)

    df = pd.DataFrame({"x": x, "y": y})
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)

def make_homoscedastic_sin(n=3000, noise_std=0.1):
    """
    X ~ Uniform(-1, 1)
    Y ~ Normal(mean = sin-like function, sd = constant noise_std)
    """
    x = np.random.uniform(-1.0, 1.0, size=n)
    mu = np.sin(2.5 * x)
    y = mu + np.random.normal(scale=noise_std, size=n)

    df = pd.DataFrame({"x": x, "y": y})
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)

# Generate data
epis_data = make_gap_epistemic_few_middle(n=5000, noise_std=0.1, p_middle=0.01)
epis_data_mixture = make_epistemic_mixture_gaps(rng, n=5000)
# other more regular datasets for testing
hetero_data = make_heteroscedastic_sin(n=5000, noise_base=0.05)
homosc_data = make_homoscedastic_sin(n=5000, noise_std=0.1)

############# Regular data MDN proof of concept #############
# starting with homoscedastic data
train, rest = train_test_split(homosc_data, test_size=0.5, random_state=42)
cal, test   = train_test_split(rest,  test_size=0.5, random_state=42)
# grid with strapolations
X_grid = np.linspace(-1.2, 1.2, 500).reshape(-1, 1)

plt.figure(figsize=(6, 4))
plt.scatter(train["x"], train["y"], s=10, alpha=0.5, label="Train")
plt.scatter(cal["x"],   cal["y"],   s=10, alpha=0.5, label="Cal")
plt.scatter(test["x"],  test["y"],  s=10, alpha=0.5, label="Test")
plt.legend()
plt.title("Homoscedastic data")
plt.show()

# Training simple MDN for latter dropout-based interval construction
X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
Y_train = train["y"].values.astype(np.float32)
X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
Y_cal = cal["y"].values.astype(np.float32)
X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
Y_test = test["y"].values.astype(np.float32)

def train_MDN_and_obtain_int(
    X_train,
    Y_train,
    X_cal,
    Y_cal,
    X_grid,
    alpha = 0.1,
    beta = 0.1,
    normalize_y = False,
    no_dropout = False,
):
    if no_dropout:
        mdn_model = MDN_model(
        input_shape = 1,
        num_components = 1,
        hidden_layers = [64],
        dropout_rate = 0.0,
        normalize_y = normalize_y,
        )

        mdn_model.fit(
        X_train,
        Y_train,
        scale = True,
        patience = 50,
        epochs = 500,
        batch_size = 32,
        lr = 1e-3,
    )
        
        # first predicting the mean
        mdn_model.set_type_base_model("regression")
        mean_grid = mdn_model.predict(X_grid, N = 10000)

        # deriving the non conformalized quantiles
        mdn_model.set_type_base_model("quantile", alpha = alpha)
        # making grid predictions
        pred_grid = mdn_model.predict(X_grid, N = 10000)

        # standard CQR
        print("Calibrating CQR intervals \n")
        q_cqr = mdn_model.predict(X_cal, N = 10000)
        nc_scores_cqr = np.maximum(q_cqr[:, 0] - Y_cal, Y_cal - q_cqr[:, 1])
        n = len(nc_scores_cqr)
        cutoff_cqr = np.quantile(nc_scores_cqr, q=np.ceil((n + 1) * (1 - alpha)) / n)

        lo_cqr = pred_grid[:, 0] - cutoff_cqr
        up_cqr = pred_grid[:, 1] + cutoff_cqr

        pred_cqr_grid = np.column_stack((lo_cqr, up_cqr))
        return pred_grid, pred_cqr_grid, mean_grid, mdn_model
    else:
        mdn_model = MDN_model(
        input_shape = 1,
        num_components = 1,
        hidden_layers = [128],
        dropout_rate = 0.3,
        normalize_y = normalize_y,
    )

        mdn_model.fit(
        X_train,
        Y_train,
        scale = True,
        patience = 30,
        epochs = 500,
        batch_size = 32,
        lr = 0.001,
    )
        # deriving the non conformalized quantiles
        mdn_model.set_type_base_model("quantile", alpha = alpha)

        # making grid predictions
        pred_grid = mdn_model.predict(X_grid, N = 10000)

        # standard CQR
        print("Calibrating CQR intervals \n")
        q_cqr = mdn_model.predict(X_cal, N = 10000)
        nc_scores_cqr = np.maximum(q_cqr[:, 0] - Y_cal, Y_cal - q_cqr[:, 1])
        n = len(nc_scores_cqr)
        cutoff_cqr = np.quantile(nc_scores_cqr, q=np.ceil((n + 1) * (1 - alpha)) / n)

        lo_cqr = pred_grid[:, 0] - cutoff_cqr
        up_cqr = pred_grid[:, 1] + cutoff_cqr

        pred_cqr_grid = np.column_stack((lo_cqr, up_cqr))

        # Obtaining the credal conformalized region by hand
        print("Calibrating Credal CP intervals \n")
        pi_calib, mu_calib, sigma_calib = mdn_model.predict_mcdropout(
            X_cal, 
            num_samples = 300,
            )

        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        i = 0
        q_low_raw, q_upp_raw = [], []

        for x in tqdm(X_cal, desc="Calibrating Credal CP with MC Dropout MDN"):
            pi_chosen = pi_calib[:, i, :]
            mu_chosen = mu_calib[:, i, :]
            sigma_chosen = sigma_calib[:, i, :]

            q_grid = mdn_model.mixture_quantile(
            [lower_q, upper_q], 
            pi_chosen, 
            mu_chosen, 
            sigma_chosen,
            rng= rng,
            )

            q_grid = q_grid

            # obtaining lower and upper quantiles for the current x
            q_low_raw.append(np.quantile(q_grid[:, 0], beta/2))
            q_upp_raw.append(np.quantile(q_grid[:, 1], 1 - beta/2))
            i += 1

        q_low_raw = np.array(q_low_raw)
        q_upp_raw = np.array(q_upp_raw)
        # with lower and upper quantiles, we can compute the modified nonconformity scores
        nc_scores = np.maximum(q_low_raw - Y_cal, Y_cal - q_upp_raw)
        n = len(nc_scores)
        cutoff = np.quantile(nc_scores, q=np.ceil((n + 1) * (1 - alpha)) / n)

        # For grid predictions
        pi_grid, mu_grid, sigma_grid = mdn_model.predict_mcdropout(
            X_grid,
            num_samples = 300,
            )

        q_low_grid, q_upp_grid = [], []
        i = 0
        for x in tqdm(X_grid, desc="Calibrating Credal CP with MC Dropout MDN"):
            pi_chosen = pi_grid[:, i, :]
            mu_chosen = mu_grid[:, i, :]
            sigma_chosen = sigma_grid[:, i, :]
            q_grid = mdn_model.mixture_quantile(
            [lower_q, upper_q], 
            pi_chosen, 
            mu_chosen, 
            sigma_chosen,
            rng= rng,
            )

            q_grid = q_grid

            # obtaining lower and upper quantiles for the current x
            q_low_grid.append(np.quantile(q_grid[:, 0], beta/2))
            q_upp_grid.append(np.quantile(q_grid[:, 1], 1 - beta/2))
            i += 1

        q_low_grid = np.array(q_low_grid)
        q_upp_grid = np.array(q_upp_grid)

        lower_cp = q_low_grid - cutoff
        upper_cp = q_upp_grid + cutoff

        pred_grid_cp = np.column_stack((lower_cp, upper_cp))
        pred_grid_raw = np.column_stack((q_low_grid, q_upp_grid))

        return pred_grid, pred_cqr_grid, pred_grid_cp, pred_grid_raw, mdn_model

def plot_MDN_intervals(
        pred_grid,
        pred_cqr_grid,
        pred_grid_cp,
        pred_grid_raw,
        X_grid,
        ):
    x = X_grid.flatten()
    order = np.argsort(x)
    x_s = x[order]

    # Use pred_grid for the "raw" MDN predictive interval if it contains quantiles/intervals.
    pred_arr = np.asarray(pred_grid)
    std_lo = pred_arr[:, 0] if pred_arr.shape[1] >= 2 else None
    std_hi = pred_arr[:, 1] if pred_arr.shape[1] >= 2 else None
    raw_lo = pred_grid_raw[:, 0]
    raw_hi = pred_grid_raw[:, 1]
    cp_lo = pred_grid_cp[:, 0]
    cp_hi = pred_grid_cp[:, 1]
    cqr_lo = pred_cqr_grid[:, 0]
    cqr_hi = pred_cqr_grid[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    # Panel 1: vanilla MDN predictive quantiles (raw)
    ax = axes[0]
    ax.fill_between(x_s, raw_lo[order], raw_hi[order], color='C0', alpha=0.25, label='MDN predictive interval (raw)')
    y_pred_grid = pred_grid.reshape(-1) if np.ndim(pred_grid) > 1 and pred_grid.shape[1] == 1 else np.asarray(pred_grid)
    ax.scatter(test["x"], test["y"], s=18, color='k', alpha=0.7)
    ax.scatter(train["x"], train["y"], s=10, color='gray', alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Credal predictive quantiles (raw)")
    ax.legend(loc='upper left', fontsize='small')

    # Panel 2: conformalized Credal CP intervals
    ax = axes[1]
    ax.fill_between(x_s, cp_lo[order], cp_hi[order], color='C1', alpha=0.25, label='Credal CP calibrated interval')
    ax.scatter(test["x"], test["y"], s=18, color='k', alpha=0.7)
    ax.scatter(train["x"], train["y"], s=10, color='gray', alpha=0.3)
    ax.set_xlabel("x")
    ax.set_title(f"Credal CP intervals (alpha={alpha:.2f})")
    ax.legend(loc='upper left', fontsize='small')

    # Panel 3: standard quantiles
    ax = axes[2]
    ax.fill_between(x_s, std_lo[order], std_hi[order], color='C2', alpha=0.25, label='Standard intervals')
    ax.scatter(test["x"], test["y"], s=18, color='k', alpha=0.7)
    ax.scatter(train["x"], train["y"], s=10, color='gray', alpha=0.3)
    ax.set_xlabel("x")
    ax.set_title("Standard intervals")
    ax.legend(loc='upper left', fontsize='small')

    # Panel 4: Conformalized Quantile Regression (CQR) intervals
    ax = axes[3]
    ax.fill_between(x_s, cqr_lo[order], cqr_hi[order], color='C4', alpha=0.25, label='CQR calibrated interval')
    ax.scatter(test["x"], test["y"], s=18, color='k', alpha=0.7)
    ax.scatter(train["x"], train["y"], s=10, color='gray', alpha=0.3)
    ax.set_xlabel("x")
    ax.set_title("CQR intervals")
    ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

def plot_two_intervals(
    pred_interval_1,
    pred_interval_2,
    mu_grid,
    X_grid,
    train_df=None,
    test_df=None,
    titles=("Interval 1", "Interval 2"),
    colors=("C0", "C1"),
    figsize=(12, 4),
):
    x = np.asarray(X_grid).reshape(-1)
    order = np.argsort(x)
    x_s = x[order]

    pi1 = np.asarray(pred_interval_1)
    pi2 = np.asarray(pred_interval_2)

    lo1, hi1 = pi1[:, 0], pi1[:, 1]
    lo2, hi2 = pi2[:, 0], pi2[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes = axes.flatten()
    # prepare mu_grid
    mu_arr = np.asarray(mu_grid).reshape(-1)
    mu_s = mu_arr[order]

    # Left plot
    ax = axes[0]
    ax.fill_between(x_s, lo1[order], hi1[order], color=colors[0], alpha=0.25, label=titles[0])
    ax.plot(x_s, mu_s, color='red', linewidth=1.5, label='mu_grid')
    if test_df is not None:
        ax.scatter(test_df["x"], test_df["y"], s=18, color="k", alpha=0.7, label="Test")
    if train_df is not None:
        ax.scatter(train_df["x"], train_df["y"], s=8, color="gray", alpha=0.3, label="Train")
    ax.set_xlabel("x")
    ax.set_title(titles[0])
    ax.legend(loc="upper left", fontsize="small")

    # Right plot
    ax = axes[1]
    ax.fill_between(x_s, lo2[order], hi2[order], color=colors[1], alpha=0.25, label=titles[1])
    ax.plot(x_s, mu_s, color='red', linewidth=1.5, label='mu_grid')
    if test_df is not None:
        ax.scatter(test_df["x"], test_df["y"], s=18, color="k", alpha=0.7, label="Test")
    if train_df is not None:
        ax.scatter(train_df["x"], train_df["y"], s=8, color="gray", alpha=0.3, label="Train")
    ax.set_xlabel("x")
    ax.set_title(titles[1])
    ax.legend(loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.show()
    return fig, axes

pred_grid, pred_cqr_grid, pred_grid_cp, pred_grid_raw, mdn_model_simple = train_MDN_and_obtain_int(
    X_train,
    Y_train,
    X_cal,
    Y_cal,
    X_grid,
    alpha = alpha,
    beta = beta,
    normalize_y = True,
)

pred_grid, pred_cqr_grid, mu_grid, mdn_model_simple = train_MDN_and_obtain_int(
    X_train,
    Y_train,
    X_cal,
    Y_cal,
    X_grid,
    alpha = alpha,
    beta = beta,
    normalize_y = True,
    no_dropout = True,
)

plot_two_intervals(
    pred_grid,
    pred_cqr_grid,
    mu_grid,
    X_grid,
    train_df = train,
    test_df = test,
    titles = ("Vanilla MDN predictive quantiles", "CQR intervals"),
    colors = ("C0", "C4"),
)

plot_MDN_intervals(
    pred_grid,
    pred_cqr_grid,
    pred_grid_cp,
    pred_grid_raw,
    X_grid,
)



############# Epistemic MDN proof of concept #############
# Train / cal / test split
train, rest = train_test_split(epis_data, test_size=0.5, random_state=42)
cal, test   = train_test_split(rest,  test_size=0.5, random_state=42)

# Quick sanity plot of the data
plt.figure(figsize=(6, 4))
plt.scatter(train["x"], train["y"], s=10, alpha=0.5, label="Train")
plt.scatter(cal["x"],   cal["y"],   s=10, alpha=0.5, label="Cal")
plt.scatter(test["x"],  test["y"],  s=10, alpha=0.5, label="Test")
plt.legend()
plt.title("Data with gap in the middle")
plt.show()

# Training, calibration, and prediction
X_train = train["x"].values.astype(np.float32).reshape(-1, 1)
Y_train = train["y"].values.astype(np.float32)

X_cal = cal["x"].values.astype(np.float32).reshape(-1, 1)
Y_cal = cal["y"].values.astype(np.float32)

X_test = test["x"].values.astype(np.float32).reshape(-1, 1)
Y_test = test["y"].values.astype(np.float32)

credal_CP_dropout = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "MDN",
    alpha = 0.1,
)

credal_CP_dropout.fit(
    X_train,
    Y_train,
    nn_type = "MC_Dropout",
    num_components = 1,
    hidden_layers=[64, 64],
    epochs = 500,
    batch_size = 32,
    lr = 0.001,
    scale=True,
)

alpha = 0.1
beta = 0.1
rng = np.random.default_rng(seed=42)

# checking the sampling code from credal_CP_dropout and its outputs
pi_calib, mu_calib, sigma_calib = credal_CP_dropout.base_model.predict_mcdropout(
                X_cal, 
                num_samples = 1000,
                )

lower_q = alpha / 2
upper_q = 1 - alpha / 2
i = 0
q_low_raw, q_upp_raw = [], []

for x in X_cal:
    pi_chosen = pi_calib[:, i, :]
    mu_chosen = mu_calib[:, i, :]
    sigma_chosen = sigma_calib[:, i, :]

    q_grid = credal_CP_dropout.base_model.mixture_quantile(
    [lower_q, upper_q], 
    pi_chosen, 
    mu_chosen, 
    sigma_chosen,
    rng= rng,
    )

    # obtaining lower and upper quantiles for the current x
    q_low_raw.append(np.quantile(q_grid[:, 0], beta/2))
    q_upp_raw.append(np.quantile(q_grid[:, 1], 1 - beta/2))
    i += 1

q_low_array = np.array(q_low_raw)
q_upp_array = np.array(q_upp_raw)
# with lower and upper quantiles, we can compute the modified nonconformity scores
nc_scores = np.maximum(q_low_array - Y_cal, 
                        Y_cal - q_upp_array)
n = len(nc_scores)
cutoff = np.quantile(
    nc_scores,
    q=np.ceil((n + 1) * (1 - alpha)) / n
    )


# Testing prediction on test set
pi_test, mu_test, sigma_test = credal_CP_dropout.base_model.predict_mcdropout(
                X_test, 
                num_samples = 1000,
                )

lower_q = alpha / 2
upper_q = 1 - alpha / 2
i = 0
q_low_raw, q_upp_raw = [], []

for x in X_test:
    pi_chosen = pi_test[:, i, :]
    mu_chosen = mu_test[:, i, :]
    sigma_chosen = sigma_test[:, i, :]

    q_grid = credal_CP_dropout.base_model.mixture_quantile(
    [lower_q, upper_q], 
    pi_chosen, 
    mu_chosen, 
    sigma_chosen,
    rng= rng,
    )

    # obtaining lower and upper quantiles for the current x
    q_low_raw.append(np.quantile(q_grid[:, 0], beta/2))
    q_upp_raw.append(np.quantile(q_grid[:, 1], 1 - beta/2))
    i += 1

q_low_array = np.array(q_low_raw)
q_upp_array = np.array(q_upp_raw)

lower_cp = q_low_array - cutoff
upper_cp = q_upp_array + cutoff
y_pred = np.column_stack((lower_cp, upper_cp))


# Evaluate marginal coverage on test set
inside = (Y_test >= lower_cp) & (Y_test <= upper_cp)
coverage = inside.mean()
n_test = len(Y_test)
print(f"Test set marginal coverage: {coverage:.3f} ({inside.sum()}/{n_test})")
print(f"Target nominal coverage: {1 - alpha:.3f}")
print(f"Average interval width: {np.mean(upper_cp - lower_cp):.4f}")
# Uncomment to see indices of miscoverage
# print("Indices of miscoverage:", np.where(~inside)[0])

# testing CQR to evaluate if this is because of the model
credal_CP_dropout.base_model.alpha = 0.1
credal_CP_dropout.base_model.base_model_type = "quantile"

# calibration
q_calib = credal_CP_dropout.base_model.predict(X_cal)

# score and cutoff
nc_scores_cqr = np.maximum(q_calib[:, 0] - Y_cal,
                        Y_cal - q_calib[:, 1])
n = len(nc_scores_cqr)
cutoff = np.quantile(
    nc_scores_cqr,
    q=np.ceil((n + 1) * (1 - alpha)) / n
    )

# prediction
q_test = credal_CP_dropout.base_model.predict(X_test)
lower_cqr = q_test[:, 0] - cutoff
upper_cqr = q_test[:, 1] + cutoff

# Evaluate marginal coverage on test set
inside_cqr = (Y_test >= lower_cqr) & (Y_test <= upper_cqr)
coverage_cqr = inside_cqr.mean()


alphas = [0.05, 0.95]
pi_chosen = pi_calib[:, -1, :]  # shape (n_samples, n_components)
mu_chosen = mu_calib[:, -1, :]  # shape (n_samples,
sigma_chosen = sigma_calib[:, -1, :]  # shape (n_samples, n_components)

rng = np.random.default_rng(seed=42)
pi_chosen = np.asarray(pi_chosen)
mu_chosen = np.asarray(mu_chosen)
sigma_chosen = np.asarray(sigma_chosen)

n_sample, _ = pi_chosen.shape
n_alphas = len(alphas)

# N samples from each mixture
samples = credal_CP_dropout.base_model.sample_from_mixture(
    pi_chosen, 
    mu_chosen, 
    sigma_chosen, 
    N=1000)

# Quantile computation
quantile_matrix = np.zeros((n_sample, n_alphas))
for j, alpha in enumerate(alphas):
    quantile_matrix[:, j] = np.quantile(samples, alpha, axis=1)

np.quantile(quantile_matrix[:, 0], 0.05)
np.quantile(quantile_matrix[:, 1], 0.95)

# Debugging also the BART and GP
credal_CP_bart = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "BART",
    alpha = 0.1,
)

credal_CP_gp = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "GP_Approx",
    alpha = 0.1,
)

# starting fitting
credal_CP_bart.fit(
    X_train, 
    Y_train,
    progressbar = True,
    n_cores = 4,
    n_MCMC = 1000,
    alpha_bart = 0.99,
)

credal_CP_gp.fit(
    X_train,
    Y_train,
    num_inducing_points = 100,
    lr_variational = 0.01,
    lr_hyperparams = 0.01,
    n_epochs = 300,
)

# ============================================
# GP calibration debugging
quantile_levels = [credal_CP_gp.alpha / 2, 1 - credal_CP_gp.alpha / 2]
quantile_levels = np.asarray(quantile_levels)

quantiles = np.asarray(quantile_levels)

credal_CP_gp.base_model.model.eval()
credal_CP_gp.base_model.likelihood.eval()

X_cal_standardized = torch.tensor(# 7. Inverse log transform if necessary
    credal_CP_gp.base_model.scaler_X.transform(X_cal), dtype=torch.float32
)

n_samples = 1000
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Get the latent function distribution f(X*) from the model
    latent_distribution = credal_CP_gp.base_model.model(X_cal_standardized)
    
    # Sample directly from the latent function distribution. 
    f_samples_tensor = latent_distribution.rsample(sample_shape=torch.Size([n_samples])) # (n_samples, n_test_samples)

# Calculate the quantile correction factor (Z_alpha * sigma_n)
# Observation noise variance (sigma_n^2) from the likelihood
sigma_n2 = credal_CP_gp.base_model.likelihood.noise.item() 
sigma_n = np.sqrt(sigma_n2)
Z_alphas = norm.ppf(quantiles)
quantile_correction = Z_alphas * sigma_n
correction_tensor = torch.tensor(quantile_correction, dtype=torch.float32)

# Calculate the quantile for each function sample
quantile_samples_tensor = f_samples_tensor[:, :, None] + correction_tensor[None, None, :] # (n_samples, n_test_samples, n_quantiles)
# Reshape to (n_test_samples, n_samples, n_quantiles)
quantile_samples = quantile_samples_tensor.permute(1, 0, 2).numpy()

# Flatten the first two axes for inverse scaling, keeping n_quantiles separate
original_shape = quantile_samples.shape
samples_2d = quantile_samples.reshape(-1, original_shape[2])

samples_inverse_scaled = credal_CP_gp.base_model.scaler_y.inverse_transform(samples_2d)
quantile_samples = samples_inverse_scaled.reshape(original_shape)

q_low_grid = quantile_samples[:, :, 0]
q_upp_grid = quantile_samples[:, :, 1]

# obtaining lower and upper quantiles for each x_calib
beta = 0.1
q_low_raw = np.quantile(q_low_grid, beta/2, axis=1)
q_upp_raw = np.quantile(q_upp_grid, 1 - beta/2, axis=1)

# with lower and upper quantiles, we can compute the modified nonconformity scores
nc_scores = np.maximum(q_low_raw - Y_cal, Y_cal - q_upp_raw)
n = len(nc_scores)
cutoff = np.quantile(nc_scores, 
                            q=np.ceil((n + 1) * (1 - 0.1)) / n)


# prediction
credal_CP_gp.base_model.model.eval()
credal_CP_gp.base_model.likelihood.eval()

X_test_standardized = torch.tensor(# 7. Inverse log transform if necessary
    credal_CP_gp.base_model.scaler_X.transform(X_test), dtype=torch.float32
)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Get the latent function distribution f(X*) from the model
    latent_distribution = credal_CP_gp.base_model.model(X_test_standardized)
    
    # Sample directly from the latent function distribution. 
    f_samples_tensor = latent_distribution.rsample(sample_shape=torch.Size([n_samples])) # (n_samples, n_test_samples)

# Calculate the quantile correction factor (Z_alpha * sigma_n)
# Observation noise variance (sigma_n^2) from the likelihood
sigma_n2 = credal_CP_gp.base_model.likelihood.noise.item() 
sigma_n = np.sqrt(sigma_n2)
Z_alphas = norm.ppf(quantiles)
quantile_correction = Z_alphas * sigma_n
correction_tensor = torch.tensor(quantile_correction, dtype=torch.float32)

# Calculate the quantile for each function sample
quantile_samples_tensor = f_samples_tensor[:, :, None] + correction_tensor[None, None, :] # (n_samples, n_test_samples, n_quantiles)
# Reshape to (n_test_samples, n_samples, n_quantiles)
quantile_samples = quantile_samples_tensor.permute(1, 0, 2).numpy()

# Flatten the first two axes for inverse scaling, keeping n_quantiles separate
original_shape = quantile_samples.shape
samples_2d = quantile_samples.reshape(-1, original_shape[2])

samples_inverse_scaled = credal_CP_gp.base_model.scaler_y.inverse_transform(samples_2d)
quantile_samples = samples_inverse_scaled.reshape(original_shape)

q_low_grid = quantile_samples[:, :, 0]
q_upp_grid = quantile_samples[:, :, 1]

q_low_raw = np.quantile(q_low_grid, beta/2, axis=1)
q_upp_raw = np.quantile(q_upp_grid, 1 - beta/2, axis=1)

lower_cp = q_low_raw - cutoff
upper_cp = q_upp_raw + cutoff

inside_gp = (Y_test >= lower_cp) & (Y_test <= upper_cp)
coverage_gp = inside_gp.mean()
n_test = len(Y_test)
print(f"GP test marginal coverage: {coverage_gp:.3f} ({inside_gp.sum()}/{n_test})")
print(f"Average GP interval width: {np.mean(upper_cp - lower_cp):.4f}")

# ============================================
# BART calibration debugging
quantile_levels = [credal_CP_bart.alpha / 2, 1 - credal_CP_bart.alpha / 2]
quantile_levels = np.asarray(quantile_levels)

# 1. Set new test data in the PyMC model
with credal_CP_bart.base_model.model_bart:
    credal_CP_bart.base_model.X_data.set_value(X_cal)
    
    if credal_CP_bart.base_model.type == "normal":
        if credal_CP_bart.base_model.var == "heteroscedastic":
            # Sample 'w' which contains both mu (w[0]) and log(sigma) (w[1])
            var_names = ["w"]
        elif credal_CP_bart.base_model.var == "homoscedastic":
            # Sample the mean function 'mu' and the scalar noise 'sigma'
            var_names = ["mu", "sigma"]
        else:
            raise ValueError(f"Unknown variance type: {credal_CP_bart.base_model.var}")

    # Sample the model parameters from the posterior (theta ~ P(theta|D))
    posterior_samples = pm.sample_posterior_predictive(
        trace=credal_CP_bart.base_model.mc_sample,
        var_names=var_names,
        predictions=True,
        random_seed=45,
        progressbar=credal_CP_bart.base_model.progressbar,
    )

# Calculate Z-scores for the required quantile levels
Z_alphas = norm.ppf(quantile_levels) # shape: (n_quantiles,)
n_quantiles = Z_alphas.shape[0]
n_test_samples = X_cal.shape[0]

w_samples = az.extract(
            posterior_samples,
            group="predictions",
            var_names=["w"],
        ).to_numpy()

# mu_samples shape: (n_samples, n_test_samples)
mu_samples = w_samples[0, :, :]
# sigma_samples shape: (n_samples, n_test_samples). Note: PyMC uses exp(w[1]) for sigma.
sigma_samples = np.exp(w_samples[1, :, :])

# Calculate quantiles: Q_alpha = mu + Z_alpha * sigma
quantile_samples_temp = (
    mu_samples[:, :, None] + sigma_samples[:, :, None] * Z_alphas[None, None, :]
)

quantile_samples = np.transpose(quantile_samples_temp, (1, 0, 2))

q_low_grid = quantile_samples[:, :, 0]
q_upp_grid = quantile_samples[:, :, 1]

beta = 0.1
q_low_raw = np.quantile(q_low_grid, beta/2, axis=0)
q_upp_raw = np.quantile(q_upp_grid, 1 - beta/2, axis=0)


nc_scores = np.maximum(q_low_raw - Y_cal, Y_cal - q_upp_raw)
n = len(nc_scores)
cutoff = np.quantile(nc_scores, 
                            q=np.ceil((n + 1) * (1 - 0.1)) / n)














