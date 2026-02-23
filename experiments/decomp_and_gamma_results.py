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
rng = np.random.default_rng(125)

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

# third type of data
def make_epistemic_mixture_gaps(rng, n):
    """
    Example where:
      - several gaps in the input space
      - noise is small and constant (mostly aleatoric)
      - mean is non-linear
    """

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

# Generate data
data = make_gap_epistemic_few_middle(n=1500, noise_std=0.1)

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
        k = 50,
    )

    bart_cutoff = credal_CP_bart.calibrate(
        X_cal, 
        Y_cal,
        N_samples_MC=1000,
        gamma_max = 0.95,
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

    # Create a 1x2 grid: left = uncertainty decomposition, right = prediction intervals
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
    
    # Increase caption/font sizes for this figure
    fontsize = 16
    plt.rcParams.update({
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    })

    # Make room at the top for legends placed above the titles
    fig.subplots_adjust(top=0.82)

    # Left: BART uncertainty decomposition
    ax = axes[0]
    l1, = ax.plot(xx, new_aleatoric, label="Aleatoric", color="C2")
    l2, = ax.plot(xx, new_epistemic, label="Epistemic", color="C1")
    ax.set_title("Normalized Uncertainty Decomposition (BART)")
    ax.set_ylabel("Uncertainty percentage", fontsize=fontsize)
    ax.set_xlabel("x", fontsize=fontsize)
    ax.grid(True)
    # Place legend above the title, horizontally
    ax.legend(handles=[l1, l2], loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=2, frameon=False)

    # Right: BART prediction intervals
    ax = axes[1]
    s = ax.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha=0.4, zorder=3)
    p = ax.fill_between(xx, lower, upper, color="C0", alpha=0.25, label="Prediction Interval")
    c = ax.plot(xx, center_ens, color="C0", lw=1, label="Interval Center")
    ax.set_title("Prediction Intervals (BART)")
    ax.set_xlabel("x", fontsize=fontsize)
    ax.grid(True)
    # Place legend above the title, horizontally
    handles = [p, c[0]]
    labels = ["Prediction Interval", "Interval Center"]
    ax.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 1.025), ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()

def plot_gamma(
    credal_CP_bart,
    X_test_grid,
    gamma_grid,
    X_test,
    Y_test,
):
    y_pred = credal_CP_bart.predict(X_test_grid)
    
     # Also get full prediction intervals (conformalized) for both methods to plot below
    pred_intervals = np.asarray(y_pred)

    lower, upper = pred_intervals[:, 0], pred_intervals[:, 1]

    xx = X_test_grid.ravel()

    # Prediction intervals and centers
    pred_intervals = np.asarray(y_pred)
    lower, upper = pred_intervals[:, 0], pred_intervals[:, 1]
    center = 0.5 * (lower + upper)

    xx = X_test_grid.ravel()
    gamma = np.asarray(gamma_grid).ravel()

    # Create a 2-row figure: top = prediction intervals, bottom = gamma(x)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Increase caption/font sizes for readability
    fontsize = 16
    plt.rcParams.update({
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    })

    # Top: BART prediction intervals (no x-axis ticks/label shown here)
    ax_top = axes[0]
    ax_top.scatter(X_test.ravel(), Y_test.ravel(), s=25, color="k", alpha=0.4, zorder=3)
    ax_top.fill_between(xx, lower, upper, color="C0", alpha=0.25, label="Prediction Interval")
    ax_top.plot(xx, center, color="C0", lw=1, label="Interval Center")
    ax_top.set_title("Prediction Intervals (BART)")
    ax_top.grid(True)
    ax_top.tick_params(labelbottom=False)  # hide x-scale on the top plot
    ax_top.legend(loc="upper left")

    # Bottom: gamma(x)
    ax_bot = axes[1]
    ax_bot.plot(xx, gamma, color="C4", lw=2, label=r"$\gamma(x)$")
    ax_bot.set_title("Adaptive gamma(x) (BART)")
    ax_bot.set_xlabel("x", fontsize=fontsize)  # use X-scale label only here
    ax_bot.set_ylabel("gamma", fontsize=fontsize)
    ax_bot.grid(True)
    ax_bot.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


####### Fitting bart-based method #######
credal_CP_bart, X_test_bart, Y_test_bart = fit_bart(train, cal, test)

X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)

# plotting uncertainty decomposition for BART
plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test_bart,
    Y_test_bart,
    normalize=True,
)

# un-normalized plot
plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test_bart,
    Y_test_bart,
    normalize=False,
)

# illustrating gamma(x) for BART
gamma_grid = credal_CP_bart.compute_gamma(X_test_grid)

plot_gamma(
    credal_CP_bart,
    X_test_grid,
    gamma_grid,
    X_test_bart,
    Y_test_bart,
)

############ Testing the other example with variable noise ############
np.random.seed(42)
n= 1500
data = make_variable_data(n)
train, rest = train_test_split(data, test_size=0.5, random_state=42)
cal, test = train_test_split(rest, test_size=0.5, random_state=42)

####### Fitting bart-based method #######
credal_CP_bart, X_test_bart, Y_test_bart = fit_bart(train, cal, test)

X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
# plotting uncertainty decomposition for BART
plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test_bart,
    Y_test_bart,
    normalize=True,
)

# last different epistemic scenario
n= 1500
data = make_epistemic_mixture_gaps(rng, n)
train, rest = train_test_split(data, test_size=0.5, random_state=42)
cal, test = train_test_split(rest, test_size=0.5, random_state=42)

credal_CP_bart, X_test_bart, Y_test_bart = fit_bart(train, cal, test)

X_test_grid = np.linspace(-4.35, 4.35, 700).astype(np.float32).reshape(-1, 1)
# plotting uncertainty decomposition for BART
plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test_bart,
    Y_test_bart,
    normalize=True,
)


# changing second epistemic scenario 
def make_variable_data_v2(n, std_dev=1/5):
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

    # sparse region [0, 0.4]: low variance and few points (hard, few points)
    mask_sparse = (x >= 0) & (x <= 0.4)
    sigma[mask_sparse] = 0.05 + 0.1 * np.abs(np.sin(12 * x[mask_sparse]))

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
data = make_variable_data_v2(n)
train, rest = train_test_split(data, test_size=0.5, random_state=42)
cal, test = train_test_split(rest, test_size=0.5, random_state=42)

credal_CP_bart, X_test_bart, Y_test_bart = fit_bart(train, cal, test)

X_test_grid = np.linspace(-1.15, 1.15, 500).astype(np.float32).reshape(-1, 1)
# plotting uncertainty decomposition for BART
plot_uncertainty_decomposition_bart(
    credal_CP_bart,
    X_test_grid,
    X_test_bart,
    Y_test_bart,
    normalize=True,
)



