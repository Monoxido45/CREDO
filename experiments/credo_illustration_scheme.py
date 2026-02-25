# package for the illustration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
config.update("jax_enable_x64", True)
from jax.scipy.stats import norm

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# package for the modelling of UQ
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.titlesize": 16,  # (no title used)
})

def simulate_data(seed=7, n_grid=220, n_pts=200):
    rng = np.random.default_rng(seed)
    xg = np.linspace(0, 1, n_grid)

    m = (
        0.25
        + 0.55 * np.sin(2 * np.pi * (xg - 0.08))
        - 0.25 * np.cos(2 * np.pi * (xg - 0.15))
        + 1.15 * (xg ** 3)
    )

    x = rng.uniform(0.08, 0.85, size=n_pts)
    m_x = np.interp(x, xg, m)
    noise = rng.normal(0, 0.18 + 0.10 * (1 - x), size=n_pts)
    y = m_x + noise

    return xg, x, y

# simulating data
xg, x, y = simulate_data()

# modeling through GP
key = jr.key(123)
D = gpx.Dataset(X=jnp.array(x.reshape(-1, 1), 
                            dtype=jnp.float64),
                            y=jnp.array(y.reshape(-1, 1),
                            dtype=jnp.float64))

# More expressive kernel: smooth long-range trend + locally-periodic component
kernel = (
    gpx.kernels.RBF()
    + (
        gpx.kernels.Matern52(lengthscale=0.12)
        * gpx.kernels.RBF()
    )
)
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
posterior = prior * likelihood

# Opimitizing the hyperparameters
opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D,
    trainable=gpx.parameters.Parameter,
)
print(-gpx.objectives.conjugate_mll(opt_posterior, D))

# obtaining sampled functions
num_samples = 30
key, subkey = jr.split(key)
latent_dist = posterior.predict(xg.reshape(-1, 1), train_data=D)
sample_functions = latent_dist.sample(subkey, sample_shape=(num_samples,))

sigma = opt_posterior.likelihood.obs_stddev.get_value()
qs = jnp.array([0.05, 0.95])
z_scores = norm.ppf(qs)

quantile_functions = sample_functions[..., None] + z_scores * sigma

q_l = quantile_functions[:, :, 0]
q_u = quantile_functions[:, :, 1]

# computing the beta and 1-beta quantiles among each of the quantile functions
beta = 0.1
q_l_beta = np.quantile(q_l, beta/2, axis=0)
q_u_beta = np.quantile(q_u, 1 - beta/2, axis=0)

# plotting those quantiles
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cL = default_colors[0]
cU = default_colors[1]


# STEP 1: conditional quantiles/model
fig, ax = plt.subplots(figsize=(6.0, 4.5))  # smaller figure
for i in range(q_l.shape[0]):
    ax.plot(xg, q_l[i], color=cL, alpha=0.16, linewidth=1.0, zorder=1)
    ax.plot(xg, q_u[i], color=cU, alpha=0.16, linewidth=1.0, zorder=1)

# plotting the data and the beta-quantiles    
ax.scatter(x, y, s=18, color=cL, alpha=0.75, zorder=3)

ax.set_xlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$\mathcal{F} = \{F_{\theta}(\cdot \mid x) : \theta \in \Theta\}$")
legend_handles = [
    Line2D([0], [0], color=cL, linewidth=3.2, alpha=0.5, label=r"$F_{\theta}^{-1}(\alpha/2 \mid x)$"),
    Line2D([0], [0], color=cU, linewidth=3.2, alpha=0.5, label=r"$F_{\theta}^{-1}(1-\alpha/2 \mid x)$"),
]
ax.legend(handles=legend_handles, loc="upper left", frameon=True)
fig.tight_layout()
fig.savefig("credo_functions.png", dpi=600, transparent=True)  # transparent background


# STEP 2: credal quantile envelope
fig, ax = plt.subplots(figsize=(6.0, 4.5))  # smaller figure
for i in range(q_l.shape[0]):
    ax.plot(xg, q_l[i], color=cL, alpha=0.16, linewidth=1.0, zorder=1)
    ax.plot(xg, q_u[i], color=cU, alpha=0.16, linewidth=1.0, zorder=1)

# plotting the data and the beta-quantiles    
ax.scatter(x, y, s=18, color=cL, alpha=0.45, zorder=3)
ax.plot(xg, q_l_beta, color=cL, alpha=0.98, linewidth=3.2, zorder=4)
ax.plot(xg, q_u_beta, color=cU, alpha=0.98, linewidth=3.2, zorder=4)

ax.fill_between(xg, q_l_beta, q_u_beta, color="lightgray", alpha=0.45, zorder=2)

lower_line = Line2D([0], [0], color=cL, linewidth=3.2, alpha=0.95, label=r"$\ell(x)$")
upper_line = Line2D([0], [0], color=cU, linewidth=3.2, alpha=0.95, label=r"$u(x)$")

legend_handles = [lower_line, upper_line]

ax.set_xlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$\mathcal{F}_{qq}(x) = \{F_{\theta}(\cdot \mid x) : \theta \in R_{qq}(x)\}$")
ax.legend(handles=legend_handles, loc="upper left", frameon=True)
fig.tight_layout()
fig.savefig("credo_unconformalized_credal.png", dpi=600, transparent=True)  # transparent background


# STEP 3: credal quantile envelope + conformalized
# conformalizing in a separate calibration
_, x_calib, y_calib = simulate_data(seed = 125)
X_calib = jnp.array(x_calib.reshape(-1, 1), dtype=jnp.float64)
y_calib = jnp.array(y_calib.reshape(-1, 1), dtype=jnp.float64)

num_samples = 1000
key, subkey = jr.split(key)
latent_dist = posterior.predict(X_calib, train_data=D)
sample_functions = latent_dist.sample(subkey, sample_shape=(num_samples,))

sigma = opt_posterior.likelihood.obs_stddev.get_value()
qs = jnp.array([0.05, 0.95])
z_scores = norm.ppf(qs)

quantile_functions = sample_functions[..., None] + z_scores * sigma

# lower and upper quantiles
q_l_calib = quantile_functions[:, :, 0]
q_u_calib = quantile_functions[:, :, 1]

# computing envelope
beta = 0.1
q_l_beta_calib = np.quantile(q_l_calib, beta/2, axis=0)
q_u_beta_calib = np.quantile(q_u_calib, 1 - beta/2, axis=0)

# computing conformity scores and cutoff
n = y_calib.shape[0]
alpha = 0.1
conformity_scores = jnp.maximum(q_l_beta_calib - y_calib.flatten(), y_calib.flatten() - q_u_beta_calib)
cutoff = np.quantile(conformity_scores, np.ceil((n + 1) * (1 - alpha)) / n)
# with cutoff computed, we can now conformalize the envelope
q_l_conformal = q_l_beta - cutoff
q_u_conformal = q_u_beta + cutoff

fig, ax = plt.subplots(figsize=(6.0, 4.5))  # smaller figure

# make the figure background fully transparent while keeping the axes face white
fig.patch.set_alpha(0.0)
ax.patch.set_facecolor("white")

# plotting the data and the beta-quantiles    
ax.scatter(x, y, s=18, color=cL, alpha=0.45, zorder=3)
ax.plot(xg, q_l_beta, color=cL, alpha=0.5, linewidth=3.2, zorder=4)
ax.plot(xg, q_u_beta, color=cU, alpha=0.5, linewidth=3.2, zorder=4)

# conformalized lines
ax.plot(xg, q_l_conformal, color=cL, alpha=0.95, linewidth=3.2, zorder=5)
ax.plot(xg, q_u_conformal, color=cU, alpha=0.95, linewidth=3.2, zorder=5)

# keep the shaded area inside the axes (use a semi-transparent color)
ax.fill_between(xg, q_l_conformal, q_u_conformal, color="lightgray", alpha=0.45, zorder=2)

lower_line = Line2D([0], [0], color=cL, linewidth=3.2, alpha=0.95, label=r"$\ell(x) - \hat{\tau}$")
upper_line = Line2D([0], [0], color=cU, linewidth=3.2, alpha=0.95, label=r"$u(x) + \hat{\tau}$")

legend_handles = [lower_line, upper_line]

ax.set_xlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"Conformalization of $[\ell(x), u(x)]$")
ax.legend(handles=legend_handles, loc="upper left", frameon=True)
fig.tight_layout()
fig.savefig("credo_conformalized_credal.png", dpi=600, transparent=True)  # outside the axes will be transparent

