######## Code for Predictive models
# used torch packages for neural networks and optimization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# BART package
import pymc as pm
import pymc_bart as pmb
from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule
import arviz as az

# dealing with multiprocessing and warnings
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# numpy, sklearn and scipy functions
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as st
from scipy.stats import norm, gamma
from scipy.optimize import brentq

# for plotting
import matplotlib.pyplot as plt

# tracking progress
from tqdm import tqdm

# importing jax and gpjax dependencies
from jaxtyping import install_import_hook
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from gpjax.likelihoods import (
    HeteroscedasticGaussian,
    LogNormalTransform,
    SoftplusTransform,
)
from gpjax.variational_families import (
    HeteroscedasticVariationalFamily,
)
from copy import deepcopy

# for the optimization of the variational parameters
from jax import config
config.update("jax_enable_x64", True)
import optax as ox


############### MDN with dropout and deep ensembles ###############
# General Base MDN architecture
class MDN_base(nn.Module):
    def __init__(self, input_shape, num_components, hidden_layers, dropout_rate=0.4):
        """
        Flexible MDN architecture

        Input: (i) input_shape (int): Input dimension.
               (ii) num_components (int): Number of mixture components.
               (iii) hidden_layers (list): List containing the number of neurons per hidden layer.
               (iv) dropout_rate (float): Dropout rate applied to each layer. Default is 0.4.
        """
        super(MDN_base, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
#        self.batch_norms = nn.ModuleList()

        # Creating hidden layers dinamically
        prev_units = input_shape
        for units in hidden_layers:
            self.layers.append(nn.Linear(prev_units, units))
            self.dropouts.append(nn.Dropout(dropout_rate))
#            self.batch_norms.append(nn.BatchNorm1d(units))
            prev_units = units

        self.fc_out = nn.Linear(prev_units, num_components * 3)
# self.batch_norms
    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
            # x = torch.tanh(layer(x))
 #           x = bn(x)
        x = self.fc_out(x)
        return x

# For gaussian Mixture Density
def gaussian_pdf(y, mu, sigma):
    return torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

# For gamma Mixture Density
def gamma_pdf(y, mu, sigma):
    alpha = (mu**2) / (sigma**2)
    beta = mu / (sigma**2)

    return ((beta ** (alpha)) * y ** (alpha - 1) * torch.exp(-beta * y)) / torch.exp(
        torch.lgamma(alpha)
    )

# MDN model class
class MDN_model(BaseEstimator):
    """
    Mixture Density Network model.
    """
    def __init__(
        self,
        input_shape,
        num_components=5,
        hidden_layers=[64],
        dropout_rate=0.4,
        base_model_type=None,
        alpha=None,
        normalize_y=False,
        log_y=False,
        type="gaussian",
    ):
        """
        Input: (i) input_shape: Input dimension.
               (ii) num_components: Number of mixture components.
               (iii) hidden_layers: List containing the number of neurons in each hidden layer. The length of the list determines the amount of hidden layers in the model.
               (iv) dropout_rate: Dropout Rate for each hidden layer. Default is 0.4.
               (v) base_model_type: Type of base model to be fitted. Default is None.
        """
        self.input_shape = input_shape
        self.num_components = num_components
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        # defining base model according to parameters
        self.model = MDN_base(
            self.input_shape, self.num_components, self.hidden_layers, self.dropout_rate
        )
        self.base_model_type = base_model_type
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.log_y = log_y
        self.type = type

    ##### auxiliary functions
    # MDN loss
    @staticmethod
    def mdn_loss(pi, mu, sigma, y_true, type="gaussian"):
        y_true = y_true.view(-1, 1).expand_as(mu)

        if type == "gaussian":
            # log N(y | mu, sigma)
            result = (
                -0.5 * ((y_true - mu) / sigma) ** 2
                - torch.log(sigma * np.sqrt(2.0 * np.pi))
            )
            log_pi = torch.log(pi + 1e-8)  # stability
            log_prob = torch.logsumexp(log_pi + result, dim=1)
            return -torch.mean(log_prob)

        elif type == "gamma":
            result = torch.sum(pi * gamma_pdf(y_true, mu, sigma), dim=1)
            return -torch.mean(torch.log(result + 1e-12))

        raise ValueError(f"Unknown type={type}")

    # Mixture coeficient obtention
    def get_mixture_coef(self, y_pred):
        K = self.num_components
        pi = F.softmax(y_pred[:, : K], dim=1)
        # pi = pi / pi.sum(dim=1, keepdim=True)
        if self.type == "gaussian":
            mu = y_pred[:, K : 2 * K]
            sigma = F.softplus(y_pred[:, 2 * K :]) + 1e-6
        elif self.type == "gamma":
            mu = F.softplus(y_pred[:, K : 2 * K])
            sigma = F.softplus(y_pred[:, 2 * K :])
        # sigma = torch.exp(y_pred[:, 2 * num_components:])
        return pi, mu, sigma

    def fit(
        self,
        X,
        y,
        proportion_train=0.7,
        epochs=500,
        lr=0.001,
        gamma=0.99,
        batch_size=32,
        step_size=5,
        weight_decay=0,
        verbose=0,
        patience=30,
        scale=False,
        random_seed_split=0,
        random_seed_fit=1250,
    ):
        """
        Fit MDN model.

        Input: (i) X (np.ndarray or torch.Tensor): Training input data.
               (ii) y (np.ndarray or torch.Tensor): Training target data.
               (iii) proportion_train (float): Proportion of data to be used for training (the rest for validation). Default is 0.7.
               (iv) epochs (int): Number of epochs for training. Default is 500.
               (v) lr (float): Learning rate for the optimizer. Default is 0.001.
               (vi) gamma (float): Gamma value for scheduler. Default is 0.99
               (vii) batch_size (int): Batch size. Default is 32.
               (viii) step_size (int): Step size for scheduler. Default is 5.
               (ix) weight_decay (float): Optimizer weight decay parameter. Default is 0.
               (x) verbose (int): Verbosity level (0, 1, or 2). If set to 0, does not print anything, if 1, prints the average loss of each epoch and if 2, prints the model learning curve at end of fitting.
               (xi) patience (int): Number of epochs with no improvement to trigger early stopping. Default is 30.
               (xii) scale (bool): Whether to scale or not the data. Default is False.
               (xiii) random_seed_split (int): Random seed fixed to perform data splitting. Default is 0.
               (xiv) random_seed_fit (int): Random seed fixed to model fitting. Default is 1250.

        Output: (i) fitted MDN_model object
        """
        self.optimizer = optim.Adamax(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        # Splitting data into train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=1 - proportion_train, random_state=random_seed_split
        )

        # checking if scaling is needed
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_val = self.scaler.transform(x_val)
            self.scale = True
        else:
            self.scale = False

        # checking if scaling the response is needed
        if self.normalize_y or self.log_y:
            if self.log_y:
                y_train, self.lmbda = st.boxcox(y_train)
                y_val = st.boxcox(y_val, lmbda=self.lmbda)

            elif self.normalize_y:
                self.y_scaler = StandardScaler()
                self.y_scaler.fit(y_train.reshape(-1, 1))
                y_train = self.y_scaler.transform(y_train.reshape(-1, 1)).flatten()
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        # checking if is an instance of numpy
        if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
            x_train, x_val = (
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(x_val, dtype=torch.float32),
            )
            y_train, y_val = (
                torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
                torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            )

        # Training and validation
        train_dataset = TensorDataset(
            x_train.clone().detach().float(),
            (
                y_train.clone().detach().float()
                if isinstance(y_train, torch.Tensor)
                else torch.tensor(y_train, dtype=torch.float32)
            ),
        )
        val_dataset = TensorDataset(
            x_val.clone().detach().float(),
            (
                y_val.clone().detach().float()
                if isinstance(y_val, torch.Tensor)
                else torch.tensor(y_val, dtype=torch.float32)
            ),
        )

        # Setting batch size
        batch_size_train = int(proportion_train * batch_size)
        batch_size_val = int((1 - proportion_train) * batch_size)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_train, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

        losses_train = []
        losses_val = []

        # early stopping
        best_val_loss = float("inf")
        counter = 0

        torch.manual_seed(random_seed_fit)
        torch.cuda.manual_seed(random_seed_fit)
        # Training loop
        for epoch in tqdm(range(epochs), desc="Fitting MDN model"):
            self.model.train()
            train_loss_epoch = 0

            # Looping through batches
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output_train = self.model(x_batch)  # Network output
                pi_train, mu_train, sigma_train = self.get_mixture_coef(output_train)
                loss_train = self.mdn_loss(pi_train, mu_train, sigma_train, y_batch)
                loss_train.backward()
                self.optimizer.step()
                train_loss_epoch += loss_train.item()

            # Computing validation loss
            self.model.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output_val = self.model(x_batch)  # Network output
                    pi_val, mu_val, sigma_val = self.get_mixture_coef(output_val)
                    loss_val = self.mdn_loss(
                        pi_val, mu_val, sigma_val, y_batch, type=self.type
                    )
                    val_loss_epoch += loss_val.item()

            # average loss by epoch
            train_loss_epoch /= len(train_loader)
            val_loss_epoch /= len(val_loader)
            losses_train.append(train_loss_epoch)
            losses_val.append(val_loss_epoch)

            self.scheduler.step()

            if verbose == 1:
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}"
                )

            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping in epoch {epoch} with best validation loss: {best_val_loss:.4f}"
                    )
                    break

        if verbose == 2:
            fig, ax = plt.subplots()
            ax.set_title("Training and Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            epochs_completed = len(losses_train)
            ax.set_xlim(0, epochs_completed)

            ax.plot(
                range(epochs_completed), losses_train, label="Train Loss", color="blue"
            )
            ax.plot(
                range(epochs_completed),
                losses_val,
                label="Validation Loss",
                color="green",
                linestyle="--",
            )
            plt.legend(loc="upper right")
            plt.show()

        return self

    def predict_mcdropout(self, x, num_samples=100, return_mean=False):
        """
        Make predictions with MC Dropout.

        Input:
            (i) x: Input data.
            (ii) num_samples: Number of Monte Carlo samples. Default is 100.
            (iii) return_mean: Whether to return the mean of the predictions. Default is False.

        Output:
            (i) Tuple containing the stacked tensors of the predictions (pi, mu, sigma) or their means if return_mean is True.
        """
        if isinstance(x, np.ndarray):
            if self.scale:
                x = self.scaler.transform(x)
            x = torch.tensor(x, dtype=torch.float32)

        # force only dropout layers to be in train mode
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        pi_predictions = []
        mu_predictions = []
        sigma_predictions = []

        for _ in range(num_samples):
            # model.train()
            with torch.no_grad():
                pred = self.model(x)
                pred_pi, pred_mu, pred_sigma = self.get_mixture_coef(pred)
                pi_predictions.append(pred_pi)
                mu_predictions.append(pred_mu)
                sigma_predictions.append(pred_sigma)
            # self.model.train(original_mode)

        # stacking predictions
        pi_predictions = torch.stack(pi_predictions)
        mu_predictions = torch.stack(mu_predictions)
        sigma_predictions = torch.stack(sigma_predictions)

        self.model.eval()
        if return_mean:
            pi_mean = torch.mean(pi_predictions, dim=0)
            mu_mean = torch.mean(mu_predictions, dim=0)
            sigma_mean = torch.mean(sigma_predictions, dim=0)

            return pi_mean, mu_mean, sigma_mean

        return pi_predictions, mu_predictions, sigma_predictions

    def set_type_base_model(self, base_model_type, alpha=None):
        """
        Set the type of base model for MDN.

        Input: (i) base_model_type (str): Type of base model to be set. Options are "regression", "quantile", or "density".
               (ii) alpha (float): Significance level for quantile base model. Default is None.
        """
        self.base_model_type = base_model_type
        if base_model_type == "quantile":
            self.alpha = alpha
        return self
    
    def predict(self, X_test, y_test=None, return_params=False, N = 1000):
        """
        Make predictions with MDN base model.

        Input: (i) X_test (np.ndarray or torch.Tensor): Input data.

        Output: (i) Numpy array containing the type of prediction selected for
            base model.
        """
        if self.scale:
            X_test = self.scaler.transform(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float32).clone().detach().float()
        self.model.eval()
        with torch.no_grad():
            pred_test = self.model(X_test)
            pi, mu, sigma = self.get_mixture_coef(pred_test)

            if self.base_model_type == "regression":
                mean_test = self.mixture_mean(pi, mu).numpy()
                if self.normalize_y:
                    mean_test = self.y_scaler.inverse_transform(mean_test.reshape(-1, 1)).flatten()
                return mean_test
            
            elif self.base_model_type == "quantile":
                alphas = [self.alpha / 2, 1 - (self.alpha / 2)]
                quantiles_test = self.mixture_quantile(
                    alphas, 
                    pi, 
                    mu, 
                    sigma, 
                    N=N, 
                    return_scale=True,
                    )
                return quantiles_test
            
            # density part
            elif self.base_model_type == "density" and (y_test is not None):
                # normalizing y_test if needed
                if self.log_y:
                    y_test = st.boxcox(y_test, lmbda=self.lmbda)
                if self.normalize_y:
                    y_test = self.y_scaler.transform(y_test.reshape(-1, 1)).flatten()

                # transforming to tensor also if needed
                if isinstance(y_test, np.ndarray):
                    y_test = torch.tensor(y_test, dtype=torch.float32)
                    density_test = self.mixture_density(y_test, pi, mu, sigma)
                if return_params:
                    return density_test, pi, mu, sigma
                else:
                    return density_test

            elif self.base_model_type == "density" and (y_test is None):
                return pi, mu, sigma

    def mixture_quantile(
            self, 
            alphas, 
            pi, 
            mu, 
            sigma, 
            rng=0, 
            method = "MC", 
            N=1000, 
            return_scale = False,
            ):   
        """
        Compute quantiles for each mixture component.

        Input:
            (i) alphas (list): List of quantiles (e.g., [0.1, 0.5, 0.9]).
            (ii) pi (np.ndarray): Mixture weights of shape (n_samples, n_components).
            (iii) mu (np.ndarray): Means of the components of shape (n_samples, n_components).
            (iv) sigma (np.ndarray): Standard deviations of the components of shape (n_samples, n_components).
            (v) rng: Fixed random Seed or generator. Default is 0.
            (vi) N (int): Number of samples to generate per mixture.

        Output:
            (i) np.ndarray: Quantile matrix of shape (n_sample, len(alphas)).
        """
        # fixing seed if a number is passed
        if rng == 0:
            rng = np.random.default_rng() # Advance state naturally
        else:
            rng = np.random.default_rng(rng) # Use the explicit seed

        pi = np.asarray(pi)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        n_sample, n_comp = pi.shape
        n_alphas = len(alphas)
        quantile_matrix = np.zeros((n_sample, n_alphas))
        if n_comp == 1:
            for j, alpha in enumerate(alphas):
                quantile_matrix[:, j] = mu.flatten() + sigma.flatten() * norm.ppf(alpha)
                if return_scale and self.normalize_y:
                    quantile_matrix[:, j] = self.y_scaler.inverse_transform(
                        quantile_matrix[:, j].reshape(-1, 1)).flatten()
        else:
            if method == "MC":
                samples = self.sample_from_mixture(pi, mu, sigma, N=N)
                # Quantile computation
                for j, alpha in enumerate(alphas):
                    quantile_matrix[:, j] = np.quantile(samples, alpha, axis=1)
                    if return_scale and self.normalize_y:
                        quantile_matrix[:, j] = self.y_scaler.inverse_transform(
                            quantile_matrix[:, j].reshape(-1, 1)).flatten()
                        
            elif method == "root":
                n_samples = pi.shape[0]
                for j, alpha in enumerate(alphas):
                    for i in range(n_samples):
                        quantile_matrix[i, j] = self.mixture_quantile_root(alpha, pi[i], mu[i], sigma[i])
                    if return_scale and self.normalize_y:
                        quantile_matrix[:, j] = self.y_scaler.inverse_transform(
                            quantile_matrix[:, j].reshape(-1, 1)).flatten()
                        
        return quantile_matrix

    def mixture_quantile_root(alpha, pi, mu, sigma):
        # pi, mu, sigma are for ONE observation (n_components,)
        def cdf_func(y):
            # Weighted sum of individual Gaussian CDFs
            return np.sum(pi * norm.cdf(y, mu, sigma)) - alpha
        
        # Define a wide search range based on the mixture means and stds
        low = np.min(mu - 5 * sigma)
        high = np.max(mu + 5 * sigma)
        
        return brentq(cdf_func, low, high)

    # Computing means using the mixture parameters
    @staticmethod
    def mixture_mean(pi, mu):
        """
        Compute mean of the Mixture density.

        Input:
            (i) pi (torch.Tensor): Mixture weights of shape (n_observations, n_components).
            (ii) mu (torch.Tensor): Means of the mixture components of shape (n_observations, n_components).

        Output:
            (i) torch.Tensor: Mixture means for each observation.
        """
        # Compute mean of mixture: weighted average of the components means
        mean = torch.sum(pi * mu, dim=1)
        return mean

    @staticmethod
    def mixture_cumulative(y, pi, mu, sigma, type="gaussian"):
        pi = np.array(pi)
        mu = np.array(mu)
        sigma = np.array(sigma)
        n_y = len(y)
        n = len(pi)
        cumulative_matrix = np.zeros((n, n_y))

        for i in range(n):
            if type == "gaussian":
                cumulative_matrix[i] = np.sum(pi[i] * norm.cdf(y, mu[i], sigma[i]))
            elif type == "gamma":
                alpha = (mu[i] ** 2) / (sigma[i] ** 2)
                beta = (mu[i]) / (sigma[i] ** 2)

                cumulative_matrix[i] = np.sum(
                    pi[i] * gamma.cdf(y, a=alpha, scale=1 / beta)
                )

        return cumulative_matrix

    def sample_from_mixture(self, pi, mu, sigma, rng=0, N=100):
        """
        Generates samples from the mixture network model for each observed sample x.

        Input:
            (i) pi (np.ndarray): Mixture weights of shape (n_samples, n_components).
            (ii) mu (np.ndarray): Means of the components of shape (n_samples, n_components).
            (iii) sigma (np.ndarray): Standard deviations of the components of shape (n_samples, n_components).
            (iv) rng: Fixed random Seed or generator. Default is 0.
            (v) N (int): Number of samples per mixture.

        Output:
            (i) np.ndarray: Generated samples, of shape (n_samples, N).
        """
        # fixing seed if a number is passed
        if rng == 0:
            rng = np.random.default_rng() # Advance state naturally
        else:
            rng = np.random.default_rng(rng) # Use the explicit seed

        # Ensures that pi, mu, and sigma are numpy arrays
        pi = np.asarray(pi)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        n_samples, n_comp = pi.shape

        # Normalize the weights to ensure they sum to 1
        pi /= np.sum(pi, axis=1, keepdims=True)

        # Repeat the weights for all samples
        pi_cumsum = np.cumsum(pi, axis=1)  # Cumulative sum for sampling
        random_vals = rng.random((n_samples, N))  # Random values between 0 and 1

        # Determine the chosen components for each sample
        components = (random_vals[..., None] < pi_cumsum[:, None, :]).argmax(axis=2)

        # Select the means and standard deviations of the chosen components
        chosen_mu = np.take_along_axis(mu, components, axis=1)
        chosen_sigma = np.take_along_axis(sigma, components, axis=1)

        if self.type == "gaussian":
            # Generate normal samples
            samples = rng.normal(loc=chosen_mu, scale=chosen_sigma)
        elif self.type == "gamma":
            alpha = (mu**2) / (sigma**2)
            beta = mu / (sigma**2)
            samples = rng.gamma(shape=alpha, scale=1 / beta)

        return samples

    def mdn_generate(self, pi, mu, sigma, rng=0):
        """
        Generates samples from the predictive distribution of the MDN using the monte-carlo parameter samples.

        Input:
        (i) pi (torch.Tensor): Tensor of MC dropout mixture probabilities for each sample (n_mcdropout, n_samples, n_components).
        (ii) mu (torch.Tensor): Tensor of MC dropout mixture means for each sample (n_mcdropout, n_samples, n_components).
        (iii) sigma (torch.Tensor): Tensor of MC dropout mixture standard deviations for each sample (n_mcdropout, n_samples, n_components).
        (iv) rng: Fixed random Seed or generator. Default is 0.

        Output:
        (i) samples (np.ndarray): Generated samples from the mixture of normals (n_samples, n_mcdropout).
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(42)

        # Converting to numpy
        pi_np = pi.detach().numpy()
        mu_np = mu.detach().numpy()
        sigma_np = sigma.detach().numpy()

        n_mc, n_obs, n_comp = pi.shape
        sample = np.zeros((n_obs, n_mc))

        for i in range(n_obs):
            for j in range(n_mc):
                # normalizing pi
                pi_i = pi_np[j, i, :]
                pi_i = pi_i / pi_i.sum()

                # sampling based on component
                component = rng.choice(n_comp, size=1, p=pi_i)

                if self.type == "gaussian":
                    sample[i, j] = rng.normal(
                        mu_np[j, i, component[0]], sigma_np[j, i, component[0]]
                    )
                elif self.type == "gamma":
                    alpha = mu_np[j, i, component[0]] ** 2 / (
                        sigma_np[j, i, component[0]] ** 2
                    )
                    beta = mu_np[j, i, component[0]] / (
                        sigma_np[j, i, component[0]] ** 2
                    )

                    sample[i, j] = rng.gamma(shape=alpha, scale=1 / beta)

        return sample

    # HPD related functions
    def mixture_density(self, y_test, pi, mu, sigma):
        """
        Computes the density of the mixture of normal distributions for each mixture and score.

        Input:
            (i) pi (torch.Tensor): Mixture weights of shape (n_samples, n_components).
            (ii) mu (torch.Tensor): Means of the components of shape (n_samples, n_components).
            (iii) sigma (torch.Tensor): Standard deviations of the components of shape (n_samples, n_components).

        Output:
            (i) density_values (torch.Tensor): Tensor of shape (n_samples,) containing the density values
            for each mixture and score.
        """
        if isinstance(pi, np.ndarray):
            pi = torch.tensor(pi)
        if isinstance(mu, np.ndarray):
            mu = torch.tensor(mu)
        if isinstance(sigma, np.ndarray):
            sigma = torch.tensor(sigma)

        y = y_test.view(-1, 1)

        density_values = torch.sum(pi * gaussian_pdf(y, mu, sigma), dim=1)
        return density_values

    def mixture_cdf_density(self, y_test, X_test):
        """
        Computes the CDF of the mixture of normal distributions for each mixture and score.
        Input:
            (i) y_test (torch.Tensor): Tensor of shape (n_samples,) with the scores for which to compute the CDF.
            (ii) X_test (torch.Tensor): Tensor of shape (n_samples, n_features) with the input data.
        Output:
            (i) CDF of density values (torch.Tensor): Tensor of shape (n_samples,) containing the CDF of the density for each testing sample
        """
        # first computing the density for observed values
        dens_values, pi, mu, sigma = self.predict(X_test, y_test, return_params=True)

        # sampling to derive CDF of density
        sample = self.sample_from_mixture(pi, mu, sigma, N=1000)
        sample = torch.tensor(sample, dtype=torch.float32)

        # computing density for each sample
        cdf_dens_values = np.zeros(y_test.shape[0])
        for i in range(sample.shape[0]):
            fixed_dens = dens_values[i].numpy()
            pi_value, mu_value, sigma_value = pi[i], mu[i], sigma[i]
            new_y = sample[i, :].reshape(-1, 1)

            pi_repeated = pi_value.repeat(new_y.shape[0], 1)
            mu_repeated = mu_value.repeat(new_y.shape[0], 1)
            sigma_repeated = sigma_value.repeat(new_y.shape[0], 1)

            # computing density for each sample
            dens_values_sim = self.mixture_density(
                new_y, pi_repeated, mu_repeated, sigma_repeated
            ).numpy()

            # compute the mean of the dens_values_sim that are less than the fixed density
            cdf_dens_values[i] = np.mean((dens_values_sim <= fixed_dens) + 0)

        return cdf_dens_values

    def predict_cdf_cutoff(self, X, cutoff, num_samples=1000):
        # predicting first the mixture parameters for X
        pi, mu, sigma = self.predict(X)

        # generating samples from the mixture
        sample = self.sample_from_mixture(pi, mu, sigma, N=num_samples)
        sample = torch.tensor(sample, dtype=torch.float32)

        # computing density for each sample
        cutoff_hpd = np.zeros(X.shape[0])
        for i in range(sample.shape[0]):
            pi_value, mu_value, sigma_value = pi[i], mu[i], sigma[i]
            new_y = sample[i, :].reshape(-1, 1)

            pi_repeated = pi_value.repeat(new_y.shape[0], 1)
            mu_repeated = mu_value.repeat(new_y.shape[0], 1)
            sigma_repeated = sigma_value.repeat(new_y.shape[0], 1)

            # computing density for each sample
            dens_values_sim = self.mixture_density(
                new_y, pi_repeated, mu_repeated, sigma_repeated
            )

            # compute the mean of the dens_values_sim that are less
            # than the cutoff
            cutoff_hpd[i] = np.quantile(dens_values_sim.numpy(), cutoff)

            # compute the mean of the dens_values_sim that are less
            # than the fixed density
        return cutoff_hpd

    def predict_mixture_density(self, X, y_grid):
        # predicting first the mixture parameters for X
        pi, mu, sigma = self.predict(X)

        if self.normalize_y:
            y_grid = self.y_scaler.transform(
                y_grid.reshape(-1, 1),
            ).flatten()

        if isinstance(y_grid, np.ndarray):
            y_grid = torch.tensor(y_grid, dtype=torch.float32)

        # computing density for each sample
        dens_values = np.zeros((X.shape[0], y_grid.shape[0]))
        for i in range(X.shape[0]):
            pi_value, mu_value, sigma_value = pi[i], mu[i], sigma[i]
            new_y = y_grid.reshape(-1, 1)

            pi_repeated = pi_value.repeat(new_y.shape[0], 1)
            mu_repeated = mu_value.repeat(new_y.shape[0], 1)
            sigma_repeated = sigma_value.repeat(new_y.shape[0], 1)

            # computing density for each sample
            dens_values_sim = self.mixture_density(
                new_y, pi_repeated, mu_repeated, sigma_repeated
            )

            # compute the mean of the dens_values_sim that are less
            # than the cutoff
            dens_values[i, :] = dens_values_sim.numpy()

            # compute the mean of the dens_values_sim that are less
            # than the fixed density
        return dens_values

    def mdn_generate_densities(self, pi, mu, sigma, rng=0):
        if isinstance(rng, int):
            rng = np.random.default_rng(42)

        # Converting to numpy
        pi_np = pi.detach().numpy()
        mu_np = mu.detach().numpy()
        sigma_np = sigma.detach().numpy()

        n_mc, n_obs, n_comp = pi.shape
        sample = np.zeros((n_obs, n_mc))

        for i in range(n_obs):
            for j in range(n_mc):
                # normalizing pi
                pi_i = pi_np[j, i, :]
                pi_i = pi_i / pi_i.sum()

                # sampling based on component
                component = rng.choice(n_comp, size=1, p=pi_i)

                if self.type == "gaussian":
                    sample[i, j] = rng.normal(
                        mu_np[j, i, component[0]], sigma_np[j, i, component[0]]
                    )
                # computing density for each sample
                density = np.sum(
                    pi_i
                    * norm.pdf(
                        sample[i, j],
                        loc=mu_np[j, i, :],
                        scale=sigma_np[j, i, :],
                    )
                )
                sample[i, j] = density
        return sample

    def mixture_cdf_no_scale(self, sample, scores):
        """
        Computes the CDF of the mixture of normal distributions for each mixture and score
        using samples generated by Monte Carlo.

        Input:
        (i) sample (torch.Tensor or np.ndarray): Tensor of shape (n_samples, n_mcdropout),
        where n_samples is the number of samples, n_mcdropout is the number of mixtures generated by MC Dropout.
        (ii) scores (torch.Tensor or np.ndarray): Tensor of shape (n_samples,) with the scores for which to compute the CDF.

        Output:
        (i) cdf_values (torch.Tensor): Tensor of shape (n_samples,) containing the CDF values
        for each mixture and score.
        """

        if isinstance(sample, np.ndarray):
            sample = torch.tensor(sample)
        if isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        n_samples, n_mixtures = sample.shape

        cdf_values = torch.zeros((n_samples,))

        for j in range(n_samples):
            score = scores[j]

            cdf_values[j] = torch.sum(sample[j, :] <= score).float() / n_mixtures

        return cdf_values

    def mixture_cdf(self, sample, scores):
        """
        Computes the CDF of the mixture of normal distributions for each mixture and score
        using samples generated by Monte Carlo.

        Input:
        (i) sample (torch.Tensor or np.ndarray): Tensor of shape (n_samples, n_mcdropout),
        where n_samples is the number of samples, n_mcdropout is the number of mixtures generated by MC Dropout.
        (ii) scores (torch.Tensor or np.ndarray): Tensor of shape (n_samples,) with the scores for which to compute the CDF.

        Output:
        (i) cdf_values (torch.Tensor): Tensor of shape (n_samples,) containing the CDF values
        for each mixture and score.
        """

        if isinstance(sample, np.ndarray):
            sample = torch.tensor(sample)
        if isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        if self.log_y:
            scores = st.boxcox(scores, lmbda=self.lmbda)
            if self.normalize_y:
                scores = self.y_scaler.transform(scores.reshape(-1, 1)).flatten()
        elif self.normalize_y:
            scores = self.y_scaler.transform(scores.reshape(-1, 1)).flatten()

        n_samples, n_mixtures = sample.shape

        cdf_values = torch.zeros((n_samples,))

        for j in range(n_samples):
            score = scores[j]

            cdf_values[j] = torch.sum(sample[j, :] <= score).float() / n_mixtures

        return cdf_values

    def mixture_ppf(self, samples, probs):
        """
        Computes the percentiles (quantiles) for each mixture and given probability
        using samples generated by Monte Carlo, with linear interpolation.

        Input:
        (i) samples (torch.Tensor): Tensor of shape (n_samples, n_mcdropouts),
        where n_samples is the number of samples, n_mcdropouts is the number of MC samples.
        (ii) probs (torch.Tensor): Tensor of shape (len(probs),) with the probabilities for which to compute the quantile.

        Output:
        (i) quantiles (torch.Tensor): Tensor of shape (n_samples, len(probs)) containing the quantiles
        for each mixture and probability.
        """
        if isinstance(samples, np.ndarray):
            samples = torch.tensor(samples, dtype=torch.float32)
        if isinstance(probs, np.ndarray):
            probs = torch.tensor(probs, dtype=torch.float32)

        n_samples, _ = samples.shape
        n_probs = len(probs)

        quantiles = torch.zeros((n_samples, n_probs), dtype=torch.float32)

        for j in range(n_samples):
            for k, prob in enumerate(probs):

                quantiles[j, k] = np.quantile(samples[j, :], prob)

        return quantiles

# Deep Ensembles MDN class
class DE_MDN_model(BaseEstimator):
    """
    Deep Ensembles MDN model for regression tasks.
    """

    def __init__(
        self,
        input_shape,
        n_models=15,
        num_components=3,
        hidden_layers=[64],
        dropout_rate=0.4,
        base_model_type=None,
        alpha=None,
        normalize_y=False,
        log_y=False,
        type="gaussian",
    ):
        """
        Input:
            (i) n_models (int): Number of models in the ensemble.
            (ii) n_epochs (int): Number of training epochs.
            (iii) batch_size (int): Batch size for training.
            (iv) learning_rate (float): Learning rate for the optimizer.
            (v) scale_x (bool): Whether to standardize input features.
            (vi) log_y (bool): Whether to apply log transformation to the target variable.
            (vii) random_seed (int): Random seed for reproducibility.

        Output:
            (i) Initialized Deep_ensembles_model object.
        """
        self.input_shape = input_shape
        self.n_models = n_models
        self.num_components = num_components
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        # defining base model according to parameters
        self.model = MDN_base(
            self.input_shape, self.num_components, self.hidden_layers, self.dropout_rate
        )
        self.base_model_type = base_model_type
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.type = type
        self.log_y = log_y
    
    @staticmethod
    def mdn_loss(pi, mu, sigma, y_true, type="gaussian"):
        y_true = y_true.view(-1,1).expand_as(mu)

        if type == "gaussian":
            result = -0.5 * ((y_true - mu) / sigma) ** 2 - torch.log(sigma * np.sqrt(2.0 * np.pi))
            log_pi = torch.log(pi + 1e-8)  # numerical stability
        elif type == "gamma":
            result = torch.sum(pi * gamma_pdf(y_true, mu, sigma), dim=1)

        log_prob = torch.logsumexp(log_pi + result, dim=1)
        return -torch.mean(log_prob)
    
    # Mixture coeficient obtention
    def get_mixture_coef(self, y_pred):
        pi = F.softmax(y_pred[:, : self.num_components], dim=1)
        # pi = pi / pi.sum(dim=1, keepdim=True)
        if self.type == "gaussian":
            mu = y_pred[:, self.num_components : 2 * self.num_components]
            sigma = F.softplus(y_pred[:, 2 * self.num_components :])

        elif self.type == "gamma":
            mu = F.softplus(y_pred[:, self.num_components : 2 * self.num_components])
            sigma = F.softplus(y_pred[:, 2 * self.num_components :])
        # sigma = torch.exp(y_pred[:, 2 * num_components:])
        return pi, mu, sigma
    
    def mixture_quantile(self, alphas, pi, mu, sigma, rng=0, N=1000):
        """
        Compute quantiles for each mixture component.

        Input:
            (i) alphas (list): List of quantiles (e.g., [0.1, 0.5, 0.9]).
            (ii) pi (np.ndarray): Mixture weights of shape (n_samples, n_components).
            (iii) mu (np.ndarray): Means of the components of shape (n_samples, n_components).
            (iv) sigma (np.ndarray): Standard deviations of the components of shape (n_samples, n_components).
            (v) rng: Fixed random Seed or generator. Default is 0.
            (vi) N (int): Number of samples to generate per mixture.

        Output:
            (i) np.ndarray: Quantile matrix of shape (n_sample, len(alphas)).
        """
        # fixing seed if a number is passed
        if rng == 0:
            rng = np.random.default_rng() # Advance state naturally
        else:
            rng = np.random.default_rng(rng) # Use the explicit seed
        pi = np.asarray(pi)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        n_sample, _ = pi.shape
        n_alphas = len(alphas)

        # N samples from each mixture
        samples = self.sample_from_mixture(pi, mu, sigma, N=N)

        # Quantile computation
        quantile_matrix = np.zeros((n_sample, n_alphas))
        for j, alpha in enumerate(alphas):
            quantile_matrix[:, j] = np.quantile(samples, alpha, axis=1)

        return quantile_matrix
    
    def sample_from_mixture(self, pi, mu, sigma, rng=0, N=100):
        """
        Generates samples from the mixture network model for each observed sample x.

        Input:
            (i) pi (np.ndarray): Mixture weights of shape (n_samples, n_components).
            (ii) mu (np.ndarray): Means of the components of shape (n_samples, n_components).
            (iii) sigma (np.ndarray): Standard deviations of the components of shape (n_samples, n_components).
            (iv) rng: Fixed random Seed or generator. Default is 0.
            (v) N (int): Number of samples per mixture.

        Output:
            (i) np.ndarray: Generated samples, of shape (n_samples, N).
        """
        # fixing seed if a number is passed
        if rng == 0:
            rng = np.random.default_rng() # Advance state naturally
        else:
            rng = np.random.default_rng(rng) # Use the explicit seed

        # Ensures that pi, mu, and sigma are numpy arrays
        pi = np.asarray(pi)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        n_samples, n_comp = pi.shape

        # Normalize the weights to ensure they sum to 1
        pi /= np.sum(pi, axis=1, keepdims=True)

        # Repeat the weights for all samples
        pi_cumsum = np.cumsum(pi, axis=1)  # Cumulative sum for sampling
        random_vals = rng.random((n_samples, N))  # Random values between 0 and 1

        # Determine the chosen components for each sample
        components = (random_vals[..., None] < pi_cumsum[:, None, :]).argmax(axis=2)

        # Select the means and standard deviations of the chosen components
        chosen_mu = np.take_along_axis(mu, components, axis=1)
        chosen_sigma = np.take_along_axis(sigma, components, axis=1)

        if self.type == "gaussian":
            # Generate normal samples
            samples = rng.normal(loc=chosen_mu, scale=chosen_sigma)
        elif self.type == "gamma":
            alpha = (mu**2) / (sigma**2)
            beta = mu / (sigma**2)
            samples = rng.gamma(shape=alpha, scale=1 / beta)

        return samples

    def fit(
        self,
        X,
        y,
        n_epochs=500,
        patience=15,
        lr=1e-3,
        weight_decay=1e-4,
        proportion_train=0.7,
        gamma=0.99,
        batch_size=32,
        step_size=3,
        scale=False,
        random_seed_split=0,
        random_seed_fit=1250,
    ):
        """
        Fit the Deep Ensembles MDN model.

        Input:
            (i) X_train: Training input data.
            (ii) y_train: Training target data.
            (iii) n_epochs (int): Number of training epochs. Default is 500.
            (iv) patience (int): Number of epochs with no improvement to trigger early stopping. Default is 15.
            (v) lr (float): Learning rate for the optimizer. Default is 1e-3.
            (vi) weight_decay (float): Weight decay for the optimizer. Default is 1e-4.
            (vii) proportion_train (float): Proportion of data to be used for training. Default is 0.7.
            (viii) gamma (float): Learning rate decay factor. Default is 0.99.
            (ix) batch_size (int): Batch size for training. Default is 32.
            (x) step_size (int): Step size for learning rate scheduler. Default is 5.
            (xi) scale (bool): Whether to standardize input features. Default is False.
            (xii) init_seed (int or None): Initial random seed for reproducibility. Default is None.
            (xiii) random_seed_split (int): Random seed for data splitting. Default is 0.
            (xiv) random_seed_fit (int): Random seed for model fitting. Default is 1250.

        Output:
            (i) Trained Deep_ensembles_model object.
        """
        # Splitting data into train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            X, 
            y, 
            test_size=1 - proportion_train, 
            random_state=random_seed_split,
        )

        # checking if scaling is needed
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_val = self.scaler.transform(x_val)
            self.scale = True
        else:
            self.scale = False

        # checking if scaling the response is needed
        if self.normalize_y or self.log_y:
            if self.log_y:
                y_train, self.lmbda = st.boxcox(y_train)
                y_val = st.boxcox(y_val, lmbda=self.lmbda)
                if self.normalize_y:
                    self.y_scaler = StandardScaler()
                    self.y_scaler.fit(y_train.reshape(-1, 1))

                    y_train = self.y_scaler.transform(y_train.reshape(-1, 1)).flatten()
                    y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

            elif self.normalize_y:
                self.y_scaler = StandardScaler()
                self.y_scaler.fit(y_train.reshape(-1, 1))
                y_train = self.y_scaler.transform(y_train.reshape(-1, 1)).flatten()
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        # checking if is an instance of numpy
        if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
            x_train, x_val = (
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(x_val, dtype=torch.float32),
            )
            y_train, y_val = (
                torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
                torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            )

        # Training and validation
        train_dataset = TensorDataset(
            x_train.clone().detach().float(),
            (
                y_train.clone().detach().float()
                if isinstance(y_train, torch.Tensor)
                else torch.tensor(y_train, dtype=torch.float32)
            ),
        )
        val_dataset = TensorDataset(
            x_val.clone().detach().float(),
            (
                y_val.clone().detach().float()
                if isinstance(y_val, torch.Tensor)
                else torch.tensor(y_val, dtype=torch.float32)
            ),
        )

        # Setting batch size
        batch_size_train = int(proportion_train * batch_size)
        batch_size_val = int((1 - proportion_train) * batch_size)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size_val, 
            shuffle=False,
            )
        
        # bootstraping the dataloader for the current model
        train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True
        )
        
        torch.manual_seed(random_seed_fit)
        torch.cuda.manual_seed(random_seed_fit)
        # Fitting each model in the ensemble
        self.models = []
        for i in tqdm(range(self.n_models), desc = "Fitting Deep Ensemble MDN models"):
            model_i = self.fit_one_model(
                train_loader,
                val_loader,
                patience,
                n_epochs,
                lr,
                weight_decay,
                step_size,
                gamma,
            )
            self.models.append(model_i)

        return self

    def fit_one_model(
            self,
            train_loader,
            val_loader,
            patience,
            n_epochs,
            lr,
            weight_decay,
            step_size,
            gamma,
        ):
        """
        Fit a single MDN model.
        Input:
            (i) train_loader: DataLoader for training data.
            (ii) val_loader: DataLoader for validation data.
            (iii) patience (int): Number of epochs with no improvement to trigger early stopping.
            (iv) n_epochs (int): Maximum number of training epochs.
            (v) verbose (int): Verbosity level (0, 1, or 2).
        Output:
            (i) Trained MDN model.
        """

        model = MDN_base(
            self.input_shape, 
            self.num_components, 
            self.hidden_layers, 
            self.dropout_rate,
        )

        optimizer = optim.Adamax(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        losses_train = []
        losses_val = []

        # early stopping
        best_val_loss = float("inf")
        counter = 0

        # Training loop
        for epoch in range(n_epochs):
            model.train()
            train_loss_epoch = 0

            # Looping through batches
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output_train = model(x_batch)  # Network output
                pi_train, mu_train, sigma_train = self.get_mixture_coef(output_train)
                loss_train = self.mdn_loss(
                    pi_train, 
                    mu_train, 
                    sigma_train, 
                    y_batch,
                    type=self.type,
                )
                loss_train.backward()
                optimizer.step()
                train_loss_epoch += loss_train.item()

            # Computing validation loss
            model.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output_val = model(x_batch)  # Network output
                    pi_val, mu_val, sigma_val = self.get_mixture_coef(output_val)
                    loss_val = self.mdn_loss(
                        pi_val, mu_val, sigma_val, y_batch, type=self.type
                    )
                    val_loss_epoch += loss_val.item()

            # average loss by epoch
            train_loss_epoch /= len(train_loader)
            val_loss_epoch /= len(val_loader)
            losses_train.append(train_loss_epoch)
            losses_val.append(val_loss_epoch)

            scheduler.step()

            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
            
        return model

    def predict_ensemble(
            self,
            X_test,
    ):
        """
        Predict parameters from each model in the ensemble.

        Input:
            (i) X_test (array-like): Test input data.
        Output:
            (i) pi_ensemble (np.ndarray): Mixture weights from each model, shape (n_test_samples, n_models, n_components).
            (ii) mu_ensemble (np.ndarray): Means from each model, shape (n_test_samples, n_models, n_components).
            (iii) sigma_ensemble (np.ndarray): Standard deviations from each model, shape (n_test_samples, n_models, n_components).
        """
        if self.scale:
            X_test_standardized = self.scaler.transform(X_test)
        else:
            X_test_standardized = X_test

        n_test_samples = X_test_standardized.shape[0]
        n_models = len(self.models)

        # Arrays to hold mixture parameters from each model
        pi_ensemble = np.zeros((n_models, n_test_samples, self.num_components))
        mu_ensemble = np.zeros((n_models, n_test_samples, self.num_components))
        sigma_ensemble = np.zeros((n_models, n_test_samples, self.num_components))

        # Get predictions from each model in the ensemble
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_test_standardized, dtype=torch.float32)
                y_pred = model(X_tensor)
                pi, mu, sigma = self.get_mixture_coef(y_pred)
                
                pi_ensemble[i, :, :] = pi.numpy()
                mu_ensemble[i, :, :] = mu.numpy()
                sigma_ensemble[i, :, :] = sigma.numpy()

        return pi_ensemble, mu_ensemble, sigma_ensemble

    def predict(self,
                X_test,
                ):
        """
        Predict quantiles for the test data using the ensemble of MDN models.
        Input:
            (i) X_test (array-like): Test input data.
        Output:
            (i) quantiles_test (np.ndarray): Predicted quantiles for the test data, shape (n_test_samples, 2).
        """ 
        pi_ensemble, mu_ensemble, sigma_ensemble = self.predict_ensemble(X_test)
        if self.base_model_type == "quantile":
            alphas = [self.alpha / 2, 1 - (self.alpha / 2)]
            low_quant_mod = np.zeros((X_test.shape[0], len(self.models)))
            up_quant_mod = np.zeros((X_test.shape[0], len(self.models)))
            for i in range(len(self.models)):
                quantiles_mod = self.mixture_quantile(
                    alphas, 
                    pi_ensemble[i, :, :], 
                    mu_ensemble[i, :, :], 
                    sigma_ensemble[i, :, :],
                    )
                low_quant_mod[:, i] = quantiles_mod[:, 0]
                up_quant_mod[:, i] = quantiles_mod[:, 1]
                
            # averaging over models
            low_quantiles_test = np.mean(low_quant_mod, axis=1)
            up_quantiles_test = np.mean(up_quant_mod, axis=1)
            quantiles_test = np.column_stack((low_quantiles_test, up_quantiles_test))
            return quantiles_test
        else:
            raise NotImplementedError("Only quantile base model prediction is implemented.")
        # TODO: implement regression and density base model predictions
    
    def set_type_base_model(self, base_model_type, alpha=None):
        """
        Set the type of base model for MDN.

        Input: (i) base_model_type (str): Type of base model to be set. Options are "regression", "quantile", or "density".
               (ii) alpha (float): Significance level for quantile base model. Default is None.
        """
        self.base_model_type = base_model_type
        if base_model_type == "quantile":
            self.alpha = alpha
        return self
    
############### Quantile neural network with dropout and deep ensembles ###############
# Direct quantile regression for regression tasks using dropout and deep ensembles. 
# Different output layer and loss function than MDN. 
# The output layer has 2 neurons, one for the lower quantile and one for the upper quantile.
# The loss function is the pinball loss.

# Quantile regression architecture made by Rosselini et.al. 2024

class QuantileRegressionNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, p_dropout=0.2):
        super(QuantileRegressionNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        self.p_dropout = p_dropout

        prev_units = input_size
        for units in hidden_layers:
            self.layers.append(nn.Linear(prev_units, units))
            self.dropouts.append(nn.Dropout(p_dropout))
            prev_units = units
        
        self.fc_out = nn.Linear(prev_units, output_size)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        x = self.fc_out(x)
        return x

# Quantile regression model with dropout and also compatible for deep ensembles
class QuantileRegressionNN(BaseEstimator):
    def __init__(
        self,
        input_size,
        alpha = 0.1,
        dropout=0.3,
        hidden_layers=[64, 64],
        use_gpu=True,
        undo_crossing=True,
    ):
        self.alpha = alpha
        self.quantiles = [alpha/2, 1-alpha/2]
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.undo_crossing = undo_crossing
        self.use_gpu = use_gpu
        self.undo_crossing = undo_crossing


        self.model = QuantileRegressionNet(
            input_size=input_size, 
            output_size=len(self.quantiles), 
            hidden_layers=hidden_layers, 
            p_dropout=dropout
        ).to(self.device)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    @staticmethod
    def quantile_loss(y_pred, y_true, quantiles):
        loss = 0
        for i, q in enumerate(quantiles):
            error = y_true - y_pred[:, i:i+1]
            loss += torch.max((q - 1) * error, q * error).mean()
        return loss
    
    def fit(
            self, 
            X, 
            y,
            weight_decay=0,
            scheduler_step=20,
            scheduler_gamma=0.99,
            epochs=500, 
            lr=1e-3, 
            batch_size=32, 
            patience=30, 
            verbose=1, 
            split_random_state=42,
            fit_random_state=1250,
            ):
        # Preprocessing
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=split_random_state)

        torch.manual_seed(fit_random_state)
        torch.cuda.manual_seed(fit_random_state)
        
        x_train = torch.tensor(self.scaler_x.fit_transform(x_train), dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.scaler_y.fit_transform(y_train.reshape(-1, 1)), dtype=torch.float32).to(self.device)
        x_val = torch.tensor(self.scaler_x.transform(x_val), dtype=torch.float32).to(self.device)
        y_val = torch.tensor(self.scaler_y.transform(y_val.reshape(-1, 1)), dtype=torch.float32).to(self.device)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_step, 
            gamma=scheduler_gamma
        )

        best_val_loss = float('inf')
        best_model_state = None
        counter = 0

        for epoch in tqdm(range(epochs), disable=(verbose==0)):
            self.model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = self.quantile_loss(pred, by, self.quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            # 3. Early Stopping Check
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(x_val)
                val_loss = self.quantile_loss(val_pred, y_val, self.quantiles).item()

            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                if verbose: print(f"Early stopping at epoch {epoch}")
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)
    
    def predict(self, X, n_mc=500, use_mcdropout=True):
        """
        Predict quantiles.
        Args:
            X: Input data.
            n_mc: Number of Monte Carlo samples.
            use_mcdropout: If True, keeps dropout active during inference.
        """
        self.model.eval()
        
        if use_mcdropout:
            # Force dropout layers to stay active
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        else:
            n_mc = 1 # Just one forward pass if not using MC Dropout

        X_scaled = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32).to(self.device)
        all_samples = np.zeros((n_mc, X.shape[0], len(self.quantiles)))
        with torch.no_grad():
            for i in range(n_mc):
                pred = self.model(X_scaled).cpu().numpy()
                
                # Correct crossing for THIS specific sample
                if self.undo_crossing:
                    pred = self._apply_crossing_fix(pred)
                
                # Inverse scale immediately so the values are in original units
                all_samples[i] = self.scaler_y.inverse_transform(pred)
        
        all_samples = np.transpose(all_samples, (1, 0, 2))
        
        q_low = all_samples[:, :, 0]   # (N, n_mc)
        q_high = all_samples[:, :, 1]  # (N, n_mc)

        self.model.train()

        return q_low, q_high

    def _apply_crossing_fix(self, y_pred):
        """
        Fixes quantile crossing using sorting (Isotone Regression).
        Ensures q_low <= q_high without destroying the predictive width.
        """
        return np.sort(y_pred, axis=1)

class QuantileRegressionNNEnsemble(BaseEstimator):
    def __init__(
        self,
        input_size,
        n_models = 15,
        alpha = 0.1,
        dropout=0,
        hidden_layers=[64, 64],
        use_gpu=True,
        undo_crossing=True,
    ):
        self.alpha = alpha
        self.quantiles = [alpha/2, 1-alpha/2]
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = len(self.quantiles)
        self.hidden_layers = hidden_layers
        self.p_dropout = dropout
        self.use_gpu = use_gpu
        self.undo_crossing = undo_crossing
        self.n_models = n_models

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    @staticmethod
    def quantile_loss(y_pred, y_true, quantiles):
        loss = 0
        for i, q in enumerate(quantiles):
            error = y_true - y_pred[:, i:i+1]
            loss += torch.max((q - 1) * error, q * error).mean()
        return loss
    
    def fit(
            self, 
            X, 
            y,
            weight_decay=0,
            scheduler_step=20,
            scheduler_gamma=0.99,
            epochs=500, 
            lr=1e-3, 
            batch_size=32, 
            patience=30, 
            verbose=1, 
            split_random_state=42,
            fit_random_state=1250,
            bootstrap=True,
    ):
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=split_random_state)
        
        x_train = torch.tensor(self.scaler_x.fit_transform(x_train), dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.scaler_y.fit_transform(y_train.reshape(-1, 1)), dtype=torch.float32).to(self.device)
        x_val = torch.tensor(self.scaler_x.transform(x_val), dtype=torch.float32).to(self.device)
        y_val = torch.tensor(self.scaler_y.transform(y_val.reshape(-1, 1)), dtype=torch.float32).to(self.device)
        self.models = []

        for i in tqdm(range(self.n_models), desc="Fitting Quantile Regression Ensemble Models", disable=(verbose==0)):
            torch.manual_seed(fit_random_state + i)
            torch.cuda.manual_seed(fit_random_state + i)

            n_train = x_train.shape[0]
            if bootstrap:
                # 1. Create a Bootstrap Sampler
                # We use uniform weights but allow replacement
                sampler = WeightedRandomSampler(
                    weights=torch.ones(n_train),
                    num_samples=n_train,
                    replacement=True
                )
                # Note: shuffle must be False when using a sampler
                train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, sampler=sampler)
            else:
                train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


            model = QuantileRegressionNet(
            input_size=self.input_size, 
            output_size=self.output_size, 
            hidden_layers=self.hidden_layers, 
            p_dropout=self.p_dropout
           ).to(self.device)
            
            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
            
            model = self.fit_one_model(
                model,
                train_loader, 
                x_val, 
                y_val, 
                patience, 
                epochs, 
                lr,
                weight_decay, 
                scheduler_step, 
                scheduler_gamma,
                )
            
            self.models.append(model)
        
        return self

    def fit_one_model(
            self,
            model,
            train_loader, 
            x_val,
            y_val,
            patience, 
            epochs, 
            lr, 
            weight_decay, 
            scheduler_step, 
            scheduler_gamma,
    ):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_step, 
            gamma=scheduler_gamma
        )

        best_val_loss = float('inf')
        best_model_state = None
        counter = 0

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = self.quantile_loss(pred, by, self.quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 3. Early Stopping Check
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = self.quantile_loss(val_pred, y_val, self.quantiles).item()

            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model
    
    def predict_ensemble(self, X_test):
        X_scaled = torch.tensor(self.scaler_x.transform(X_test), dtype=torch.float32).to(self.device)
        all_samples = np.zeros((len(self.models), X_test.shape[0], len(self.quantiles)))
        i = 0
        for model in self.models: 
            model.eval()

            with torch.no_grad():
                pred = model(X_scaled).cpu().numpy()

                # Correct crossing for THIS specific sample
                if self.undo_crossing:
                    pred = self._apply_crossing_fix(pred)
                
                # Inverse scale immediately so the values are in original units
                all_samples[i] = self.scaler_y.inverse_transform(pred)
            model.train()
            i += 1
        
        all_samples = np.transpose(all_samples, (1, 0, 2))
        q_low = all_samples[:, :, 0]
        q_high = all_samples[:, :, 1]

        return q_low, q_high
    
    def _apply_crossing_fix(self, y_pred):
        """
        Fixes quantile crossing using sorting (Isotone Regression).
        Ensures q_low <= q_high without destroying the predictive width.
        """
        return np.sort(y_pred, axis=1)

      
############### Classification using MC Dropout ###############
# Neural Network Classifier Base Architecture
class NN_base(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_layers, dropout_rate=0.4):
        """
        Flexible NN architecture for classification

        Input: (i) input_shape (int): Input dimension.
               (ii) num_classes (int): Number of output classes.
               (iii) hidden_layers (list): List containing the number of neurons per hidden layer.
               (iv) dropout_rate (float): Dropout rate applied to each layer. Default is 0.4.
        """
        super(NN_base, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Creating hidden layers dynamically
        prev_units = input_shape
        for units in hidden_layers:
            self.layers.append(nn.Linear(prev_units, units))
            self.batch_norms.append(nn.BatchNorm1d(units))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_units = units

        self.fc_out = nn.Linear(prev_units, num_classes)

    def forward(self, x):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = F.relu(layer(x))
            x = bn(x)
            x = dropout(x)
        x = self.fc_out(x)
        return x

# MC Dropout Classifier class
class MC_classifier(BaseEstimator):
    """
    Monte Carlo Dropout Classifier.
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        hidden_layers=[64],
        dropout_rate=0.4,
    ):
        """
        Input: (i) input_shape: Input dimension.
               (ii) num_classes: Number of output classes.
               (iii) hidden_layers: List containing the number of neurons in each hidden layer. The length of the list determines the amount of hidden layers in the model.
               (iv) dropout_rate: Dropout Rate for each hidden layer. Default is 0.4.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = NN_base(
            self.input_shape, self.num_classes, self.hidden_layers, self.dropout_rate
        )

    def fit(
        self,
        X,
        y,
        proportion_train=0.7,
        epochs=500,
        lr=0.001,
        gamma=0.99,
        batch_size=32,
        step_size=5,
        weight_decay=0,
        verbose=0,
        patience=30,
        scale=False,
        random_seed_split=0,
        random_seed_fit=1250,
    ):
        """
        Fit MC Dropout Classifier.

        Input: (i) X (np.ndarray or torch.Tensor): Training input data.
               (ii) y (np.ndarray or torch.Tensor): Training target data.
               (iii) proportion_train (float): Proportion of data to be used for training (the rest for validation). Default is 0.7.
               (iv) epochs (int): Number of epochs for training. Default is 500.
               (v) lr (float): Learning rate for the optimizer. Default is 0.001.
               (vi) gamma (float): Gamma value for scheduler. Default is 0.99
               (vii) batch_size (int): Batch size. Default is 32.
               (viii) step_size (int): Step size for scheduler. Default is 5.
               (ix) weight_decay (float): Optimizer weight decay parameter. Default is 0.
               (x) verbose (int): Verbosity level (0, 1, or 2). If set to 0, does not print anything, if 1, prints the average loss of each epoch and if 2, prints the model learning curve at end of fitting.
               (xi) patience (int): Number of epochs with no improvement to trigger early stopping. Default is 30.
               (xii) scale (bool): Whether to scale or not the data. Default is False.
               (xiii) random_seed_split (int): Random seed fixed to perform data splitting. Default is 0.
               (xiv) random_seed_fit (int): Random seed fixed to model fitting. Default is 1250.

        Output: (i) fitted MC_classifier object
        """
        self.optimizer = optim.Adamax(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        # Splitting data into train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=1 - proportion_train, random_state=random_seed_split
        )

        # checking if scaling is needed
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_val = self.scaler.transform(x_val)
            self.scale = True
        else:
            self.scale = False

        # checking if is an instance of numpy
        if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
            x_train, x_val = (
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(x_val, dtype=torch.float32),
            )
            y_train, y_val = (
                torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
                torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            )

        # Training and validation
        train_dataset = TensorDataset(
            x_train.clone().detach().float(),
            (
                y_train.clone().detach().float()
                if isinstance(y_train, torch.Tensor)
                else torch.tensor(y_train, dtype=torch.float32)
            ),
        )
        val_dataset = TensorDataset(
            x_val.clone().detach().float(),
            (
                y_val.clone().detach().float()
                if isinstance(y_val, torch.Tensor)
                else torch.tensor(y_val, dtype=torch.float32)
            ),
        )

        # Setting batch size
        batch_size_train = int(proportion_train * batch_size)
        batch_size_val = int((1 - proportion_train) * batch_size)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_train, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

        losses_train = []
        losses_val = []

        # early stopping
        best_val_loss = float("inf")
        counter = 0

        torch.manual_seed(random_seed_fit)
        torch.cuda.manual_seed(random_seed_fit)
        # Training loop
        for epoch in tqdm(range(epochs), desc="Fitting MC Dropout Classifier"):
            self.model.train()
            train_loss_epoch = 0

            # Looping through batches
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output_train = self.model(x_batch)  # Network output
                loss_train = F.cross_entropy(output_train, y_batch.long().view(-1))
                loss_train.backward()
                self.optimizer.step()
                train_loss_epoch += loss_train.item()

            # Computing validation loss
            self.model.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output_val = self.model(x_batch)  # Network output
                    loss_val = F.cross_entropy(output_val, y_batch.long().view(-1))
                    val_loss_epoch += loss_val.item()

            # average loss by epoch
            train_loss_epoch /= len(train_loader)
            val_loss_epoch /= len(val_loader)
            losses_train.append(train_loss_epoch)
            losses_val.append(val_loss_epoch)

            self.scheduler.step()

            if verbose == 1:
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}"
                )

            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping in epoch {epoch} with best validation loss: {best_val_loss:.4f}"
                    )
                    break

        if verbose == 2:
            fig, ax = plt.subplots()
            ax.set_title("Training and Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            epochs_completed = len(losses_train)
            ax.set_xlim(0, epochs_completed)

            ax.plot(
                range(epochs_completed), losses_train, label="Train Loss", color="blue"
            )
            ax.plot(
                range(epochs_completed),
                losses_val,
                label="Validation Loss",
                color="green",
                linestyle="--",
            )
            plt.legend(loc="upper right")
            plt.show()

        return self

    def predict_mc_dropout(self, x, num_samples=100, return_mean=False):
        """
        Make predictions with MC Dropout.

        Input:
            (i) x: Input data.
            (ii) num_samples: Number of Monte Carlo samples. Default is 100.
            (iii) return_mean: Whether to return the mean of the predictions. Default is False.

        Output:
            (i) Tuple containing the stacked tensors of the predictions or their means if return_mean is True.
        """
        if isinstance(x, np.ndarray):
            if self.scale:
                x = self.scaler.transform(x)
            x = torch.tensor(x, dtype=torch.float32)

        self.model.eval()
        self.model.train()

        predictions = []

        for _ in range(num_samples):
            with torch.no_grad():
                pred = F.softmax(self.model(x), dim=1)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        if return_mean:
            mean_predictions = torch.mean(predictions, dim=0)
            return mean_predictions

        return predictions

    def predict_pmf(self, X_test, num_samples=100, random_seed=45):
        probs = (
            self.predict_mc_dropout(
                X_test,
                num_samples,
                return_mean=True,
            )
            .detach()
            .numpy()
        )

        return probs


############### basic and heteroscedastic GP models ###############
# Using GPyjax
class GP_model(BaseEstimator):
    """
    Gaussian Process model.

    Gaussian Process regressor using GPyjax implementation.
    """

    def __init__(
            self,
            kernel= None,
            kernel_noise= None,
            normalize_y=True,
            heteroscedastic=False,
            variational=False,
    ):
        """
        Input:
        (i) kernel: Kernel specifying the covariance structure of the GP. Default is None.
        (ii) normalize_y: Whether to normalize the target variable. Default is True.
        (iii) heteroscedastic: Whether the model is heteroscedastic. Default is False.

        Output:
        (i) Initialized GP_model object.
        """
        if kernel is None:
            self.kernel = (gpx.kernels.RationalQuadratic() + 
                           gpx.kernels.Matern52()
                           )
        else:
            self.kernel = kernel
        
        if kernel_noise is None:
            self.kernel_noise = gpx.kernels.RBF()
        else:
            self.kernel_noise = kernel_noise

        self.normalize_y = normalize_y
        self.heteroscedastic = heteroscedastic
        self.variational = variational

    def fit(
            self, 
            X, 
            y, 
            scale=False, 
            random_state=0,
            activation_sigma = "softplus",
            ):
        """
        Fit GP model.

        Input:
        (i) X (np.ndarray): Training input data.
        (ii) y (np.ndarray): Training target data.
        (iii) scale (bool): Whether to scale or not the data. Default is False.
        (iv) random_state (int): Random seed fixed to perform model fitting. Default is 0.
        (v) activation_sigma (str): Activation function used for the noise process. Default is "softplus".

        Output:
        (i) fitted GP_model object.
        """
        if X.shape[0] > 5000 and not self.heteroscedastic and not self.variational:
            print("Changing to variational because of the large amount of data.")
            self.variational = True

        # scaling part
        if scale:
            self.scaler_X = StandardScaler()
            X = self.scaler_X.fit_transform(X)
            self.scale_x = True
        else:
            self.scale_x = False
        if self.normalize_y:
            self.y_scaler = StandardScaler()
            y = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        key = jr.PRNGKey(random_state)
        x_jp, y_jp = jnp.array(X, dtype=jnp.float64), jnp.array(y.reshape(-1, 1), dtype=jnp.float64)
        self.train_data = gpx.Dataset(X=x_jp, y=y_jp)
        
        # usual GP without heteroscedasticity
        if not self.heteroscedastic:
            meanf = gpx.mean_functions.Zero()
            prior = gpx.gps.Prior(mean_function=meanf, kernel=self.kernel)
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.train_data.n)
            posterior_model = prior * likelihood

            if self.variational:
                num_inducing = min(25, self.train_data.n) 
                kmeans = KMeans(n_clusters=num_inducing, n_init='auto', random_state=random_state)
                z_init = jnp.array(kmeans.fit(X).cluster_centers_)

                self.q = gpx.variational_families.VariationalGaussian(
                posterior=posterior_model,
                inducing_inputs=z_init
                )
                key, subkey = jr.split(key)
                optimiser = ox.chain(
                    ox.clip_by_global_norm(1.0), # Prevents massive gradient updates
                    ox.adam(learning_rate=0.005),
                    ox.zero_nans()
                )

                objective = lambda model, data: -gpx.objectives.elbo(model, data)
                jitted_objective = jit(objective)

                # optimizing hyperparameters
                self.opt_posterior, history = gpx.fit(
                    model=self.q,
                    objective=jitted_objective,
                    train_data=self.train_data,
                    optim=optimiser,
                    num_iters=1000,
                    key=subkey,
                    verbose = False,
                )
            else:
                self.opt_posterior, history = gpx.fit_scipy(
                    model=posterior_model,
                    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
                    train_data=self.train_data,
                    trainable=gpx.parameters.Parameter,
                )

        elif self.heteroscedastic:
            print("Fitting heteroscedastic model.")
            if activation_sigma == "lognormal":
                self.activation_sigma = "lognormal"
                noise_transform = LogNormalTransform()
            elif activation_sigma == "softplus":
                self.activation_sigma = "softplus"
                noise_transform = SoftplusTransform()

            signal_prior = gpx.gps.Prior(
                mean_function=gpx.mean_functions.Zero(),
                kernel=self.kernel,
            )
            noise_prior = gpx.gps.Prior(
                mean_function=gpx.mean_functions.Zero(),
                kernel=self.kernel_noise,
            )
            likelihood = HeteroscedasticGaussian(
                num_datapoints=self.train_data.n,
                noise_prior=noise_prior,
                noise_transform=noise_transform,
            )
            posterior = signal_prior * likelihood

            # inducing points
            num_inducing = min(25, self.train_data.n) 
            kmeans = KMeans(n_clusters=num_inducing, n_init='auto', random_state=random_state)
            z_init = jnp.array(kmeans.fit(X).cluster_centers_)

            self.q = HeteroscedasticVariationalFamily(
                posterior=posterior,
                inducing_inputs=z_init,
                inducing_inputs_g=z_init,
            )

            key, subkey = jr.split(key)
            optimiser = ox.chain(
                ox.clip_by_global_norm(1.0), # Prevents massive gradient updates
                ox.adam(learning_rate=0.005),
                ox.zero_nans()
            )

            # optimizing hyperparameters
            objective = lambda model, data: -gpx.objectives.heteroscedastic_elbo(model, data)
            jitted_objective = jit(objective)

            self.opt_posterior, history = gpx.fit(
                model=self.q,
                objective=jitted_objective,
                train_data=self.train_data,
                optim=optimiser,
                num_iters=1000,
                key=subkey,
                verbose = False,
            )
            print("fitted heteroscedastic GP")
        return self
    
    def predict_quantiles(
            self, 
            X_test, 
            quantiles=[0.05, 0.95], 
            key = jr.PRNGKey(0),
            n_MC = 1000,):
        """
        Predict quantiles for the test data.

        Input:
        (i) X_test (np.ndarray): Test input data.

        Output:
        (i) Tuple containing the several lower and upper quantiles for each test sample.
        """
        if self.scale_x:
            X_test = self.scaler_X.transform(X_test)

        x_jp = jnp.array(X_test, dtype=jnp.float64)

        if not self.heteroscedastic:
            key, subkey = jr.split(key)
            if not self.variational:
                latent_dist = self.opt_posterior.predict(x_jp, train_data=self.train_data)
                sample_functions = latent_dist.sample(subkey, sample_shape=(n_MC,))
                sigma = self.opt_posterior.likelihood.obs_stddev.get_value()
            else:
                latent_dist = self.opt_posterior.predict(x_jp)
                sample_functions = latent_dist.sample(subkey, sample_shape=(n_MC,))
                sigma = self.opt_posterior.posterior.likelihood.obs_stddev.get_value()

            z_scores = norm.ppf(quantiles)
            quantile_functions = sample_functions[..., None] + z_scores * sigma
            q_l = quantile_functions[:, :, 0]
            q_u = quantile_functions[:, :, 1]
        else:
            key, subkey_f = jr.split(key)
            # latent mean process
            f_dist, g_dist = self.opt_posterior.predict_latents(x_jp)

            f_samples = f_dist.sample(subkey_f, sample_shape=(n_MC,))

            # latent noise process
            key, subkey_g = jr.split(key)
            g_samples = g_dist.sample(subkey_g, sample_shape=(n_MC,))

            if self.activation_sigma == "lognormal":
                sigma_samples = jnp.exp(g_samples / 2.0)
            elif self.activation_sigma == "softplus":
                sigma_samples = jnp.sqrt(jnp.log(1.0 + jnp.exp(g_samples)))
            z_scores = norm.ppf(jnp.array(quantiles))

            quantile_functions = (
                f_samples[..., None] + 
                sigma_samples[..., None] * z_scores[None, None, :]
            )

            q_l = quantile_functions[:, :, 0]
            q_u = quantile_functions[:, :, 1]

        if self.normalize_y:
            mean = jnp.array(self.y_scaler.mean_)
            scale = jnp.array(self.y_scaler.scale_)
            q_l = (q_l * scale) + mean
            q_u = (q_u * scale) + mean
            # original_shape = q_l.shape
            # q_l = self.y_scaler.inverse_transform(q_l.flatten().reshape(-1, 1)).reshape(original_shape)
            # q_u = self.y_scaler.inverse_transform(q_u.flatten().reshape(-1, 1)).reshape(original_shape)

        return q_l, q_u

############### BART model ###############
class BART_model(BaseEstimator):
    """
    Bayesian Additive Regression Trees model.
    """

    def __init__(
        self,
        m=100,
        type="normal",
        var="heteroscedastic",
        alpha_bart=0.95,
        beta_bart=2,
        response="constant",
        split_prior=None,
        separate_trees=False,
        n_cores=3,
        n_chains=4,
        normalize_y=True,
        progressbar=False,
    ):
        """
        Input:
        (i) m (int, optional): Number of regression trees. Default is 100.
        (ii) type (str, optional): Type of model. Default is "normal".
        (iii) var (str, optional): Type of variance. Default is "heteroscedastic".
        (iv) alpha (float, optional): Alpha parameter for the model. Default is 0.95.
        (v) beta (float, optional): Beta parameter for the model. Default is 2.
        (vi) response (str, optional): Type of response. Default is 'constant'.
        (vii) split_prior (optional): Prior for the split. Default is None.
        (viii) separate_trees (bool, optional): Whether to use separate trees. Default is False.
        (ix) n_cores (int, optional): Number of cores to use. Default is 3.
        (x) n_chains (int, optional): Number of chains for MCMC. Default is 4.
        (xi) normalize_y (bool, optional): Whether to normalize the target variable. Default is False.

        Output:
        (i) Initialized BART_model object.
        """
        self.m = m
        self.type = type
        self.var = var
        self.alpha_bart = alpha_bart
        self.beta_bart = beta_bart
        self.response = response
        self.split_prior = split_prior
        self.separate_trees = separate_trees
        self.n_cores = n_cores
        self.n_chains = n_chains
        self.normalize_y = normalize_y
        if self.normalize_y:
            self.scaler_y = StandardScaler()
        self.progressbar = progressbar

    def fit(self, X, y, n_sample=2000, random_seed=1250):
        """
        Fit the model to the provided data.

        Input:
        (i) X (numpy.ndarray): The input features for the model.
        (ii) y (numpy.ndarray): The target values for the model.
        (iii) n_sample (int, optional): The number of samples to draw in the MCMC process (default is 2000).
        (iv) random_seed (int, optional): The random seed for reproducibility (default is 1250).

        Output:
        (i) self (object): Returns the instance itself.

        Notes:
        This method fits a Bayesian Additive Regression Trees (BART) model to the data.
        The model can handle both homoscedastic and heteroscedastic variance structures.
        Depending on the `type` and `var` attributes of the instance, different models
        are constructed and fitted using PyMC3.

        Attributes:
        (i) model_bart (pm.Model): The fitted BART model.
        (ii) mc_sample (pm.backends.base.MultiTrace): The MCMC samples obtained from fitting the model.
        """

        n_obs = X.shape[0]

        # changing splitting styles according to variables being binary or not
        binary_columns = [i for i in range(X.shape[1]) if np.unique(X[:, i]).size == 2]

        if len(binary_columns) > 0:
            X = X.astype(float)
            self.type_X = True
        else:
            self.type_X = False

        # making splitting list
        split_types = np.repeat(ContinuousSplitRule, repeats=X.shape[1])
        split_types[binary_columns] = OneHotSplitRule
        split_types = split_types.tolist()

        if self.normalize_y:
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # fitting data to BART
        if self.type == "normal":
            if self.var == "heteroscedastic":
                with pm.Model() as model_bart:
                    self.X_data = pm.Data("data_X", X)
                    w = pmb.BART(       
                        "w",
                        self.X_data,
                        y,
                        m=self.m,
                        shape=(2, n_obs),
                        alpha=self.alpha_bart,
                        beta=self.beta_bart,
                        response=self.response,
                        split_prior=self.split_prior,
                        split_rules=split_types,
                        separate_trees=self.separate_trees,
                    )
                    pm.Normal(
                        "y_pred",
                        w[0],
                        np.exp(w[1]),
                        observed=y,
                        shape=self.X_data.shape[0],
                    )

                    # running MCMC in the training sample
                    self.mc_sample = pm.sample(
                        n_sample,
                        chains=self.n_chains,
                        random_seed=random_seed,
                        cores=self.n_cores,
                        progressbar=self.progressbar,
                    )
            elif self.var == "homoscedastic":
                with pm.Model() as model_bart:
                    self.X_data = pm.Data("data_X", X)

                    # variance
                    sigma = pm.HalfNormal("sigma", 5)

                    # mu
                    mu = pmb.BART(
                        "mu,",
                        self.X_data,
                        y,
                        m=self.m,
                        alpha=self.alpha_bart,
                        beta=self.beta_bart,
                        response=self.response,
                        split_prior=self.split_prior,
                        split_rules=split_types,
                        separate_trees=self.separate_trees,
                    )

                    pm.Normal(
                        "y_pred",
                        mu,
                        sigma,
                        observed=y,
                        shape=self.X_data.shape[0],
                    )

                    # running MCMC in the training sample
                    self.mc_sample = pm.sample(
                        n_sample,
                        chains=self.n_chains,
                        random_seed=random_seed,
                        cores=self.n_cores,
                        progressbar=self.progressbar,
                    )
            self.model_bart = model_bart
        elif self.type == "gamma":
            if self.var == "heteroscedastic":
                with pm.Model() as model_bart:
                    self.X_data = pm.Data(
                        "data_X",
                        X,
                    )
                    Y = y

                    w = pmb.BART(
                        "w",
                        self.X_data,
                        np.log(Y),
                        m=100,
                        size=2,
                        alpha=self.alpha_bart,
                        beta=self.beta_bart,
                        response=self.response,
                        split_prior=self.split_prior,
                        split_rules=split_types,
                        separate_trees=self.separate_trees,
                    )

                    pm.Gamma(
                        "y_pred",
                        mu=pm.math.exp(w[0]),
                        sigma=pm.math.exp(w[1]),
                        shape=self.X_data.shape[0],
                        observed=Y,
                    )

                    self.mc_sample = pm.sample(
                        n_sample,
                        chains=self.n_chains,
                        random_seed=random_seed,
                        cores=self.n_cores,
                        progressbar=self.progressbar,
                    )
            self.model_bart = model_bart

        # categorical regression for cases like APS
        elif self.type == "categorical":
            # treating categorical data accordingly
            # converting y to integer
            if y.dtype != np.int64:
                y = y.astype(np.int64)
            n_cat = np.unique(y).shape[0]

            # fitting model
            with pm.Model() as model_bart:
                self.X_data = pm.Data("data_X", X)
                mu = pmb.BART(
                    "mu", self.X_data, y, m=self.m, shape=(n_cat, self.X_data.shape[0])
                )

                theta = pm.Deterministic("theta_obs", pm.math.softmax(mu, axis=0))

                # Likelihood (sampling distribution) of observations
                y_obs = pm.Categorical("y_obs", p=theta.T, observed=y)

                # running MCMC in the training sample
                self.mc_sample = pm.sample(
                    n_sample,
                    chains=self.n_chains,
                    random_seed=random_seed,
                    cores=self.n_cores,
                    progressbar=self.progressbar,
                )
            self.model_bart = model_bart
        return self

    def predict_pmf(self, X_test, random_seed=0):
        """
        Predict the probability mass function (PMF) for the given test data.

        Input:
        (i) X_test (array-like): The input features for the test data.
        (ii) y_test (array-like): The true target values for the test data.
        (iii) random_seed (int, optional): The random seed for reproducibility of the posterior predictive sampling. Default is 0.

        Output:
        (i) pmf_array (numpy.ndarray): An array containing the PMF values for the test data, with rows indicating the samples and columns indicating the classes.
        """
        if self.type_X:
            X_test = X_test.astype(float)

        with self.model_bart:
            self.X_data.set_value(X_test)
            posterior_predictive_test = pm.sample_posterior_predictive(
                trace=self.mc_sample,
                random_seed=random_seed,
                var_names=["theta_obs"],
                predictions=True,
                progressbar=True,
            )

        pred_sample = (
            az.extract(
                posterior_predictive_test,
                group="predictions",
                var_names=["theta_obs"],
            )
            .mean("sample")
            .T.to_numpy()
        )

        return pred_sample

    def predict_cdf(self, X_test, y_test, random_seed=0):
        """
        Predict the cumulative distribution function (CDF) for the given test data.

        Input:
        (i) X_test (array-like): The input features for the test data.
        (ii) y_test (array-like): The true target values for the test data.
        (iii) random_seed (int, optional): The random seed for reproducibility of the posterior predictive sampling. Default is 0.

        Output:
        (i) cdf_array (numpy.ndarray): An array containing the CDF values for the test data.
        """
        if self.type_X:
            X_test = X_test.astype(float)
        if self.normalize_y:
            y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        with self.model_bart:
            self.X_data.set_value(X_test)
            posterior_predictive_test = pm.sample_posterior_predictive(
                trace=self.mc_sample,
                random_seed=random_seed,
                var_names=["y_pred"],
                predictions=True,
                progressbar=self.progressbar,
            )

        pred_sample = az.extract(
            posterior_predictive_test,
            group="predictions",
            var_names=["y_pred"],
        ).T.to_numpy()

        # subtracting pred_sample from y_test
        cdf_array = np.mean((np.subtract(pred_sample, y_test) <= 0) + 0, axis=0)
        return cdf_array

    def predict_cutoff(self, X_test, t, random_seed=0):
        """
        Predict cutoff values for the given test data.

        Input:
        (i) X_test (array-like): Test data for which the cutoff values are to be predicted.
        (ii) t (float or array-like): Quantile(s) to compute, which should be between 0 and 1 inclusive.
        (iii) random_seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is 0.

        Output:
        (i) cutoffs (array-like): Predicted cutoff values for the test data.
        """
        if self.type_X:
            X_test = X_test.astype(float)

        with self.model_bart:
            self.X_data.set_value(X_test)
            posterior_predictive_test = pm.sample_posterior_predictive(
                trace=self.mc_sample,
                random_seed=random_seed,
                var_names=["y_pred"],
                predictions=True,
                progressbar=self.progressbar,
            )

        pred_sample = az.extract(
            posterior_predictive_test,
            group="predictions",
            var_names=["y_pred"],
        ).T.to_numpy()

        cutoffs = np.quantile(pred_sample, q=t, axis=0)

        # inverse transforming
        if self.normalize_y:
            cutoffs = self.scaler_y.inverse_transform(cutoffs.reshape(-1, 1)).flatten()
        return cutoffs
    
    def sample_quantiles_from_posterior(self, X_test, quantile_levels, random_seed=0):
        """
        Generates quantile samples Q_alpha(Y|X, theta) from the BART posterior.
        Each sample represents a quantile from a different function/noise draw (theta) from the posterior.

        Input:
        (i) X_test (array-like): Test data for which quantiles are to be predicted.
        (ii) quantile_levels (list or np.ndarray): The quantile levels (e.g., [0.05, 0.95]).
        (iii) random_seed (int, optional): Seed for reproducibility. Default is 0.

        Output:
        (i) np.ndarray: Quantile samples Q_alpha(Y|X), shape (n_test_samples, n_MCMC_samples, n_quantiles).
        """
        quantile_levels = np.asarray(quantile_levels)

        if self.type_X:
            X_test = X_test.astype(float)
        
        # 1. Set new test data in the PyMC model
        with self.model_bart:
            self.X_data.set_value(X_test)
            
            if self.type == "normal":
                if self.var == "heteroscedastic":
                    # Sample 'w' which contains both mu (w[0]) and log(sigma) (w[1])
                    var_names = ["w"]
                elif self.var == "homoscedastic":
                    # Sample the mean function 'mu' and the scalar noise 'sigma'
                    var_names = ["mu", "sigma"]
                else:
                    raise ValueError(f"Unknown variance type: {self.var}")
            else:
                 # TODO: Implement also for gamma
                 raise NotImplementedError(f"Quantile sampling is not implemented for BART type: {self.type}")

            # Sample the model parameters from the posterior (theta ~ P(theta|D))
            posterior_samples = pm.sample_posterior_predictive(
                trace=self.mc_sample,
                var_names=var_names,
                predictions=True,
                random_seed=random_seed,
                progressbar=self.progressbar,
            )

        # Calculate Z-scores for the required quantile levels
        Z_alphas = norm.ppf(quantile_levels) # shape: (n_quantiles,)
        n_quantiles = Z_alphas.shape[0]
        n_test_samples = X_test.shape[0]

        if self.type == "normal":
            if self.var == "heteroscedastic":
                # Extract 'w' samples (n_samples, n_test_samples, 2)
                # The PyMC dimension order is usually (chain * draw, *data_dims)
                w_samples = az.extract(
                    posterior_samples,
                    group="predictions",
                    var_names=["w"],
                ).to_numpy()
                
                self.w_samples = w_samples
                print(w_samples.shape)
                # mu_samples shape: (n_samples, n_test_samples)
                mu_samples = w_samples[0, :, :]
                # sigma_samples shape: (n_samples, n_test_samples). Note: PyMC uses exp(w[1]) for sigma.
                sigma_samples = np.exp(w_samples[1, :, :])
                
            elif self.var == "homoscedastic":
                # mu_samples shape: (n_samples, n_test_samples)
                mu_samples = az.extract(
                    posterior_samples,
                    group="predictions",
                    var_names=["mu"],
                ).to_numpy().T
                
                # sigma is a scalar parameter, sampled (n_samples,)
                sigma_samples = az.extract(
                    posterior_samples,
                    group="predictions",
                    var_names=["sigma"],
                ).to_numpy()
                
                # Broadcast scalar sigma to (n_samples, n_test_samples) for element-wise operation
                sigma_samples = np.repeat(sigma_samples[:, None], n_test_samples, axis=1)
            
            # Calculate quantiles: Q_alpha = mu + Z_alpha * sigma
            quantile_samples_temp = (
                mu_samples[:, :, None] + sigma_samples[:, :, None] * Z_alphas[None, None, :]
            )
            
            # (n_test_samples, n_MCMC_samples, n_quantiles)
            quantile_samples = np.transpose(quantile_samples_temp, (1, 0, 2))

        # Inverse scaling
        if self.normalize_y:
            original_shape = quantile_samples.shape
            samples_2d = quantile_samples.reshape(-1, n_quantiles)
            
            samples_inverse_scaled = self.scaler_y.inverse_transform(samples_2d)
            
            quantile_samples = samples_inverse_scaled.reshape(original_shape)
        
        return quantile_samples

    def predict(self, X_test, quantiles, random_seed=0):
        """
        Return predictive quantiles for each test sample.

        Output is a matrix of shape (n_test_samples, n_quantiles) where each row
        corresponds to a test sample and each column to a requested quantile.

        This method uses the posterior predictive samples of the PyMC BART model
        for the observed variable `y_pred` and computes empirical quantiles across
        posterior predictive draws.
        """
        if self.type_X:
            X_test = X_test.astype(float)

        if self.type == "normal":
            var_names = ["w"] if self.var == "heteroscedastic" else ["mu", "sigma"]
        elif self.type == "gamma":
            var_names = ["w"] 
        else:
            raise NotImplementedError(f"Quantile estimation not ready for {self.type}")

        # Get Posterior Samples of the parameters
        with self.model_bart:
            self.X_data.set_value(X_test)
            post_samples = pm.sample_posterior_predictive(
                trace=self.mc_sample,
                var_names=var_names,
                predictions=True,
                random_seed=random_seed,
                progressbar=self.progressbar,
            )

        # Calculate Gaussian Quantiles
        z_scores = norm.ppf(quantiles)
        
        extracted = az.extract(post_samples, group="predictions", var_names=var_names)
        n_test = X_test.shape[0]
        quantile_results = np.zeros((n_test, len(quantiles)))

        if self.type == "normal":
            if self.var == "heteroscedastic":
                w_vals = extracted.to_numpy()
                mu_draws = w_vals[0]
                sigma_draws = np.exp(w_vals[1]) 
            else:
                mu_draws = extracted["mu"].to_numpy()
                sigma_draws = extracted["sigma"].to_numpy()

            for i, tau in enumerate(quantiles):
                q_draws = mu_draws + (z_scores[i] * sigma_draws)
                quantile_results[:, i] = q_draws.mean(axis=1)

        elif self.type == "gamma":
            w_vals = extracted.to_numpy()
            mu_draws = np.exp(w_vals[0])
            sigma_draws = np.exp(w_vals[1])
            
            # For Gamma, we need to convert (mu, sigma) to (alpha, beta) or (shape, scale)
            shape_draws = (mu_draws / sigma_draws)**2
            scale_draws = (sigma_draws**2) / mu_draws

            for i, tau in enumerate(quantiles):
                q_draws = gamma.ppf(tau, a=shape_draws, scale=scale_draws)
                quantile_results[:, i] = q_draws.mean(axis=1)

        if self.normalize_y:
            quantile_results = self.scaler_y.inverse_transform(quantile_results)

        return quantile_results





