# Adapted from EPIC's repository:
from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator

from sklearn.model_selection import train_test_split
import scipy.stats as st
import torch
from abc import ABC, abstractmethod

# used torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import scipy.stats as st
from scipy.stats import norm
from scipy.stats import gamma
from sklearn.preprocessing import StandardScaler

from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
from tqdm import tqdm

############################## Score class and quantile score ##############################
# defining score basic class
class Scores(ABC):
    """
    Base class to build any conformity score of choosing.
    In this class, one can define any conformity score for any base model of interest, already fitted or not.
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, is_fitted, **kwargs):
        self.is_fitted = is_fitted
        if self.is_fitted:
            self.base_model = base_model
        elif base_model is not None:
            self.base_model = base_model(**kwargs)

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the base model to training data
        --------------------------------------------------------
        Input: (i)    X: Training feature matrix.
               (ii)   y: Training label vector.

        Output: Scores object
        """
        pass

    @abstractmethod
    def compute(self, X_calib, y_calib):
        """
        Compute the conformity score in the calibration set
        --------------------------------------------------------
        Input: (i)    X_calib: Calibration feature matrix
               (ii)   y_calib: Calibration label vector

        Output: Conformity score vector
        """
        pass

    @abstractmethod
    def predict(self, X_test, cutoff):
        """
        Compute prediction intervals specified cutoff(s).
        --------------------------------------------------------
        Input: (i)    X_test: Test feature matrix
               (ii)   cutoff: Cutoff vector

        Output: Prediction intervals for test sample.
        """
        pass

# quantile score
# need to specify only base-model
class QuantileScore(Scores):
    """
    Quantile conformity score
    --------------------------------------------------------
    Input: (i)    base_model: Point prediction model object with fit and predict methods.
           (ii)   is_fitted: Boolean indicating if the regression model is already fitted.
    """

    def fit(self, X, y):
        """
        Fit the quantilic regression model to training data
        --------------------------------------------------------
        Input: (i)    X: Training feature matrix.
               (ii)   y: Training label vector.

        Output: RegressionScore object
        """
        if not self.is_fitted:
            self.base_model.fit(X, y)
        return self

    def compute(self, X_calib, y_calib, ensemble=False):
        """
        Compute the conformity score in the calibration set
        --------------------------------------------------------
        Input: (i)    X_calib: Calibration feature matrix
               (ii)   y_calib: Calibration label vector

        Output: Conformity score vector
        """
        if ensemble:
            pred = self.base_model.predict(X_calib).T[:, np.array([0, 2])]
        else:
            pred = self.base_model.predict(X_calib)

        scores = np.column_stack((pred[:, 0] - y_calib, y_calib - pred[:, 1]))
        res = np.max(scores, axis=1)
        return res

    def predict(self, X_test, cutoff, ensemble=False):
        """
        Compute prediction intervals using each observation cutoffs.
        --------------------------------------------------------
        Input: (i)    X_test: Test feature matrix
               (ii)   cutoff: Cutoff vector

        Output: Prediction intervals for test sample.
        """
        if ensemble:
            quantiles = self.base_model.predict(X_test).T[:, np.array([0, 2])]
        else:
            quantiles = self.base_model.predict(X_test)

        pred = np.vstack((quantiles[:, 0] - cutoff, quantiles[:, 1] + cutoff)).T
        return pred


################################## MDN base model ##############################
#### Mixture density network models
# General Base Mixture Density Network architecture
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
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Creating hidden layers dinamically
        prev_units = input_shape
        for units in hidden_layers:
            self.layers.append(nn.Linear(prev_units, units))
            self.batch_norms.append(nn.BatchNorm1d(units))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_units = units

        self.fc_out = nn.Linear(prev_units, num_components * 3)

    def forward(self, x):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
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


# Mixture Density Network general model:
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

        # original_mode = self.model.training
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

        if return_mean:
            pi_mean = torch.mean(pi_predictions, dim=0)
            mu_mean = torch.mean(mu_predictions, dim=0)
            sigma_mean = torch.mean(sigma_predictions, dim=0)

            return pi_mean, mu_mean, sigma_mean

        return pi_predictions, mu_predictions, sigma_predictions

    def predict(self, X_test, y_test=None, return_params=False):
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
            return mean_test
        elif self.base_model_type == "quantile":
            alphas = [self.alpha / 2, 1 - (self.alpha / 2)]
            quantiles_test = self.mixture_quantile(alphas, pi, mu, sigma)
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
            (i) np.ndarray: Quantile matrix of shape (n_samples, len(alphas)).
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(42)

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

    def sample_from_mixture(self, pi, mu, sigma, rng=0, N=1):
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
        if isinstance(rng, int):
            rng = np.random.default_rng(42)

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
        n_samples, _ = samples.shape
        n_probs = len(probs)

        quantiles = np.zeros((n_samples, n_probs))

        for j in range(n_samples):
            for k, prob in enumerate(probs):
                quantiles[j, k] = np.quantile(samples[j, :], prob)
        return torch.from_numpy(quantiles)

# EPIC with MDN for the predictive model
class EPIC_split(BaseEstimator):
    """
    General Epistemic Conformal Prediction class applied to any continuous or approximately continuous conformity scores.
    """

    def __init__(
        self,
        nc_score,
        base_model,
        alpha,
        is_fitted=False,
        base_model_type=None,
        **kwargs,
    ):
        """
        Input: (i) nc_score (Scores class): Conformity score of choosing. It
                can be specified by instantiating a conformal score class based on the Scores basic class.
               (ii) base_model (BaseEstimator class): Base model with fit and predict methods to be embedded in the conformity score class.
               (iii) alpha (float): Float between 0 and 1 specifying the miscoverage level of resulting prediction region.
               (iv) base_model_type (bool): Boolean indicating whether the base model ouputs quantiles or not. Default is False.
               (v) is_fitted (bool): Whether the base model is already fitted or not. Default is False.
               (vi) **kwargs: Additional keyword arguments passed to fit base_model.
        """
        self.base_model_type = base_model_type
        self.is_fitted = is_fitted
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(
                base_model, is_fitted=is_fitted, alpha=alpha, **kwargs
            )
        else:
            self.nc_score = nc_score(base_model, is_fitted=is_fitted, **kwargs)

        # checking if base model is fitted
        self.base_model = self.nc_score.base_model
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit base model embeded in the conformal score class to the training set.
        --------------------------------------------------------

        Input: (i)    X: Training numpy feature matrix
            (ii)   y: Training label array

        Output: LocartSplit object
        """
        self.nc_score.fit(X, y)
        return self

    def calib(
        self,
        X_calib,
        y_calib,
        random_seed=1250,
        scale=False,
        num_components=5,
        hidden_layers=[64, 64],
        dropout_rate=0.5,
        random_seed_split=0,
        random_seed_fit=45,
        split_calib=True,
        epistemic_test_thres=2000,
        N_samples_MC=500,
        normalize_y=False,
        log_y=False,
        type="gaussian",
        ensemble=False,
        **kwargs,
    ):
        """
        Calibrate conformity score using Predictive distribution.
        --------------------------------------------------------

        Input: (i) X_calib (np.ndarray): Calibration numpy feature matrix
               (ii) y_calib (np.ndarray): Calibration label array
               (iii) random_seed (int): Random seed used for data splitting for fit epistemic modeling of the conformal scores.
               (iv) epistemic_model (str): String indicating which predictive modeling approach to use. Options are: "BART", "GP_simple", "GP_variational", "MC_dropout". Default is "MC_dropout".
               (v) scale (bool): Whether to scale the X features or not. Default is False
               (vi) num_components (int): Number of components for the MDN model. Default is 5.
               (vii) hidden_layers (list): List with the amount of neural units per hidden layer for MDN model. Default is [64, 64].
               (viii) dropout_rate (float): Dropout Rate for MDN model. Default is 0.5.
               (ix) kernel (object): Kernel passed to GP simple model. Default is None (sklearn default).
               (x) normalize_y (bool): Whether to normalize the conformity score. Default is False.
               (xi) log_y (bool): Whether to use boxcox transformation on conformity score. Available only for non-negative conformal score. Default is False.
               (xii) type (str): Type of base distribution used in MDN or BART. Options are "normal", "gamma" and "gaussian". Default is "gaussian".
               (xiii) ensemble (bool): Whether the base model outputs three statistics (quantiles and median) or not. Default is False.
               (xiv) n_cores (int): Number of cores to use for parallel processing. Default is 6.
               (xv) progress (bool): Whether to print BART MCMC progress
               (xvi) **kwargs: Additional keyword arguments passed to epistemic model fitting step.
        Output: Vector of cutoffs.
        """
        # computing the scores
        scores = self.nc_score.compute(X_calib, y_calib, ensemble=ensemble)

        if X_calib.shape[0] >= epistemic_test_thres:
            epistemic_test_size = 1000 / X_calib.shape[0]
        else:
            epistemic_test_size = 0.3

        # splitting calibration into a training set and a cutoff set
        if split_calib:
            (
                X_calib_train,
                X_calib_test,
                scores_calib_train,
                scores_calib_test,
            ) = train_test_split(
                X_calib,
                scores,
                test_size=epistemic_test_size,
                random_state=random_seed,
            )
        else:
            (
                X_calib_train,
                X_calib_test,
                scores_calib_train,
                scores_calib_test,
            ) = (
                X_calib,
                X_calib,
                scores,
                scores,
            )

        # fitting epistemic model
        self.epistemic_obj = MDN_model(
            input_shape=X_calib.shape[1],
            num_components=num_components,
            dropout_rate=dropout_rate,
            hidden_layers=hidden_layers,
            normalize_y=normalize_y,
            log_y=log_y,
            type=type,
        )

        # fitting the epistemic model to data
        self.epistemic_obj.fit(
            X_calib_train,
            scores_calib_train,
            scale=scale,
            random_seed_fit=random_seed_fit,
            random_seed_split=random_seed_split,
            **kwargs,
        )

        # monte carlo dropout predictions for each parameter
        with torch.no_grad():
            pi_prime, mu_prime, sigma_prime = self.epistemic_obj.predict_mcdropout(
                X_calib_test, num_samples=N_samples_MC
            )
        # Computing new cumulative score s' or s_prime
        sample_s = self.epistemic_obj.mdn_generate(
            pi_prime, mu_prime, sigma_prime, random_seed
        )
        s_prime_calibration = self.epistemic_obj.mixture_cdf(
            sample_s, scores_calib_test
        )

        # converting to numpy
        s_prime_calibration_np = s_prime_calibration.flatten()
        n = s_prime_calibration_np.shape[0]

        self.t_cutoff = np.quantile(
            s_prime_calibration_np, np.ceil((n + 1) * (1 - self.alpha)) / n
        )

        return self.t_cutoff

    def predict(
        self,
        X_test,
        N_samples_MC=500,
        random_seed=45,
        ensemble=False,
    ):
        """
        Predict 1 - alpha prediction regions for each test sample using epistemic cutoffs.
        --------------------------------------------------------
        Input: (i) X_test (np.ndarray): Test numpy feature matrix
               (ii) N_samples_MC (int): Number of samples to simulate from MC dropout. Default is 500.
               (iii) random_seed (int): Random seed fixed to generate samples. Used in MC dropout and BART.
               (iv) ensemble (bool): Whether the base model outputs three statistics (quantiles and median) or not. Default is False.

        Output: Prediction regions for each test sample.
        """
        # predictions for mc_dropout
        with torch.no_grad():
            pi_test, mu_test, sigma_test = self.epistemic_obj.predict_mcdropout(
                X_test, num_samples=N_samples_MC
            )
        # computing t_inverse for obtaining region in
        # the original non conf score
        sample_test = self.epistemic_obj.mdn_generate(
            pi_test, mu_test, sigma_test, random_seed
        )

        # boxcox transformation for reg split
        if self.epistemic_obj.log_y:
            t_inverse_test_og = (
                self.epistemic_obj.mixture_ppf(sample_test, [self.t_cutoff])
                .numpy()
                .flatten()
            )
            # if also normalizing y
            if self.epistemic_obj.normalize_y:
                # first inverting normalization
                t_inverse_test = self.epistemic_obj.y_scaler.inverse_transform(
                    t_inverse_test_og.reshape(-1, 1)
                ).flatten()

                # then inverting box cox transformation
                t_inverse_test = inv_boxcox(
                    t_inverse_test,
                    self.epistemic_obj.lmbda,
                )
            else:
                # inverting box cox transformation
                t_inverse_test = inv_boxcox(
                    t_inverse_test_og,
                    self.epistemic_obj.lmbda,
                )

        # standard normalization
        elif self.epistemic_obj.normalize_y:
            t_inverse_test = self.epistemic_obj.y_scaler.inverse_transform(
                self.epistemic_obj.mixture_ppf(sample_test, [self.t_cutoff])
                .numpy()
                .flatten()
                .reshape(-1, 1)
            ).flatten()

        else:
            t_inverse_test = (
                self.epistemic_obj.mixture_ppf(sample_test, [self.t_cutoff])
                .numpy()
                .flatten()
            )

        pred = self.nc_score.predict(
            X_test,
            t_inverse_test,
            ensemble=ensemble,
        )
        return pred
