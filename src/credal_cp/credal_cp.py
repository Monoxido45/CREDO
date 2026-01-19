from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator

from sklearn.model_selection import train_test_split
import scipy.stats as st

from sklearn.utils.validation import check_is_fitted
from credal_cp.epistemic_models import (
    MDN_model,
    GP_model,
    GPApprox_model,
    BART_model,
    DE_MDN_model,
)
from tqdm import tqdm


class CredalCPRegressor(BaseEstimator):
    """
    CredalCPRegressor

    Credal Conformal Prediction regressor wrapper that constructs a nonconformity-score
    object around an existing fitted base model and stores configuration used to
    build credal sets.

    nc_type : string
        Type of nonconformity score to be used. Options include:
        - "Quantile": Quantile-based imprecise nonconformity score.

    base_model : string or fitted estimator
        String indicating the type of base model for Y given X to be used or a an already fitted estimator.

    alpha : float
        Significance level used when constructing credal sets or, when applicable,
        forwarded to the nc_score constructor. Typical values are in (0, 1).

    **kwargs
        Additional keyword arguments forwarded to the nc_score constructor.

    Attributes
    base_model : estimator
        The provided fitted base model.

    nc_score : object
        The instantiated nonconformity-score object returned by calling nc_score(...)
        during initialization. This object is expected to expose the methods used
        by the credal CP algorithm (e.g., methods to compute nonconformity scores,
        quantiles, or predictive sets), depending on the chosen scorer implementation.

    alpha : float
        Stored alpha value.

    Notes
    -----
    - This class does not itself fit the base_model; it expects base_model to be
      already trained. The nc_score is initialized with is_fitted=True for that reason.
    - The heuristic to decide whether to forward `alpha` to the nc_score constructor
      is: forward if "Quantile" appears in str(nc_score) OR base_model_type is True.
      This supports scorer classes that require an alpha parameter (for example,
      quantile-based scorers).
    - Any additional keyword arguments are passed directly to the nc_score constructor
      and may influence scorer behavior.

    Examples
    --------
    # Pseudocode example (replace with concrete implementations):
    # nc = QuantileNCScoreClass
    # fitted_model = some_fitted_regressor
    # credal = CredalCPRegressor(nc_score=nc, base_model=fitted_model, alpha=0.1)
    """
    def __init__(
        self,
        nc_type,
        base_model,
        alpha,
    ):
        self.nc_type = nc_type
        self.base_model = base_model

        # metadata about the provided base_model
        # Simplify: require sklearn estimators to be instantiated (instances of BaseEstimator).
        # If a user passes an estimator class, raise an informative error.
        if isinstance(base_model, type) and issubclass(base_model, BaseEstimator):
            raise ValueError("Please pass an instantiated sklearn estimator (an instance of BaseEstimator), not an estimator class.")
        self.base_model_type = None
        self.base_is_sklearn = False
        self.base_is_fitted = False

        # case 2: an sklearn Estimator instance
        if isinstance(base_model, BaseEstimator):
            self.base_model = base_model
            self.base_is_sklearn = True
            try:
                check_is_fitted(self.base_model)
                self.base_is_fitted = True
            except Exception:
                self.base_is_fitted = False
            self.base_model_type = "sklearn_fitted_estimator" if self.base_is_fitted else "sklearn_unfitted_estimator"

        # case 3: non-sklearn object
        else:
            self.base_model = base_model
            self.base_is_sklearn = False
            self.base_is_fitted = False
            self.base_model_type = "string_unfitted"
        # checking if base model is fitted
        self.alpha = alpha
    
    def fit(self, 
            X, 
            y,
            nn_type="MC_Dropout",
            proportion_train=0.7,
            n_models = 15,
            epochs=500,
            lr=0.001,
            gamma=0.99,
            batch_size=32,
            step_size=5,
            weight_decay=0,
            verbose=0,
            patience=15,
            scale=False,
            random_seed_split=0,
            random_seed_fit=1250,
            n_MCMC=2000,
            **fit_params):
        self.nn_type = nn_type
        if self.base_is_sklearn and not self.base_is_fitted:
            self.base_model.fit(X, y, **fit_params)
            self.base_is_fitted = True
        elif not self.base_is_sklearn and self.base_model_type == "string_unfitted":
            # MDN + Dropout or other BNN approximations
            if self.base_model == "MDN" and self.nn_type == "MC_Dropout":
                print("Fitting MC Dropout MDN model")
                self.base_model_type = "MDN"
                self.base_model = MDN_model(
                    input_shape = X.shape[1],
                   **fit_params
                )

                self.base_model.fit(
                    X,
                    y,
                    proportion_train=proportion_train,
                    epochs=epochs,
                    lr=lr,
                    gamma=gamma,
                    batch_size=batch_size,
                    step_size=step_size,
                    weight_decay=weight_decay,
                    verbose=verbose,
                    patience=patience,
                    scale=scale,
                    random_seed_split=random_seed_split,
                    random_seed_fit=random_seed_fit,
                )
                self.base_is_fitted = True
            elif self.base_model == "MDN" and self.nn_type == "Ensemble":
                print("Fitting Deep Ensemble MDN model")
                self.base_model_type = "MDN"
                self.base_model = DE_MDN_model(
                    input_shape = X.shape[1],
                    n_models = n_models,
                     **fit_params
                )

                self.base_model.fit(
                    X,
                    y,
                    proportion_train=proportion_train,
                    n_epochs=epochs,
                    lr=lr,
                    gamma=gamma,
                    batch_size=batch_size,
                    step_size=step_size,
                    weight_decay=weight_decay,
                    patience=patience,
                    scale=scale,
                    random_seed_split=random_seed_split,
                    random_seed_fit=random_seed_fit,
                )


            # Analytic GP (slow for large datasets)
            elif self.base_model == "GP":
                print("Fitting Gaussian Process model")
                self.base_model_type = "GP"
                self.base_model = GP_model(
                   **fit_params
                )

                self.base_model.fit(
                    X,
                    y,
                    scale = scale,
                    random_state=random_seed_fit,
                )
                self.base_is_fitted = True
            
            # Variational GP (faster for large datasets)
            elif self.base_model == "GP_Approx":
                print("Fitting Approximate Gaussian Process model")
                self.base_model_type = "GP_Approx"
                self.base_model = GPApprox_model(
                   **fit_params
                )

                self.base_model.fit(
                    X,
                    y,
                    proportion_train=proportion_train,
                    batch_size=batch_size,
                    patience=patience,
                    verbose=verbose,
                    random_seed_split=random_seed_split,
                    random_seed_fit=random_seed_fit,
                )
                self.base_is_fitted = True
            
            # BART model
            elif self.base_model == "BART":
                print("Fitting BART model")
                self.base_model_type = "BART"
                self.base_model = BART_model(
                   **fit_params
                )

                self.base_model.fit(
                    X,
                    y,
                    n_sample=n_MCMC, 
                    random_seed=random_seed_fit
                )
                self.base_is_fitted = True
        return self
    
    def calibrate(
            self, 
            X_calib, 
            y_calib,
            beta=0.1,
            random_seed_calib=0,
            N_samples_MC=300,
            ):
        """
        Fit posterior over parametric family then creates imprecise quantiles as the modified conformal scores.
        Parameters
        ----------
        X_calib : array-like, shape (n_samples, n_features)
            Calibration data features.

        y_calib : array-like, shape (n_samples,)
            Calibration data targets.

        Returns
        -------
        self : object
            Returns self.
        """
        self.beta = beta
        self.rng = np.random.default_rng(random_seed_calib)
        # For this, use mixture quantile to derive the vector of quantiles to be used for each x
        if self.base_model_type == "MDN" and self.nn_type == "MC_Dropout":
            # obtaining samples from posterior for each x_calib
            pi_calib, mu_calib, sigma_calib = self.base_model.predict_mcdropout(
                X_calib, 
                num_samples = N_samples_MC,
                )

            self.pi_calib = pi_calib
            self.mu_calib = mu_calib
            # using samples to obtain quantile vector for each x_calib
            if self.nc_type == "Quantile":
                lower_q = self.alpha / 2
                upper_q = 1 - self.alpha / 2
                i = 0
                q_low_raw, q_upp_raw = [], []

                for x in tqdm(X_calib, desc="Calibrating Credal CP with MC Dropout MDN"):
                    pi_chosen = pi_calib[:, i, :]
                    mu_chosen = mu_calib[:, i, :]
                    sigma_chosen = sigma_calib[:, i, :]

                    q_grid = self.base_model.mixture_quantile(
                    [lower_q, upper_q], 
                    pi_chosen, 
                    mu_chosen, 
                    sigma_chosen,
                    rng= self.rng,
                    )

                    self.q_grid = q_grid

                    # obtaining lower and upper quantiles for the current x
                    q_low_raw.append(np.quantile(q_grid[:, 0], self.beta/2))
                    q_upp_raw.append(np.quantile(q_grid[:, 1], 1 - self.beta/2))
                    i += 1
                
                q_low_raw = np.array(q_low_raw)
                q_upp_raw = np.array(q_upp_raw)
                # with lower and upper quantiles, we can compute the modified nonconformity scores
                self.nc_scores = np.maximum(q_low_raw - y_calib, y_calib - q_upp_raw)
                n = len(self.nc_scores)
                self.cutoff = np.quantile(self.nc_scores, 
                                          q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        
        elif self.base_model_type == "MDN" and self.nn_type == "Ensemble":
            # obtaining samples from posterior for each x_calib
            pi_calib, mu_calib, sigma_calib = self.base_model.predict_ensemble(
                X_calib,
                )
        
            # using samples to obtain quantile vector for each x_calib
            if self.nc_type == "Quantile":
                lower_q = self.alpha / 2
                upper_q = 1 - self.alpha / 2
                i = 0
                q_low_raw, q_upp_raw = [], []

                for x in tqdm(X_calib, desc="Calibrating Credal CP with Ensemble MDN"):
                    pi_chosen = pi_calib[:, i, :]
                    mu_chosen = mu_calib[:, i, :]
                    sigma_chosen = sigma_calib[:, i, :]

                    q_grid = self.base_model.mixture_quantile(
                    [lower_q, upper_q], 
                    pi_chosen, 
                    mu_chosen, 
                    sigma_chosen,
                    rng= self.rng,
                    )

                    # obtaining lower and upper quantiles for the current x
                    q_low_raw.append(np.quantile(q_grid[:, 0], self.beta/2))
                    q_upp_raw.append(np.quantile(q_grid[:, 1], 1 - self.beta/2))
                    i += 1
                
                q_low_array = np.array(q_low_raw)
                q_upp_array = np.array(q_upp_raw)
                # with lower and upper quantiles, we can compute the modified nonconformity scores
                self.nc_scores = np.maximum(q_low_array - y_calib, 
                                            y_calib - q_upp_array)
                n = len(self.nc_scores)
                self.cutoff = np.quantile(
                    self.nc_scores,
                    q=np.ceil((n + 1) * (1 - self.alpha)) / n
                    )
                
        elif self.base_model_type == "GP" or self.base_model_type == "GP_Approx":
            if self.nc_type == "Quantile":
                lower_q = self.alpha / 2
                upper_q = 1 - self.alpha / 2
                i = 0
                
                q_samples = self.base_model.sample_quantiles_from_posterior(
                    X_calib,
                    quantiles=[lower_q, upper_q],
                    n_samples=N_samples_MC,
                    random_seed=random_seed_calib,
                )

                self.q_samples = q_samples

                q_low_grid = q_samples[:, :, 0]
                q_upp_grid = q_samples[:, :, 1]

                # obtaining lower and upper quantiles for each x_calib
                q_low_raw = np.quantile(q_low_grid, self.beta/2, axis=1)
                q_upp_raw = np.quantile(q_upp_grid, 1 - self.beta/2, axis=1)

                # with lower and upper quantiles, we can compute the modified nonconformity scores
                self.nc_scores = np.maximum(q_low_raw - y_calib, y_calib - q_upp_raw)
                n = len(self.nc_scores)
                self.cutoff = np.quantile(self.nc_scores, 
                                          q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        
        elif self.base_model_type == "BART":
            # obtaining samples from the posterior for each x_calib
            if self.nc_type == "Quantile":
                lower_q = self.alpha / 2
                upper_q = 1 - self.alpha / 2
                i = 0
                
                q_samples = self.base_model.sample_quantiles_from_posterior(
                    X_calib,
                    quantile_levels=[lower_q, upper_q],
                    random_seed=random_seed_calib,
                )

                q_low_grid = q_samples[:, :, 0]
                q_upp_grid = q_samples[:, :, 1]

                # obtaining lower and upper quantiles for each x_calib
                q_low_raw = np.quantile(q_low_grid, self.beta/2, axis=0)
                q_upp_raw = np.quantile(q_upp_grid, 1 - self.beta/2, axis=0)

                # with lower and upper quantiles, we can compute the modified nonconformity scores
                self.nc_scores = np.maximum(q_low_raw - y_calib, y_calib - q_upp_raw)
                n = len(self.nc_scores)
                self.cutoff = np.quantile(self.nc_scores, 
                                          q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return self.cutoff
    
    def predict(
            self,
            X_test,
            n_samples=300,
            ):
        """
        Interval prediction using the fitted base model.

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test data features.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted values.
        """
        if self.base_model_type == "MDN" and self.nn_type == "MC_Dropout":
            pi_test, mu_test, sigma_test = self.base_model.predict_mcdropout(
                X_test, 
                num_samples = n_samples,
                )
            
            if self.nc_type == "Quantile":
                # formulating lower and upper quantiles for each x_test
                lower_q = self.alpha / 2
                upper_q = 1 - self.alpha / 2
                i = 0
                q_low_pred, q_upp_pred = [], []

                for x in X_test:
                    pi_chosen = pi_test[:, i, :]
                    mu_chosen = mu_test[:, i, :]
                    sigma_chosen = sigma_test[:, i, :]

                    q_grid = self.base_model.mixture_quantile(
                    [lower_q, upper_q], 
                    pi_chosen, 
                    mu_chosen, 
                    sigma_chosen,
                    rng= self.rng,
                    )
                    
                    # obtaining lower and upper quantiles for the current x
                    q_low_pred.append(np.quantile(q_grid[:, 0], self.beta/2))
                    q_upp_pred.append(np.quantile(q_grid[:, 1], 1 - self.beta/2))
                    i += 1

                q_low_array = np.array(q_low_pred)
                q_upp_array = np.array(q_upp_pred)

                lower_cp = q_low_array - self.cutoff
                upper_cp = q_upp_array + self.cutoff
                
                y_pred = np.column_stack((lower_cp, upper_cp))
                return y_pred
        
        elif self.base_model_type == "MDN" and self.nn_type == "Ensemble":
            pi_test, mu_test, sigma_test = self.base_model.predict_ensemble(
                X_test,
                )
            if self.nc_type == "Quantile":
                # formulating lower and upper quantiles for each x_test
                lower_q = self.alpha / 2
                upper_q = 1 - self.alpha / 2
                i = 0
                q_low_pred, q_upp_pred = [], []

                for x in X_test:
                    pi_chosen = pi_test[:, i, :]
                    mu_chosen = mu_test[:, i, :]
                    sigma_chosen = sigma_test[:, i, :]

                    # generating quantiles
                    q_grid = self.base_model.mixture_quantile(
                    [lower_q, upper_q], 
                    pi_chosen, 
                    mu_chosen, 
                    sigma_chosen,
                    rng= self.rng,
                    )

                    # obtaining lower and upper quantiles for the current x
                    q_low_pred.append(np.quantile(q_grid[:, 0], self.beta/2))
                    q_upp_pred.append(np.quantile(q_grid[:, 1], 1 - self.beta/2))
                    i += 1

                q_low_array = np.array(q_low_pred)
                q_upp_array = np.array(q_upp_pred)

                lower_cp = q_low_array - self.cutoff
                upper_cp = q_upp_array + self.cutoff
                y_pred = np.column_stack((lower_cp, upper_cp))
                return y_pred
        
        elif self.base_model_type == "GP" or self.base_model_type == "GP_Approx":
            if self.nc_type == "Quantile":
                q_samples = self.base_model.sample_quantiles_from_posterior(
                    X_test,
                    quantiles=[self.alpha / 2, 1 - self.alpha / 2],
                    n_samples=n_samples,
                    random_seed=None,
                )

                q_low_grid = q_samples[:, :, 0]
                q_upp_grid = q_samples[:, :, 1]

                # obtaining lower and upper quantiles for each x_test
                q_low_pred = np.quantile(q_low_grid, self.beta/2, axis=1)
                q_upp_pred = np.quantile(q_upp_grid, 1 - self.beta/2, axis=1)

                lower_cp = q_low_pred - self.cutoff
                upper_cp = q_upp_pred + self.cutoff

                y_pred = np.column_stack((lower_cp, upper_cp))
                return y_pred
        
        elif self.base_model_type == "BART":
            if self.nc_type == "Quantile":
                q_samples = self.base_model.sample_quantiles_from_posterior(
                    X_test,
                    quantile_levels=[self.alpha / 2, 1 - self.alpha / 2],
                    random_seed=None,
                )

                q_low_grid = q_samples[:, :, 0]
                q_upp_grid = q_samples[:, :, 1]

                # obtaining lower and upper quantiles for each x_test
                q_low_pred = np.quantile(q_low_grid, self.beta/2, axis=0)
                q_upp_pred = np.quantile(q_upp_grid, 1 - self.beta/2, axis=0)

                lower_cp = q_low_pred - self.cutoff
                upper_cp = q_upp_pred + self.cutoff

                y_pred = np.column_stack((lower_cp, upper_cp))
                return y_pred
        
            
