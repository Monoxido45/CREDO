from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
import scipy.stats as st
import torch

from credal_cp.scores import (
    LocalRegressionScore,
    RegressionScore,
    APSScore,
    QuantileScore,
)

from scipy.special import inv_boxcox
from sklearn.utils.validation import check_is_fitted
from credal_cp.epistemic_models import (
    MDN_model,
    GP_model,
    GPApprox_model,
    BART_model,
    MC_classifier,
)



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
            n_MCMC=2000,
            **fit_params):
        if self.base_is_sklearn and not self.base_is_fitted:
            self.base_model.fit(X, y, **fit_params)
            self.base_is_fitted = True
        elif not self.base_is_sklearn:
            # TODO: add deep ensemble option/add into epistemic_models
            # MDN + Dropout or other BNN approximations
            if self.base_model == "MDN":
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
            # Analytic GP (slow for large datasets)
            elif self.base_model == "GP":
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
            bnn_type="MC_Dropout",
            num_components=5,
            hidden_layers=[64, 64],
            dropout_rate=0.5,
            random_seed_split=0,
            random_seed_fit=45,
            split_calib=True,
            epistemic_test_thres=2000,
            N_samples_MC=500,
            kernel=None,
            normalize_y=False,
            log_y=False,
            type="gaussian",
            ensemble=False,
            n_cores=6,
            progress=False,
            **kwargs,
            ):
        """
        Fit poserior over parametric family then creates imprecise quantiles as the modified conformal scores.
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
        # TODO: implement dropout-based imprecise method
        # For this, use mixture quantile to derive the vector of quantiles to be used for each x
        # TODO: implement the BART and GP part also
        self.nc_score.calibrate(X_calib, y_calib)
        return self