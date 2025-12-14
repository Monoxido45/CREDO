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

    nc_score : callable or class
        Constructor (callable) for the nonconformity-score object. It will be called
        as nc_score(base_model, is_fitted=True, ...) to produce a concrete scorer
        instance used by this regressor. When the string "Quantile" appears in
        str(nc_score) or when `base_model_type` is True, `alpha` will also be
        forwarded to nc_score at construction time (i.e., nc_score(..., alpha=alpha, ...)).
        Accepts either classes or factory functions that return an object implementing
        the required nonconformity-score API.

    base_model : estimator
        A fitted base estimator (e.g., sklearn-like regressor). This object is
        forwarded to the nc_score constructor as the underlying model for scoring.
        The constructor assumes this model is already fitted and passes is_fitted=True
        to the nc_score callable.

    alpha : float
        Significance level used when constructing credal sets or, when applicable,
        forwarded to the nc_score constructor. Typical values are in (0, 1).

    base_model_type : bool, optional
        Optional flag used as an explicit indicator that the base model / scoring
        routine requires `alpha` to be passed to the nc_score constructor. If set
        to True, `alpha` is forwarded to nc_score; if False or None, forwarding of
        `alpha` is decided by inspecting str(nc_score) for the substring "Quantile".
        Default: None.

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

    base_model_type : bool or None
        Stored flag indicating the explicit behavior for forwarding `alpha`.

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
        nc_score,
        base_model,
        alpha,
        base_model_type=None,
        **kwargs,
    ):
        self.base_model_type = base_model_type
        self.base_model = base_model
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(
                base_model, is_fitted=True, alpha=alpha, **kwargs
            )
        else:
            self.nc_score = nc_score(base_model, is_fitted=True, **kwargs)

        # checking if base model is fitted
        self.alpha = alpha
    
    def fit(self, X = None, y = None):
        """
        Fit method for compatibility; does nothing as base_model is assumed fitted.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data features.

        y : array-like, shape (n_samples,)
            Training data targets.

        Returns
        -------
        self : object
            Returns self.
        """
        return self
    
    def calibrate(
            self, 
            X_calib, 
            y_calib,
            epistemic_model="MC_dropout",
            scale=False,
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
        self.nc_score.calibrate(X_calib, y_calib)
        return self