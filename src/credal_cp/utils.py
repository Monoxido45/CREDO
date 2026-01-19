import numpy as np
from sklearn.base import BaseEstimator

class CQR(BaseEstimator):
    """
    A simple implementation of Conformalized Quantile Regression (CQR).
    This class serves as a placeholder for the actual CQR implementation.
    """

    def __init__(self,
                 base_model,
                 type_model = "MDN",
                 is_fitted = True,
                 variation = "standard",
                 alpha=0.1,
                 ):
        self.alpha = alpha
        self.base_model = base_model
        self.is_fitted = is_fitted
        self.type_model = type_model
        self.variation = variation

    def fit(self, X, y):
        # Placeholder for fitting logic
        if self.is_fitted:
            return self
        else:
            self.base_model.fit(X, y)
            return self
    
    def calibrate(self, X_cal, Y_cal):
        if self.type_model == "MDN":
            self.base_model.set_type_base_model("quantile", self.alpha)
            quantiles_calib = self.base_model.predict(X_cal)
            self.cal_scores = np.maximum(
                quantiles_calib[:, 0] - Y_cal,
                Y_cal - quantiles_calib[:, 1]
            )
        # computing quantile
        self.quantile_calib = np.quantile(
            self.cal_scores,
            np.ceil((1 + len(Y_cal)) * (1 - self.alpha)) / len(Y_cal)
        )
        return self.quantile_calib
        
    def predict(self, X_test):
        quantiles_test = self.base_model.predict(X_test)
        if self.variation == "standard":
            lower_bound = quantiles_test[:, 0] - self.quantile_calib
            upper_bound = quantiles_test[:, 1] + self.quantile_calib
        return lower_bound, upper_bound