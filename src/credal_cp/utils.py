import numpy as np
from sklearn.base import BaseEstimator

class CQR(BaseEstimator):
    """
    A simple implementation of Conformalized Quantile Regression (CQR) and its variants like CQR-r.
    This class wraps around a base quantile regression model (like MDN or BART)
    and provides methods for fitting, calibrating, and predicting conformal prediction intervals.
    Parameters:
    ----------
    base_model : object
        A quantile regression model that has a `predict` method capable of returning quantiles.
    type_model : str, optional
        Type of the base model, either "MDN" or "BART". Default is "MDN".
    is_fitted : bool, optional
        Indicates whether the base model is already fitted. Default is True.
    variation : str, optional
        Type of CQR variation to use: "standard" for standard CQR, "cqr-r" for relative CQR. Default is "standard".
    alpha : float, optional
        Significance level for the prediction intervals. Default is 0.1.
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
        # predicting quantiles on calibration set
        if self.type_model == "MDN":
            self.base_model.set_type_base_model("quantile", self.alpha)
            quantiles_calib = self.base_model.predict(X_cal)
        elif self.type_model == "BART":
            quantiles_calib = self.base_model.predict(X_cal, quantiles = [self.alpha/2, 1 - self.alpha/2])
        
        # computing nonconformity scores
        if self.variation == "standard":
                self.cal_scores = np.maximum(
                quantiles_calib[:, 0] - Y_cal,
                Y_cal - quantiles_calib[:, 1]
                )
        elif self.variation == "cqr-r":
            int_width = quantiles_calib[:, 1] - quantiles_calib[:, 0]
            self.cal_scores = np.maximum(
                (quantiles_calib[:, 0] - Y_cal)/int_width,
                (Y_cal - quantiles_calib[:, 1])/int_width
            )
        # computing quantile
        self.quantile_calib = np.quantile(
            self.cal_scores,
            np.ceil((1 + len(Y_cal)) * (1 - self.alpha)) / len(Y_cal)
        )
        return self.quantile_calib
        
    def predict(self, X_test):
        if self.type_model == "MDN":
            quantiles_test = self.base_model.predict(X_test)
        elif self.type_model == "BART":
            quantiles_test = self.base_model.predict(X_test, quantiles = [self.alpha/2, 1 - self.alpha/2])

        if self.variation == "standard":
            lower_bound = quantiles_test[:, 0] - self.quantile_calib
            upper_bound = quantiles_test[:, 1] + self.quantile_calib
        elif self.variation == "cqr-r":
            int_width = quantiles_test[:, 1] - quantiles_test[:, 0]
            lower_bound = quantiles_test[:, 0] - (self.quantile_calib * int_width)
            upper_bound = quantiles_test[:, 1] + (self.quantile_calib * int_width)
        
        PI_test = np.column_stack((lower_bound, upper_bound))
        return PI_test
    


# Interval score loss
def interval_score_loss(high_est, low_est, actual, alpha):
    return (
        high_est
        - low_est
        + 2 / alpha * (low_est - actual) * (actual < low_est)
        + 2 / alpha * (actual - high_est) * (actual > high_est)
    )

def average_interval_score_loss(high_est, low_est, actual, alpha):
    return np.mean(interval_score_loss(high_est, low_est, actual, alpha))


# general interval length
def compute_interval_length(upper_int, lower_int):
    return upper_int - lower_int


def average_coverage_clf(pred_sets, labels):
    empirical_coverage = pred_sets[np.arange(pred_sets.shape[0]), labels].mean()

    return empirical_coverage

# pearson correlation
def corr_coverage_widths(high_est, low_est, actual):
    coverage_indicator_vector = coverage_indicators(high_est, low_est, actual)
    widths_vector = compute_interval_length(high_est, low_est)
    return np.abs(np.corrcoef(coverage_indicator_vector, widths_vector)[0, 1])


# marginal coverage
def coverage_indicators(high_est, low_est, actual):
    return (high_est >= actual) & (low_est <= actual)


def average_coverage(high_est, low_est, actual):
    return np.mean(coverage_indicators(high_est, low_est, actual))
