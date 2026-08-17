import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Interval score loss
def interval_score_components(high_est, low_est, actual, alpha):
    """Return width, lower-tail miss, and upper-tail miss contributions."""
    high_est = np.asarray(high_est).reshape(-1)
    low_est = np.asarray(low_est).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    width = high_est - low_est
    lower_miss = 2 / alpha * (low_est - actual) * (actual < low_est)
    upper_miss = 2 / alpha * (actual - high_est) * (actual > high_est)
    return width, lower_miss, upper_miss


def interval_score_loss(high_est, low_est, actual, alpha):
    width, lower_miss, upper_miss = interval_score_components(
        high_est, low_est, actual, alpha
    )
    return (
        width + lower_miss + upper_miss
    )

def average_interval_score_loss(high_est, low_est, actual, alpha):
    return np.mean(interval_score_loss(high_est, low_est, actual, alpha))

# general interval length
def compute_interval_length(upper_int, lower_int):
    upper_int = np.asarray(upper_int).reshape(-1)
    lower_int = np.asarray(lower_int).reshape(-1)
    return upper_int - lower_int


def average_interval_width(high_est, low_est):
    return np.nanmean(compute_interval_length(high_est, low_est))


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
    high_est = np.asarray(high_est).reshape(-1)
    low_est = np.asarray(low_est).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    return (high_est >= actual) & (low_est <= actual)


def average_coverage(high_est, low_est, actual):
    return np.mean(coverage_indicators(high_est, low_est, actual))


def scarcity_scores_knn(
    X_reference,
    X_query,
    k=None,
    heuristic="log",
    eps=1e-8,
):
    X_reference = np.asarray(X_reference)
    X_query = np.asarray(X_query)
    if X_reference.ndim == 1:
        X_reference = X_reference.reshape(-1, 1)
    if X_query.ndim == 1:
        X_query = X_query.reshape(-1, 1)

    n_reference = X_reference.shape[0]
    if n_reference == 0 or X_query.shape[0] == 0:
        return np.full(X_query.shape[0], np.nan)

    if k is None:
        if heuristic == "exp":
            k = int(np.ceil(np.sqrt(n_reference)))
        elif heuristic == "log":
            k = int(np.ceil(np.log(n_reference + 1)))
        else:
            raise ValueError("heuristic must be 'log' or 'exp'")
    k = max(1, min(int(k), n_reference))

    scaler = StandardScaler()
    X_reference_scaled = scaler.fit_transform(X_reference)
    X_query_scaled = scaler.transform(X_query)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_reference_scaled)
    distances, _ = nn.kneighbors(X_query_scaled)
    score = distances[:, -1]
    return score / (np.median(score) + eps)


def scarcity_scores_isolation_forest(
    X_reference,
    X_query,
    n_estimators=200,
    max_samples="auto",
    max_features=1.0,
    bootstrap=False,
    random_state=None,
    eps=1e-8,
):
    """Estimate covariate scarcity from an Isolation Forest fitted on reference data.

    The returned score is larger for observations that are easier to isolate.
    The forest is fitted only on ``X_reference`` (the training covariates), so
    the test-set scarcity ranking does not use test-set labels or detector
    assignments. ``score_samples`` is used instead of ``decision_function`` so
    the score remains a continuous support diagnostic rather than a thresholded
    outlier decision.
    """
    X_reference = np.asarray(X_reference)
    X_query = np.asarray(X_query)
    if X_reference.ndim == 1:
        X_reference = X_reference.reshape(-1, 1)
    if X_query.ndim == 1:
        X_query = X_query.reshape(-1, 1)

    if X_reference.shape[0] == 0 or X_query.shape[0] == 0:
        return np.full(X_query.shape[0], np.nan)

    if max_samples != "auto":
        try:
            numeric_max_samples = float(max_samples)
        except (TypeError, ValueError):
            numeric_max_samples = max_samples
        if isinstance(numeric_max_samples, float) and numeric_max_samples.is_integer():
            max_samples = int(numeric_max_samples)
        else:
            max_samples = numeric_max_samples

    scaler = StandardScaler()
    X_reference_scaled = scaler.fit_transform(X_reference)
    X_query_scaled = scaler.transform(X_query)

    detector = IsolationForest(
        n_estimators=int(n_estimators),
        max_samples=max_samples,
        max_features=float(max_features),
        bootstrap=bool(bootstrap),
        random_state=random_state,
        n_jobs=-1,
    )
    detector.fit(X_reference_scaled)

    # Isolation Forest assigns lower score_samples values to more isolated
    # points; negate them so that larger values consistently mean more scarce.
    score = -detector.score_samples(X_query_scaled)
    return score / (np.median(score) + eps)


def coverage_by_score_quantile(
    high_est,
    low_est,
    actual,
    scores,
    n_bins=4,
    bin_edges=None,
):
    scores = np.asarray(scores).reshape(-1)
    covered = coverage_indicators(high_est, low_est, actual).astype(float)
    if scores.shape[0] != covered.shape[0]:
        raise ValueError("scores and interval arrays must have the same length")

    finite_scores = np.isfinite(scores)
    scores = scores[finite_scores]
    covered = covered[finite_scores]

    if covered.shape[0] == 0:
        n_output_bins = n_bins if bin_edges is None else len(bin_edges) - 1
        return np.full(n_output_bins, np.nan), np.zeros(n_output_bins), bin_edges

    if bin_edges is None:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(scores, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    n_output_bins = len(bin_edges) - 1
    bin_ids = np.digitize(scores, bin_edges[1:-1], right=True)

    coverages = np.full(n_output_bins, np.nan)
    counts = np.zeros(n_output_bins, dtype=int)
    for bin_idx in range(n_output_bins):
        mask = bin_ids == bin_idx
        counts[bin_idx] = int(mask.sum())
        if counts[bin_idx] > 0:
            coverages[bin_idx] = np.mean(covered[mask])

    return coverages, counts, bin_edges



########## All CQR functions for toy examples ##########
def _coerce_interval_predictions(predictions):
    if isinstance(predictions, np.ndarray):
        if predictions.ndim == 2:
            if predictions.shape[0] == 2 and predictions.shape[1] != 2:
                return predictions.T
            if predictions.shape[1] == 2:
                return predictions
            if predictions.shape[1] > 2:
                return predictions[:, [0, -1]]
        raise ValueError("Expected a 2D array with lower and upper predictions.")

    if isinstance(predictions, (list, tuple)):
        if len(predictions) < 2:
            raise ValueError("Expected lower and upper predictions.")
        lower = np.asarray(predictions[0]).reshape(-1)
        upper = np.asarray(predictions[-1]).reshape(-1)
        return np.column_stack((lower, upper))

    raise ValueError("Unsupported prediction format for interval predictions.")


def _extract_interval_models(base_model):
    if hasattr(base_model, "lower_model") and hasattr(base_model, "upper_model"):
        return base_model.lower_model, base_model.upper_model

    if isinstance(base_model, (list, tuple)) and len(base_model) >= 2:
        return base_model[0], base_model[-1]

    raise ValueError(
        "Expected a fitted interval model pair or an object exposing "
        "`lower_model` and `upper_model`."
    )


def _conformal_cutoff(scores, alpha):
    scores = np.sort(np.asarray(scores).reshape(-1))
    n = scores.shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return scores[k - 1]


def _weighted_quantile(values, quantile, sample_weight=None):
    values = np.asarray(values).reshape(-1)
    if sample_weight is None:
        return np.quantile(values, quantile)

    weights = np.asarray(sample_weight).reshape(-1)
    mask = weights > 0
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]

    if total <= 0:
        return np.quantile(values, quantile)

    return np.interp(quantile * total, cumulative, values)


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
            if not hasattr(self.base_model, "fit"):
                raise ValueError(
                    "The provided base model does not expose `fit`. "
                    "Pass a pre-fitted model pair with is_fitted=True."
                )
            self.base_model.fit(X, y)
            return self

    def _predict_quantiles(self, X):
        if self.type_model == "MDN":
            self.base_model.set_type_base_model("quantile", self.alpha)
            return _coerce_interval_predictions(self.base_model.predict(X))

        if self.type_model == "BART":
            return _coerce_interval_predictions(
                self.base_model.predict(X, quantiles=[self.alpha / 2, 1 - self.alpha / 2])
            )

        if self.type_model.lower() == "catboost":
            lower_model, upper_model = _extract_interval_models(self.base_model)
            return np.column_stack((lower_model.predict(X), upper_model.predict(X)))

        if hasattr(self.base_model, "predict"):
            return _coerce_interval_predictions(self.base_model.predict(X))

        return _coerce_interval_predictions(self.base_model)
    
    def calibrate(self, X_cal, Y_cal):
        # predicting quantiles on calibration set
        quantiles_calib = self._predict_quantiles(X_cal)
        
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
        self.quantile_calib = _conformal_cutoff(self.cal_scores, self.alpha)
        return self.quantile_calib
        
    def predict(self, X_test):
        quantiles_test = self._predict_quantiles(X_test)

        if self.variation == "standard":
            lower_bound = quantiles_test[:, 0] - self.quantile_calib
            upper_bound = quantiles_test[:, 1] + self.quantile_calib
        elif self.variation == "cqr-r":
            int_width = quantiles_test[:, 1] - quantiles_test[:, 0]
            lower_bound = quantiles_test[:, 0] - (self.quantile_calib * int_width)
            upper_bound = quantiles_test[:, 1] + (self.quantile_calib * int_width)
        
        PI_test = np.column_stack((lower_bound, upper_bound))
        return PI_test


class AdaptiveGammaCQR(BaseEstimator):
    """
    Plain CQR with a local adaptive gamma(x) schedule.

    This class keeps the standard conformalization step from CQR, but replaces the
    fixed nominal quantile levels used by the base model with local quantile levels
    induced by gamma(x). The local gamma schedule is estimated from X alone using
    nearest-neighbor sparsity, mirroring the adaptive-gamma idea used elsewhere in
    the repository without relying on the CredalCPRegressor wrapper.

    Supported base models:
    - `type_model="BART"` with `sample_quantiles_from_posterior`
    - `type_model="catboost"` with a fitted lower/upper quantile model pair
      (either `[lower_model, upper_model]` or a wrapper exposing
      `lower_model` / `upper_model`)
    - `type_model="rfqr"` with a fitted RF quantile forest or a fitted `uacqr`
      object exposing `cqr_base_model`
    """

    def __init__(
        self,
        base_model,
        type_model="BART",
        is_fitted=True,
        alpha=0.1,
        gamma=0.1,
        k=None,
        heuristic="log",
    ):
        self.base_model = base_model
        self.type_model = type_model
        self.is_fitted = is_fitted
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        self.heuristic = heuristic

    def fit(self, X, y=None, **fit_params):
        if not self.is_fitted:
            if not hasattr(self.base_model, "fit"):
                raise ValueError(
                    "The provided base model does not expose `fit`. "
                    "Pass a pre-fitted model pair with is_fitted=True."
                )
            self.base_model.fit(X, y, **fit_params)
            self.is_fitted = True

        self.fit_gamma(X, k=self.k, heuristic=self.heuristic)
        return self

    def fit_gamma(self, X, C_base=6.672, heuristic="log", k=None):
        self.scaler_x = StandardScaler().fit(X)
        X_scaled = self.scaler_x.transform(X)

        if k is None:
            n, d = X.shape
            if heuristic == "exp":
                k = int(np.ceil(C_base * (n ** (4 / (4 + d)))))
            elif heuristic == "log":
                k = int(np.ceil(d * np.log(n)))
            else:
                raise ValueError(f"Unknown heuristic: {heuristic}")
        else:
            k = int(k)

        k = max(2, min(k, X.shape[0]))
        self.gamma_model = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, _ = self.gamma_model.kneighbors(X_scaled)
        last_neighbor_dist = distances[:, -1]
        self.q_lo_gamma = np.quantile(last_neighbor_dist, 0.5)
        self.q_hi_gamma = np.quantile(last_neighbor_dist, 0.95)
        self.k_ = k
        return self

    @staticmethod
    def sigma(u):
        return 1 / (1 + np.exp(-u))

    def compute_gamma(self, X, eps=1e-5, gamma_max=0.75, tau=1.0, gamma_min=None):
        X_scaled = self.scaler_x.transform(X)
        distances, _ = self.gamma_model.kneighbors(X_scaled)
        last_neighbor_dist = np.log1p(distances[:, -1])
        q_lo = np.log1p(self.q_lo_gamma)
        q_hi = np.log1p(self.q_hi_gamma)

        scarce_score = (last_neighbor_dist - q_lo) / (q_hi - q_lo + eps)
        if gamma_min is None:
            gamma_min = self.gamma
        else:
            gamma_min = max(gamma_min, 0.01)

        gamma_values = gamma_max - ((gamma_max - gamma_min) * self.sigma(scarce_score / tau))
        return gamma_values

    def _get_quantile_samples(self, X, random_seed=0):
        quantile_levels = [self.alpha / 2, 1 - self.alpha / 2]

        if self.type_model == "BART":
            if not hasattr(self.base_model, "sample_quantiles_from_posterior"):
                raise AttributeError(
                    "The provided base model must implement sample_quantiles_from_posterior."
                )

            return self.base_model.sample_quantiles_from_posterior(
                X,
                quantile_levels=quantile_levels,
                random_seed=random_seed,
            )

        if self.type_model.lower() == "catboost":
            lower_model, upper_model = _extract_interval_models(self.base_model)

            if not hasattr(lower_model, "tree_count_") or not hasattr(upper_model, "tree_count_"):
                raise AttributeError(
                    "CatBoost adaptive CQR requires fitted quantile models exposing `tree_count_`."
                )

            n_members = min(lower_model.tree_count_, upper_model.tree_count_)
            q_low_grid = np.empty((n_members, len(X)))
            q_upp_grid = np.empty((n_members, len(X)))

            for t in range(n_members):
                q_low_grid[t] = lower_model.predict(X, ntree_end=t + 1)
                q_upp_grid[t] = upper_model.predict(X, ntree_end=t + 1)

            return np.stack((q_low_grid, q_upp_grid), axis=-1)

        if self.type_model.lower() == "rfqr":
            rf_model = getattr(self.base_model, "cqr_base_model", self.base_model)

            required_attrs = ("apply", "y_train_", "y_weights_", "y_train_leaves_", "estimators_")
            if not all(hasattr(rf_model, attr) for attr in required_attrs):
                raise AttributeError(
                    "RFQR adaptive CQR requires a fitted RF quantile forest or a fitted "
                    "`uacqr` object exposing `cqr_base_model`."
                )

            leaf_ids = rf_model.apply(X)
            if leaf_ids.ndim == 1:
                leaf_ids = leaf_ids[:, None]

            y_train = np.asarray(rf_model.y_train_).reshape(-1)
            y_weights = np.asarray(rf_model.y_weights_)
            y_train_leaves = np.asarray(rf_model.y_train_leaves_)

            n_members = y_train_leaves.shape[0]
            q_low_grid = np.empty((n_members, len(X)))
            q_upp_grid = np.empty((n_members, len(X)))

            for t in range(n_members):
                for i in range(len(X)):
                    in_leaf = y_train_leaves[t] == leaf_ids[i, t]
                    leaf_y = y_train[in_leaf]
                    leaf_w = y_weights[t, in_leaf]

                    if leaf_y.size == 0:
                        q_low_grid[t, i] = np.nan
                        q_upp_grid[t, i] = np.nan
                        continue

                    q_low_grid[t, i] = _weighted_quantile(leaf_y, quantile_levels[0], sample_weight=leaf_w)
                    q_upp_grid[t, i] = _weighted_quantile(leaf_y, quantile_levels[1], sample_weight=leaf_w)

            return np.stack((q_low_grid, q_upp_grid), axis=-1)

        raise NotImplementedError(
            "AdaptiveGammaCQR currently supports type_model='BART', "
            "type_model='catboost', and type_model='rfqr'."
        )

    def calibrate(
        self,
        X_cal,
        Y_cal,
        random_seed_calib=0,
        gamma_max=0.75,
        gamma_min=None,
        tau=1.0,
        eps=1e-5,
    ):
        gamma_values = self.compute_gamma(
            X_cal,
            eps=eps,
            gamma_max=gamma_max,
            gamma_min=gamma_min,
            tau=tau,
        )
        q_samples = self._get_quantile_samples(X_cal, random_seed=random_seed_calib)
        q_low_grid = q_samples[:, :, 0]
        q_upp_grid = q_samples[:, :, 1]

        q_low_raw = np.array(
            [np.quantile(q_low_grid[:, i], gamma_values[i] / 2) for i in range(len(X_cal))]
        )
        q_upp_raw = np.array(
            [np.quantile(q_upp_grid[:, i], 1 - gamma_values[i] / 2) for i in range(len(X_cal))]
        )

        self.cal_scores = np.maximum(q_low_raw - Y_cal, Y_cal - q_upp_raw)
        n = len(self.cal_scores)
        self.quantile_calib = _conformal_cutoff(self.cal_scores, self.alpha)

        self.gamma_cal_ = gamma_values
        self.gamma_max_ = gamma_max
        self.gamma_min_ = self.gamma if gamma_min is None else gamma_min
        self.tau_ = tau
        return self.quantile_calib

    def predict(self, X_test, random_seed_test=0):
        gamma_values = self.compute_gamma(
            X_test,
            gamma_max=self.gamma_max_,
            gamma_min=self.gamma_min_,
            tau=self.tau_,
        )
        q_samples = self._get_quantile_samples(X_test, random_seed=random_seed_test)
        q_low_grid = q_samples[:, :, 0]
        q_upp_grid = q_samples[:, :, 1]

        q_low_pred = np.array(
            [np.quantile(q_low_grid[:, i], gamma_values[i] / 2) for i in range(len(X_test))]
        )
        q_upp_pred = np.array(
            [np.quantile(q_upp_grid[:, i], 1 - gamma_values[i] / 2) for i in range(len(X_test))]
        )

        lower_bound = q_low_pred - self.quantile_calib
        upper_bound = q_upp_pred + self.quantile_calib

        self.gamma_test_ = gamma_values
        return np.column_stack((lower_bound, upper_bound))
