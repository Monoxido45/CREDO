import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Interval score loss
def interval_score_loss(high_est, low_est, actual, alpha):
    high_est = np.asarray(high_est).reshape(-1)
    low_est = np.asarray(low_est).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
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


def _min_average_subarray_at_least(values, min_count, tol=1e-6, max_iter=40):
    low = float(np.min(values))
    high = float(np.max(values))
    for _ in range(max_iter):
        mid = (low + high) / 2
        transformed = values - mid
        prefix = np.concatenate([[0.0], np.cumsum(transformed)])
        max_prefix = prefix[0]
        has_average_below_mid = False
        for end in range(min_count, values.shape[0] + 1):
            max_prefix = max(max_prefix, prefix[end - min_count])
            if prefix[end] - max_prefix <= 0:
                has_average_below_mid = True
                break
        if has_average_below_mid:
            high = mid
        else:
            low = mid
        if high - low <= tol:
            break
    return high


def worst_slab_coverage(
    X,
    high_est,
    low_est,
    actual,
    n_directions=100,
    min_fraction=0.1,
    random_state=0,
    include_axes=False,
):
    """Random-projection approximation to worst-slab coverage."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    covered = coverage_indicators(high_est, low_est, actual).astype(float)
    n, d = X.shape
    if covered.shape[0] != n:
        raise ValueError("X and interval arrays must have the same number of rows")

    if n == 0 or d == 0:
        return np.nan

    min_count = int(np.ceil(min_fraction * n))
    min_count = max(1, min(min_count, n))

    X_scaled = StandardScaler().fit_transform(X)
    directions = []
    if include_axes:
        directions.append(np.eye(d))

    n_random = max(0, int(n_directions))
    if n_random > 0:
        rng = np.random.default_rng(random_state)
        random_dirs = rng.normal(size=(n_random, d))
        norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
        random_dirs = random_dirs / np.maximum(norms, 1e-12)
        directions.append(random_dirs)

    if not directions:
        directions.append(np.eye(d))
    directions = np.vstack(directions)

    worst_coverage = np.inf
    for direction in directions:
        projected = X_scaled @ direction
        order = np.argsort(projected)
        covered_sorted = covered[order]
        slab_coverage = _min_average_subarray_at_least(covered_sorted, min_count)
        worst_coverage = min(worst_coverage, slab_coverage)

    return worst_coverage


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
