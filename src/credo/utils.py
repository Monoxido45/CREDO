import numpy as np

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
