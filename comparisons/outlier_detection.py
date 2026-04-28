import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def detector_suffix(outlier_detector):
    return "" if outlier_detector == "lof" else f"_{outlier_detector}"


def parse_max_samples(value):
    if value == "auto":
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer() and numeric >= 1:
        return int(numeric)
    return numeric


def select_outlier_inlier_indices(
    X,
    y,
    outlier_detector="lof",
    contamination=0.05,
    inlier_size=0.2,
    n_neighbors=15,
    n_components=2,
    tsne_random_state=120,
    iforest_n_estimators=200,
    iforest_max_samples="auto",
    iforest_max_features=1.0,
    iforest_bootstrap=False,
    random_state=None,
):
    tsne = TSNE(n_components=n_components, random_state=tsne_random_state)
    X_embedded = tsne.fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_embedded)

    if outlier_detector == "lof":
        detector = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        out_pred = detector.fit_predict(X_scaled)
        normality_scores = detector.negative_outlier_factor_
    elif outlier_detector == "isolation_forest":
        detector = IsolationForest(
            n_estimators=iforest_n_estimators,
            contamination=contamination,
            max_samples=parse_max_samples(iforest_max_samples),
            max_features=iforest_max_features,
            bootstrap=iforest_bootstrap,
            random_state=random_state,
            n_jobs=-1,
        )
        out_pred = detector.fit_predict(X_scaled)
        normality_scores = detector.decision_function(X_scaled)
    else:
        raise ValueError(f"Unknown outlier_detector={outlier_detector}")

    outlier_indexes = np.where(out_pred == -1)[0]
    inlier_indexes = np.setdiff1d(np.arange(len(y)), outlier_indexes)
    inlier_scores = normality_scores[inlier_indexes]
    size = max(1, int((len(y) - len(outlier_indexes)) * inlier_size))
    most_inlier_idxs = inlier_indexes[np.argsort(inlier_scores)[::-1][:size]]

    return outlier_indexes, most_inlier_idxs
