import os
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
import gc
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from tqdm import tqdm

original_path = os.getcwd()
os.chdir(os.path.join(original_path, "comparisons"))

from credo.credal_cp import CredalCPRegressor
from credo.epistemic_models import QuantileRegressionNN as CredoQuantileRegressionNN
from credo.utils import (
    average_interval_score_loss,
    interval_score_components,
    average_coverage,
    average_interval_width,
    compute_interval_length,
    coverage_by_score_quantile,
    scarcity_scores_isolation_forest,
    scarcity_scores_knn,
)
from jaxtyping import install_import_hook
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# uacqr part
from uacqr import uacqr
from helper import QuantileRegressionNN as SharedQuantileRegressionNN
# EPIC part
from epic import QuantileScore, EPIC_split
from outlier_detection import detector_suffix, select_outlier_inlier_indices
import pickle
import os

os.chdir(original_path)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise ValueError("Boolean value expected.")


def expand_choice(value, choices):
    """Expand a single detector/score choice while preserving deterministic order."""
    if value == "both":
        return list(choices)
    return [value]


parser = ArgumentParser()
parser.add_argument("-alpha", "--alpha",type=float, default=0.1, help="miscoverage level for conformal prediction")
parser.add_argument("-gamma","--gamma", type=float, default=0.2, help="fixed CREDO gamma parameter")
parser.add_argument("-n_rep", "--n_rep", type=int, default=50, help="number of repetitions for the experiment")
parser.add_argument("-n_MCMC", "--n_MCMC", type=int, default=1000, help="number of MCMC samples")
parser.add_argument("-seed_initial", "--seed_initial", type=int, default=125,
                     help="initial seed for random generator to create seeds for repetitions")
parser.add_argument("-dataset", "--dataset", type=str, default="airfoil", help="dataset to use for the experiment")
parser.add_argument("-base_model", "--base_model", type=str, default="qnn", help="Base quantile model for competitors: 'qnn', 'qnn_mc', 'rfqr', or 'catboost'")
parser.add_argument("-uacqr_model", "--uacqr_model", type=str, default=None, help="Deprecated alias for --base_model")
parser.add_argument("-results_tag", "--results_tag", type=str, default="", help="Optional suffix for result folders/files, useful for sensitivity tests.")
parser.add_argument("-outlier_analysis", "--outlier_analysis", type=str2bool, default=False, help="whether to perform outlier/inlier analysis using the selected detector")
parser.add_argument("-n_cores", "--n_cores", type=int, default=4, help="number of cores to use for parallel processing")
parser.add_argument("-kernel", "--kernel", type=str, default="RBF + Matern52", 
                    help="kernel to use for Gaussian Process regression in CREDO: 'RBF', 'Matern32', 'Matern52' or 'RationalQuadratic'")
parser.add_argument("-kernel_noise", "--kernel_noise", type=str, default="RBF", 
                    help="kernel to use for Gaussian Process noise in CREDO: 'RBF', 'Matern32', 'Matern52' or 'RationalQuadratic'")
parser.add_argument("-activation_noise", "--activation_noise", type=str, default="softplus", 
                    help="activation function for noise in Gaussian Process")
parser.add_argument("-outlier_same_time", "--outlier_same_time", type=str2bool, default=False, 
                    help="whether to analyze outliers at the same time as fitting the models or as a separate step after fitting")
parser.add_argument("-gamma_max", "--gamma_max", type=float, default=0.9, help="maximum adaptive gamma value")
parser.add_argument("-gamma_min", "--gamma_min", type=float, default=0.1, help="minimum adaptive gamma value; defaults to --gamma")
parser.add_argument("-tau_gamma", "--tau_gamma", type=float, default=1.0, help="temperature for the scarcity-to-gamma map")
parser.add_argument("-credo_dropout", "--credo_dropout", type=float, default=0.1, help="MC-dropout rate used to construct the CREDO envelope")
parser.add_argument(
    "-qnn_hidden_layers",
    "--qnn_hidden_layers",
    type=int,
    nargs=3,
    default=[64, 64, 32],
    metavar=("H1", "H2", "H3"),
    help="Hidden-layer widths for the shared QNN backbone (default: 64 64 32)",
)
parser.add_argument("-k_gamma", "--k_gamma", type=int, default=None, help="fixed k for kNN scarcity; defaults to the selected heuristic")
parser.add_argument("-heuristic_gamma", "--heuristic_gamma", type=str, default="log", help="k heuristic for adaptive gamma: 'log' or 'exp'")
parser.add_argument("-scarcity_bins", "--scarcity_bins", type=int, default=4, help="number of scarcity-score quantile bins for stratified coverage")
parser.add_argument("-scarcity_k", "--scarcity_k", type=int, default=None, help="fixed k for diagnostic kNN scarcity score; defaults to --scarcity_heuristic")
parser.add_argument("-scarcity_heuristic", "--scarcity_heuristic", type=str, default="log", help="k heuristic for diagnostic scarcity score: 'log' or 'exp'")
parser.add_argument("-scarcity_method", "--scarcity_method", choices=["knn", "isolation_forest", "both"], default="both", help="method used to define the diagnostic scarcity score")
parser.add_argument("-scarcity_iforest_n_estimators", "--scarcity_iforest_n_estimators", type=int, default=200, help="number of trees for the Isolation Forest scarcity score")
parser.add_argument("-scarcity_iforest_max_samples", "--scarcity_iforest_max_samples", default="auto", help="max_samples for the Isolation Forest scarcity score")
parser.add_argument("-scarcity_iforest_max_features", "--scarcity_iforest_max_features", type=float, default=1.0, help="max_features for the Isolation Forest scarcity score")
parser.add_argument("-scarcity_iforest_bootstrap", "--scarcity_iforest_bootstrap", action="store_true", help="bootstrap samples for the Isolation Forest scarcity score")
parser.add_argument("-credo_dropout_training", "--credo_dropout_training", type=str2bool, default=False, help="whether CREDO's QNN dropout is active during training; MC-dropout remains used for the credal envelope")
parser.add_argument(
    "--shared-qnn-backbone",
    dest="shared_qnn_backbone",
    action="store_true",
    default=False,
    help="reuse one QNN backbone (kept for pilot compatibility; disabled by default)",
)
parser.add_argument(
    "--separate-qnn-backbone",
    dest="shared_qnn_backbone",
    action="store_false",
    help="fit separate QNNs; intended only for ablation or backward-compatibility checks",
)
parser.add_argument(
    "--credo-first-qnn-backbone",
    dest="credo_first_qnn_backbone",
    action="store_true",
    help="fit the original CREDO QNN first and reuse it for CQR/UACQR/EPIC (experimental)",
)
parser.add_argument("-outlier_detector", "--outlier_detector", choices=["lof", "isolation_forest", "both"], default="both", help="Detector used for outlier/inlier diagnostics.")
parser.add_argument("-outlier_contamination", "--outlier_contamination", type=float, default=0.05)
parser.add_argument("-outlier_neighbors", "--outlier_neighbors", type=int, default=15)
parser.add_argument("-inlier_size", "--inlier_size", type=float, default=0.2)
parser.add_argument("-tsne_components", "--tsne_components", type=int, default=2)
parser.add_argument("-tsne_random_state", "--tsne_random_state", type=int, default=120)
parser.add_argument("-iforest_n_estimators", "--iforest_n_estimators", type=int, default=200)
parser.add_argument("-iforest_max_samples", "--iforest_max_samples", default="auto")
parser.add_argument("-iforest_max_features", "--iforest_max_features", type=float, default=1.0)
parser.add_argument("-iforest_bootstrap", "--iforest_bootstrap", action="store_true")
args = parser.parse_args()

alpha = args.alpha
gamma = args.gamma
n_rep = args.n_rep
n_MCMC = args.n_MCMC
seed_initial = args.seed_initial
dataset = args.dataset
base_model_arg = args.uacqr_model if args.uacqr_model is not None else args.base_model
base_model_arg = base_model_arg.lower()
if base_model_arg in ["qnn", "neural_net"]:
    uacqr_model = "neural_net"
    base_model_label = "QNN"
    base_model_slug = "qnn"
elif base_model_arg in ["qnn_mc", "qnn-mc", "qnn_dropout", "qnn-dropout"]:
    uacqr_model = "neural_net"
    base_model_label = "QNN_MC"
    base_model_slug = "qnn_mc"
elif base_model_arg == "catboost":
    uacqr_model = "catboost"
    base_model_label = "CatBoost"
    base_model_slug = "catboost"
elif base_model_arg == "rfqr":
    uacqr_model = "rfqr"
    base_model_label = "RFQR"
    base_model_slug = "rfqr"
else:
    raise ValueError(f"Unknown base_model={base_model_arg}")
base_model_is_qnn_mc = base_model_slug == "qnn_mc"
results_tag = "".join(
    char if char.isalnum() or char == "_" else "_"
    for char in args.results_tag.strip()
)
if results_tag:
    base_model_slug = f"{base_model_slug}_{results_tag}"
scarcity_methods = expand_choice(args.scarcity_method, ["knn", "isolation_forest"])
if args.scarcity_method == "isolation_forest":
    base_model_slug = f"{base_model_slug}_scarcity_iforest"
# With both diagnostics enabled, keep the standard model folder and distinguish
# the KNN/Isolation Forest outputs in the metric filenames instead.
scarcity_result_suffixes = {
    "knn": "",
    "isolation_forest": "_isolation_forest",
}
scarcity_result_suffix = scarcity_result_suffixes.get(args.scarcity_method, "")
outlier_analysis = args.outlier_analysis
n_cores = args.n_cores
kernel = args.kernel
kernel_noise = args.kernel_noise
activation_noise = args.activation_noise
outlier_same_time = args.outlier_same_time
gamma_max = args.gamma_max
gamma_min = args.gamma_min
tau_gamma = args.tau_gamma
credo_dropout = args.credo_dropout
k_gamma = args.k_gamma
heuristic_gamma = args.heuristic_gamma
scarcity_bins = args.scarcity_bins
scarcity_k = args.scarcity_k
scarcity_heuristic = args.scarcity_heuristic
scarcity_method = args.scarcity_method
scarcity_iforest_n_estimators = args.scarcity_iforest_n_estimators
scarcity_iforest_max_samples = args.scarcity_iforest_max_samples
scarcity_iforest_max_features = args.scarcity_iforest_max_features
scarcity_iforest_bootstrap = args.scarcity_iforest_bootstrap
credo_dropout_training = args.credo_dropout_training
shared_qnn_backbone = args.shared_qnn_backbone
credo_first_qnn_backbone = args.credo_first_qnn_backbone
outlier_detector = args.outlier_detector
outlier_detectors = expand_choice(outlier_detector, ["lof", "isolation_forest"])
outlier_result_suffix = "_both" if outlier_detector == "both" else detector_suffix(outlier_detector)
outlier_contamination = args.outlier_contamination
outlier_neighbors = args.outlier_neighbors
outlier_inlier_size = args.inlier_size
outlier_tsne_components = args.tsne_components
outlier_tsne_random_state = args.tsne_random_state
iforest_n_estimators = args.iforest_n_estimators
iforest_max_samples = args.iforest_max_samples
iforest_max_features = args.iforest_max_features
iforest_bootstrap = args.iforest_bootstrap

QNN_HIDDEN_LAYERS = list(args.qnn_hidden_layers)
QNN_DROPOUT_CREDO = credo_dropout
QNN_DROPOUT_COMPETITORS = QNN_DROPOUT_CREDO if base_model_is_qnn_mc else 0.0
QNN_COMPETITOR_EPOCH_MODEL_TRACKING = base_model_is_qnn_mc

def make_competitor_qnn_params(batch_size, B=100):
    return {
        "lr": 1e-3,
        "epochs": 2000,
        "batch_size": batch_size,
        "dropout": QNN_DROPOUT_COMPETITORS,
        "epoch_model_tracking": QNN_COMPETITOR_EPOCH_MODEL_TRACKING,
        "normalize": True,
        "weight_decay": 1e-6,
        "hidden_layers": QNN_HIDDEN_LAYERS,
        "batch_norm": True,
        "gamma": 0.99,
        "step_size": 10,
        "verbose": False,
        "undo_quantile_crossing": True,
        "use_gpu": True,
        "patience": 50,
        "validation_fraction": 0.2,
        "min_saved_models": min(B + 1, 1000),
        "max_saved_models": 1000,
    }


class SharedQNNAdapter(BaseEstimator):
    """Expose the shared helper QNN through CREDO's base-model interface."""

    def __init__(self, model):
        self.model = model

    def predict(self, X, n_mc=500, use_mcdropout=True):
        ensemble_size = n_mc if use_mcdropout else None
        predictions = self.model.predict(
            X,
            ensembling=ensemble_size,
            use_mcdropout=use_mcdropout,
        )
        return predictions[0], predictions[-1]


class CredoQNNUACQRAdapter:
    """Expose the CREDO QNN and its saved training states to UACQR.

    CREDO uses MC-dropout for its own envelope. UACQR instead receives the
    deterministic predictions from the QNN states saved during training,
    matching its epoch-based neural-network ensemble protocol.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X, ensembling=None, use_mcdropout=False):
        X = np.asarray(X)
        if ensembling is None:
            lower, upper = self.model.predict(X, n_mc=1, use_mcdropout=False)
            return lower[:, 0], upper[:, 0]

        states = self.model.saved_models[-int(ensembling):]
        if len(states) < int(ensembling):
            raise RuntimeError(
                "The CREDO QNN has fewer saved training states than the "
                "UACQR ensemble requires."
            )

        current_state = deepcopy(self.model.model.state_dict())
        lower_samples = []
        upper_samples = []
        try:
            for state in states:
                self.model.model.load_state_dict(state)
                lower, upper = self.model.predict(X, n_mc=1, use_mcdropout=False)
                lower_samples.append(lower[:, 0])
                upper_samples.append(upper[:, 0])
        finally:
            self.model.model.load_state_dict(current_state)
            self.model.model.eval()

        return np.stack((np.stack(lower_samples, axis=1),
                         np.stack(upper_samples, axis=1)), axis=0)


def fit_credo_qnn_backbone(X_train, y_train, batch_size, random_state):
    """Fit the original CREDO QNN once for the inheritance pilot."""

    model = CredoQuantileRegressionNN(
        input_size=np.asarray(X_train).shape[1],
        alpha=alpha,
        dropout=QNN_DROPOUT_CREDO,
        hidden_layers=QNN_HIDDEN_LAYERS,
        use_gpu=True,
        undo_crossing=True,
    )
    model.fit(
        np.asarray(X_train),
        np.asarray(y_train),
        weight_decay=1e-6,
        scheduler_step=10,
        scheduler_gamma=0.99,
        epochs=2000,
        lr=1e-3,
        batch_size=batch_size,
        patience=50,
        verbose=0,
        split_random_state=0,
        fit_random_state=random_state,
        dropout_during_fit=credo_dropout_training,
        epoch_model_tracking=True,
        min_saved_models=101,
        max_saved_models=1000,
    )
    return model


def fit_shared_qnn(X_train, y_train, batch_size, random_state):
    """Fit the common QNN once; dropout is used for the envelope, not training."""

    params = make_competitor_qnn_params(batch_size, B=100)
    params.update(
        dropout=QNN_DROPOUT_CREDO,
        epoch_model_tracking=True,
        dropout_during_fit=False,
    )
    model = SharedQuantileRegressionNN(
        quantiles=[alpha / 2, "mean", 1 - alpha / 2],
        random_state=random_state,
        **params,
    )
    model.fit(X_train, y_train)
    return model

def generate_seeds(seed_initial, n_rep):
    np.random.seed(seed_initial)
    seeds = np.random.randint(0, 2**31 - 1, size=n_rep)
    return seeds


def resolve_iforest_max_samples(dataset, outlier_detector, iforest_max_samples):
    if (
        dataset == "WEC"
        and outlier_detector == "isolation_forest"
        and str(iforest_max_samples).lower() == "auto"
    ):
        print(
            "Using WEC-specific Isolation Forest max_samples=2048 "
            "instead of auto to better capture the large test-set geometry."
        )
        return 2048
    return iforest_max_samples


def resolve_outlier_embedding(dataset, outlier_detector, outlier_embedding="tsne"):
    if (
        dataset == "WEC"
        and outlier_detector == "isolation_forest"
        and outlier_embedding == "tsne"
    ):
        print(
            "Using WEC-specific Isolation Forest detector on standardized "
            "original features instead of the t-SNE embedding."
        )
        return "original"
    return outlier_embedding


def compute_outlier_metrics_for_detector(
        X_test,
        y_test,
        interval_map,
        detector,
        inlier_size,
        n_neighbors,
        contamination,
        tsne_random_state,
        n_components,
        outlier_embedding,
        iforest_n_estimators,
        iforest_max_samples,
        iforest_max_features,
        iforest_bootstrap,
        random_state,
):
    """Compute outlier coverage and width ratios from already-fitted intervals."""
    outlier_space = "t-SNE" if outlier_embedding == "tsne" else "standardized original feature space"
    print(f"Performing outlier detection with {outlier_space} and {detector}")
    outlier_indexes, inlier_indexes = select_outlier_inlier_indices(
        X_test,
        y_test,
        outlier_detector=detector,
        outlier_embedding=outlier_embedding,
        contamination=contamination,
        inlier_size=inlier_size,
        n_neighbors=n_neighbors,
        n_components=n_components,
        tsne_random_state=tsne_random_state,
        iforest_n_estimators=iforest_n_estimators,
        iforest_max_samples=iforest_max_samples,
        iforest_max_features=iforest_max_features,
        iforest_bootstrap=iforest_bootstrap,
        random_state=random_state,
    )

    y_test = np.asarray(y_test)
    cover_values = []
    ratio_values = []
    for method, interval in interval_map.items():
        interval = np.asarray(interval)
        finite = np.isfinite(interval[:, 0]) & np.isfinite(interval[:, 1])
        out_idx = outlier_indexes[finite[outlier_indexes]]
        in_idx = inlier_indexes[finite[inlier_indexes]]

        if len(out_idx) == 0:
            cover_values.append(np.nan)
            ratio_values.append(np.nan)
            continue

        cover_values.append(
            average_coverage(
                interval[out_idx, 1], interval[out_idx, 0], y_test[out_idx]
            )
        )
        if len(in_idx) == 0:
            ratio_values.append(np.nan)
        else:
            out_width = np.mean(compute_interval_length(
                interval[out_idx, 1], interval[out_idx, 0]
            ))
            in_width = np.mean(compute_interval_length(
                interval[in_idx, 1], interval[in_idx, 0]
            ))
            ratio_values.append(out_width / in_width if in_width != 0 else np.nan)

    return np.asarray(cover_values), np.asarray(ratio_values)

def fit_methods(
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        mdn_params,
        i,
        batch_size = 32,
        scale_y = False,
        outlier_same_time = False,
        inlier_size = 0.2,
        n_neighbors = 15,
        contamination = 0.05,
        tsne_random_state=120,
        n_components = 2,
        outlier_detector = "lof",
        outlier_embedding = "tsne",
        iforest_n_estimators = 200,
        iforest_max_samples = "auto",
        iforest_max_features = 1.0,
        iforest_bootstrap = False,
        scarcity_method = "knn",
        scarcity_iforest_n_estimators = 200,
        scarcity_iforest_max_samples = "auto",
        scarcity_iforest_max_features = 1.0,
        scarcity_iforest_bootstrap = False,
        credo_dropout_training = False,
        shared_qnn_backbone = None,
): 
    if shared_qnn_backbone is None:
        shared_qnn_backbone = globals()["shared_qnn_backbone"]

    if scale_y:
        y_scaler = StandardScaler().set_output(transform="pandas")
        y_train = y_scaler.fit_transform(y_train.to_frame())
        y_calib = y_scaler.transform(y_calib.to_frame())
        y_test = y_scaler.transform(y_test.to_frame())

    # Fitting UACQR
    print(f"Fitting UACQR")
    if uacqr_model == "rfqr":
        rfqr_params = {
        "n_estimators": 300,
        "max_features" : "sqrt",
    }
        uacqr_params = {
        "model_type": "rfqr",
        "B": 100, 
        "uacqrs_agg": "std",
        "base_model_type": "Quantile",
        }

        uacqr_results = uacqr(
        rfqr_params,
        bootstrapping_for_uacqrp=True,
        uacqrs_bagging=True,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        alpha = alpha,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=i,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
     )
    elif uacqr_model == "catboost":
        catboost_params = {
                "iterations": 1000,
                "depth": 6,
                "l2_leaf_reg": 3, 
                "random_strength": 1,
                "random_strength": 1,
                "bagging_temperature": 1,
                "thread_count": n_cores,
                # auto-tuned learning rate
        }

        uacqr_params = {
        "model_type": "catboost",
        "B": 1000,
        "uacqrs_agg": "std",
        "base_model_type": "Quantile",
        }

        uacqr_results = uacqr(
        catboost_params,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=i,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
        )
    elif uacqr_model in ["neural_net", "qnn"]:
        uacqr_params = {
        "model_type": "neural_net",
        "B": 100,
        "uacqrs_agg": "std",
        "base_model_type": "Quantile",
        }
        qnn_params = make_competitor_qnn_params(batch_size, B=uacqr_params["B"])

        uacqr_results = uacqr(
        qnn_params,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=i,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
        bootstrapping_for_uacqrp=False,
        )
    else:
        raise ValueError(f"Unknown uacqr_model={uacqr_model}")
    
    shared_qnn = None
    credo_qnn_backbone = None
    if credo_first_qnn_backbone and uacqr_model in ["neural_net", "qnn"]:
        print("Using the original CREDO QNN as the shared backbone")
        credo_qnn_backbone = fit_credo_qnn_backbone(
            X_train,
            y_train,
            batch_size,
            i,
        )
        inherited_qnn = CredoQNNUACQRAdapter(credo_qnn_backbone)
        uacqr_results.nn_model = inherited_qnn
        uacqr_results.cqr_base_model = inherited_qnn
        uacqr_results.models_B = []
        uacqr_results.modeled_quantiles = [alpha / 2, 1 - alpha / 2]
    elif shared_qnn_backbone and uacqr_model in ["neural_net", "qnn"]:
        print("Using one shared QNN backbone for UACQR, EPIC, and CREDO")
        shared_qnn = fit_shared_qnn(X_train, y_train, batch_size, i)
        uacqr_results.nn_model = shared_qnn
        uacqr_results.cqr_base_model = shared_qnn
        uacqr_results.models_B = []
        uacqr_results.modeled_quantiles = [alpha / 2, "mean", 1 - alpha / 2]
    else:
        uacqr_results.fit(X_train, y_train)
    uacqr_results.calibrate(X_calib, y_calib)
    uacqr_pred_test = uacqr_results.predict_uacqr(X_test)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_calib = X_calib.to_numpy()
    y_calib = y_calib.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Fitting EPICSCORE
    print(f"Fitting EPICSCORE")
    epic_obj = EPIC_split(
            QuantileScore,
            uacqr_results,
            alpha=alpha,
            is_fitted=True,
            base_model_type=uacqr_params["base_model_type"],
        )
    epic_obj.fit(X_train, y_train)
    epic_obj.calib(
        X_calib,
        y_calib,
        num_components=mdn_params["num_components"],
        dropout_rate=mdn_params["dropout_rate"],
        hidden_layers=mdn_params["hidden_layers"],
        patience=mdn_params["patience"],
        epochs=mdn_params["epochs"],
        normalize_y=mdn_params["normalize_y"],
        scale=mdn_params["scale"],
        batch_size=mdn_params["batch_size"],
        verbose=mdn_params["verbose"],
        type=mdn_params["type"],
        ensemble=False,
    )
    pred_epic_mdn_test = epic_obj.predict(X_test)
    del epic_obj
    gc.collect()

    print(f"Fitting vanilla CREDO with QNN")
    # Fitting CREDO with QNN
    if credo_qnn_backbone is not None:
        credal_CP_qnn = CredalCPRegressor(
            nc_type="Quantile",
            base_model=credo_qnn_backbone,
            alpha=alpha,
            adaptive_gamma=False,
            gamma=gamma,
            is_fitted=True,
        )
        credal_CP_qnn.fit(
            X_train,
            y_train,
            nn_type="MC_Dropout",
            base_model_type="QNN",
        )
    elif shared_qnn is not None:
        credal_CP_qnn = CredalCPRegressor(
            nc_type="Quantile",
            base_model=SharedQNNAdapter(shared_qnn),
            alpha=alpha,
            adaptive_gamma=False,
            gamma=gamma,
            is_fitted=True,
        )
        credal_CP_qnn.fit(
            X_train,
            y_train,
            nn_type="MC_Dropout",
            base_model_type="QNN",
        )
    else:
        credal_CP_qnn = CredalCPRegressor(
            nc_type="Quantile",
            base_model="QNN",
            alpha=alpha,
            adaptive_gamma=False,
            gamma=gamma,
        )
        credal_CP_qnn.fit(
            X_train,
            y_train,
            weight_decay=1e-6,
            step_size=10,
            gamma=0.99,
            hidden_layers=QNN_HIDDEN_LAYERS,
            dropout=QNN_DROPOUT_CREDO,
            dropout_during_fit=credo_dropout_training,
            epochs=2000,
            patience=50,
            lr=1e-3,
            batch_size=batch_size,
            verbose=1,
            random_seed_fit=i,
        )

    credal_CP_qnn.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)
    credo_CP_qnn_pred = credal_CP_qnn.predict(X_test)

    print(f"Fitting adaptive CREDO with QNN")
    if credo_qnn_backbone is not None:
        credal_CP_qnn_adaptive = CredalCPRegressor(
            nc_type="Quantile",
            base_model=credo_qnn_backbone,
            alpha=alpha,
            adaptive_gamma=True,
            gamma=gamma,
            is_fitted=True,
        )
        credal_CP_qnn_adaptive.fit(
            X_train,
            y_train,
            nn_type="MC_Dropout",
            base_model_type="QNN",
            heuristic_gamma=heuristic_gamma,
            k=k_gamma,
        )
    elif shared_qnn is not None:
        credal_CP_qnn_adaptive = CredalCPRegressor(
            nc_type="Quantile",
            base_model=SharedQNNAdapter(shared_qnn),
            alpha=alpha,
            adaptive_gamma=True,
            gamma=gamma,
            is_fitted=True,
        )
        credal_CP_qnn_adaptive.fit(
            X_train,
            y_train,
            nn_type="MC_Dropout",
            base_model_type="QNN",
            heuristic_gamma=heuristic_gamma,
            k=k_gamma,
        )
    else:
        credal_CP_qnn_adaptive = CredalCPRegressor(
            nc_type="Quantile",
            base_model="QNN",
            alpha=alpha,
            adaptive_gamma=True,
            gamma=gamma,
        )
        credal_CP_qnn_adaptive.fit(
            X_train,
            y_train,
            weight_decay=1e-6,
            step_size=10,
            gamma=0.99,
            hidden_layers=QNN_HIDDEN_LAYERS,
            dropout=QNN_DROPOUT_CREDO,
            dropout_during_fit=credo_dropout_training,
            epochs=2000,
            patience=50,
            lr=1e-3,
            batch_size=batch_size,
            verbose=1,
            random_seed_fit=i,
            heuristic_gamma=heuristic_gamma,
            k=k_gamma,
        )
    credal_CP_qnn_adaptive.calibrate(X_calib, 
                                  y_calib,
                                  N_samples_MC=n_MCMC, 
                                  gamma_max=gamma_max,
                                  gamma_min=gamma_min,
                                  tau=tau_gamma,
                                  )
    credo_CP_qnn_pred_adaptive = credal_CP_qnn_adaptive.predict(X_test)
    del credal_CP_qnn
    del credal_CP_qnn_adaptive
    gc.collect()

    lower_cqr = uacqr_pred_test["CQR"]["lower"]
    upper_cqr = uacqr_pred_test["CQR"]["upper"]
    cqr_int = np.column_stack((lower_cqr, upper_cqr))

    lower_cqrr = uacqr_pred_test["CQR-r"]["lower"]
    upper_cqrr = uacqr_pred_test["CQR-r"]["upper"]
    cqrr_int = np.column_stack((lower_cqrr, upper_cqrr))

    lower_uacqrs = uacqr_pred_test["UACQR-S"]["lower"]
    upper_uacqrs = uacqr_pred_test["UACQR-S"]["upper"]
    uacqrs_int = np.column_stack((lower_uacqrs, upper_uacqrs))

    lower_uacqrp = uacqr_pred_test["UACQR-P"]["lower"]
    upper_uacqrp = uacqr_pred_test["UACQR-P"]["upper"]
    uacqrp_int = np.column_stack((lower_uacqrp, upper_uacqrp))

    # UACQR-P can produce infinite sentinel intervals when the calibrated
    # ensemble rank exceeds all finite quantile predictions. Filter S/P
    # independently so an invalid P interval does not invalidate S.
    lower_s = np.asarray(uacqr_pred_test["UACQR-S"]["lower"])
    upper_s = np.asarray(uacqr_pred_test["UACQR-S"]["upper"])
    finite_s = np.isfinite(lower_s) & np.isfinite(upper_s)

    lower_p = np.asarray(uacqr_pred_test["UACQR-P"]["lower"])
    upper_p = np.asarray(uacqr_pred_test["UACQR-P"]["upper"])
    finite_p = np.isfinite(lower_p) & np.isfinite(upper_p)

    # unified mask: keep only indices that are finite in both methods
    good_mask = finite_s & finite_p
    n_total = good_mask.shape[0]
    n_removed = int((~good_mask).sum())
    print(f"Combined removal: excluding {n_removed} of {n_total} test points with infinite bounds in either UACQR-S or UACQR-P")

    # apply the unified filter to the stored predictions and to uacqrs_int
    y_test_uacqr = y_test[good_mask]
    uacqrs_int = uacqrs_int[good_mask]
    uacqrp_int = uacqrp_int[good_mask]

    if not outlier_same_time:
        del uacqr_results
        gc.collect()
    
    # evaluating metrics of interest
    # marginal coverage
    cover_credo_qnn = average_coverage(
        credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test
    )
    cover_credo_qnn_adaptive = average_coverage(
        credo_CP_qnn_pred_adaptive[:, 1], credo_CP_qnn_pred_adaptive[:, 0], y_test
    )
    cover_cqr = average_coverage(
        cqr_int[:, 1], cqr_int[:, 0], y_test
    )
    cover_cqrr = average_coverage(
        cqrr_int[:, 1], cqrr_int[:, 0], y_test
    )
    cover_uacqrs = average_coverage(
        uacqrs_int[:, 1], uacqrs_int[:, 0],
        y_test_uacqr
    )
    cover_uacqrp = average_coverage(
        uacqrp_int[:, 1], uacqrp_int[:, 0],
        y_test_uacqr
    )
    cover_epic_mdn = average_coverage(
        pred_epic_mdn_test[:, 1], pred_epic_mdn_test[:, 0],
        y_test
    )

    # ISL
    isl_credo_qnn = average_interval_score_loss(
        credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test, alpha
    )
    isl_credo_qnn_adaptive = average_interval_score_loss(
        credo_CP_qnn_pred_adaptive[:, 1], credo_CP_qnn_pred_adaptive[:, 0], y_test, alpha
    )
    isl_cqr = average_interval_score_loss(
        cqr_int[:, 1], cqr_int[:, 0], y_test, alpha
    )
    isl_cqrr = average_interval_score_loss(
        cqrr_int[:, 1], cqrr_int[:, 0], y_test, alpha
    )
    isl_uacqrs = average_interval_score_loss(
        uacqrs_int[:, 1], uacqrs_int[:, 0],
        y_test_uacqr, alpha
    )
    isl_uacqrp = average_interval_score_loss(
        uacqrp_int[:, 1], uacqrp_int[:, 0],
        y_test_uacqr, alpha
    )
    isl_epic_mdn = average_interval_score_loss(
        pred_epic_mdn_test[:, 1], pred_epic_mdn_test[:, 0],
        y_test, alpha
    )

    # Mean interval length and conditional-coverage diagnostics
    length_credo_qnn = average_interval_width(
        credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0]
    )
    length_credo_qnn_adaptive = average_interval_width(
        credo_CP_qnn_pred_adaptive[:, 1], credo_CP_qnn_pred_adaptive[:, 0]
    )
    length_cqr = average_interval_width(cqr_int[:, 1], cqr_int[:, 0])
    length_cqrr = average_interval_width(cqrr_int[:, 1], cqrr_int[:, 0])
    length_uacqrs = average_interval_width(uacqrs_int[:, 1], uacqrs_int[:, 0])
    length_uacqrp = average_interval_width(uacqrp_int[:, 1], uacqrp_int[:, 0])
    length_epic_mdn = average_interval_width(
        pred_epic_mdn_test[:, 1], pred_epic_mdn_test[:, 0]
    )

    if scarcity_method == "knn":
        scarcity_scores = scarcity_scores_knn(
            X_train,
            X_test,
            k=scarcity_k,
            heuristic=scarcity_heuristic,
        )
    elif scarcity_method == "isolation_forest":
        scarcity_scores = scarcity_scores_isolation_forest(
            X_train,
            X_test,
            n_estimators=scarcity_iforest_n_estimators,
            max_samples=scarcity_iforest_max_samples,
            max_features=scarcity_iforest_max_features,
            bootstrap=scarcity_iforest_bootstrap,
            random_state=i,
        )
    elif scarcity_method == "both":
        # Keep the legacy calculation below on KNN while recomputing both
        # diagnostics together after the common interval quantities exist.
        scarcity_scores_by_method = {
            "knn": scarcity_scores_knn(
                X_train, X_test, k=scarcity_k, heuristic=scarcity_heuristic
            ),
            "isolation_forest": scarcity_scores_isolation_forest(
                X_train,
                X_test,
                n_estimators=scarcity_iforest_n_estimators,
                max_samples=scarcity_iforest_max_samples,
                max_features=scarcity_iforest_max_features,
                bootstrap=scarcity_iforest_bootstrap,
                random_state=i,
            ),
        }
        scarcity_scores = scarcity_scores_by_method["knn"]
    else:
        raise ValueError(f"Unknown scarcity_method={scarcity_method}")
    _, _, scarcity_bin_edges = coverage_by_score_quantile(
        credo_CP_qnn_pred[:, 1],
        credo_CP_qnn_pred[:, 0],
        y_test,
        scarcity_scores,
        n_bins=scarcity_bins,
    )
    scarcity_scores_uacqr = scarcity_scores[good_mask]
    scarcity_cover_credo_qnn, _, _ = coverage_by_score_quantile(
        credo_CP_qnn_pred[:, 1],
        credo_CP_qnn_pred[:, 0],
        y_test,
        scarcity_scores,
        bin_edges=scarcity_bin_edges,
    )
    scarcity_cover_credo_qnn_adaptive, _, _ = coverage_by_score_quantile(
        credo_CP_qnn_pred_adaptive[:, 1],
        credo_CP_qnn_pred_adaptive[:, 0],
        y_test,
        scarcity_scores,
        bin_edges=scarcity_bin_edges,
    )
    scarcity_cover_cqr, _, _ = coverage_by_score_quantile(
        cqr_int[:, 1],
        cqr_int[:, 0],
        y_test,
        scarcity_scores,
        bin_edges=scarcity_bin_edges,
    )
    scarcity_cover_cqrr, _, _ = coverage_by_score_quantile(
        cqrr_int[:, 1],
        cqrr_int[:, 0],
        y_test,
        scarcity_scores,
        bin_edges=scarcity_bin_edges,
    )
    scarcity_cover_uacqrs, _, _ = coverage_by_score_quantile(
        uacqrs_int[:, 1],
        uacqrs_int[:, 0],
        y_test_uacqr,
        scarcity_scores_uacqr,
        bin_edges=scarcity_bin_edges,
    )
    scarcity_cover_uacqrp, _, _ = coverage_by_score_quantile(
        uacqrp_int[:, 1],
        uacqrp_int[:, 0],
        y_test_uacqr,
        scarcity_scores_uacqr,
        bin_edges=scarcity_bin_edges,
    )
    scarcity_cover_epic_mdn, _, _ = coverage_by_score_quantile(
        pred_epic_mdn_test[:, 1],
        pred_epic_mdn_test[:, 0],
        y_test,
        scarcity_scores,
        bin_edges=scarcity_bin_edges,
    )

    
    if n_removed == n_total:
      isl_uacqrs, isl_uacqrp= np.nan, np.nan
      cover_uacqrs, cover_uacqrp = np.nan, np.nan
      length_uacqrs, length_uacqrp = np.nan, np.nan
      scarcity_cover_uacqrs[:] = np.nan
      scarcity_cover_uacqrp[:] = np.nan
    
    isl_array = np.array([
        isl_credo_qnn,
        isl_credo_qnn_adaptive,
        isl_cqr,
        isl_cqrr,
        isl_uacqrs,
        isl_uacqrp,
        isl_epic_mdn,
    ])
    cover_array = np.array([
        cover_credo_qnn,
        cover_credo_qnn_adaptive,
        cover_cqr,
        cover_cqrr,
        cover_uacqrs,
        cover_uacqrp,
        cover_epic_mdn,
    ])
    length_array = np.array([
        length_credo_qnn,
        length_credo_qnn_adaptive,
        length_cqr,
        length_cqrr,
        length_uacqrs,
        length_uacqrp,
        length_epic_mdn,
    ])
    scarcity_cover_array = np.vstack([
        scarcity_cover_credo_qnn,
        scarcity_cover_credo_qnn_adaptive,
        scarcity_cover_cqr,
        scarcity_cover_cqrr,
        scarcity_cover_uacqrs,
        scarcity_cover_uacqrp,
        scarcity_cover_epic_mdn,
    ])
    scarcity_worst_cover_array = np.array([
        np.nan if np.all(np.isnan(row)) else np.nanmin(row)
        for row in scarcity_cover_array
    ])
    if scarcity_method == "both":
        scarcity_cover_arrays = {"knn": scarcity_cover_array}
        scarcity_worst_cover_arrays = {"knn": scarcity_worst_cover_array}
        score = scarcity_scores_by_method["isolation_forest"]
        _, _, if_edges = coverage_by_score_quantile(
            credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test,
            score, n_bins=scarcity_bins,
        )
        score_uacqr = score[good_mask]
        if_cover_rows = []
        for upper, lower, y_values, score_values in [
            (credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test, score),
            (credo_CP_qnn_pred_adaptive[:, 1], credo_CP_qnn_pred_adaptive[:, 0], y_test, score),
            (cqr_int[:, 1], cqr_int[:, 0], y_test, score),
            (cqrr_int[:, 1], cqrr_int[:, 0], y_test, score),
            (uacqrs_int[:, 1], uacqrs_int[:, 0], y_test_uacqr, score_uacqr),
            (uacqrp_int[:, 1], uacqrp_int[:, 0], y_test_uacqr, score_uacqr),
            (pred_epic_mdn_test[:, 1], pred_epic_mdn_test[:, 0], y_test, score),
        ]:
            if len(y_values) == 0:
                if_cover_rows.append(np.full(scarcity_bins, np.nan))
            else:
                row, _, _ = coverage_by_score_quantile(
                    upper, lower, y_values, score_values, bin_edges=if_edges
                )
                if_cover_rows.append(row)
        if_cover_array = np.vstack(if_cover_rows)
        scarcity_cover_arrays["isolation_forest"] = if_cover_array
        scarcity_worst_cover_arrays["isolation_forest"] = np.array([
            np.nan if np.all(np.isnan(row)) else np.nanmin(row)
            for row in if_cover_array
        ])
    else:
        scarcity_cover_arrays = {scarcity_method: scarcity_cover_array}
        scarcity_worst_cover_arrays = {scarcity_method: scarcity_worst_cover_array}

    # Keep the interval score auditable: SMIS is the sum of interval width and
    # the two one-sided miscoverage penalties. These means are saved per run so
    # a poor score can be attributed to width or to misses, rather than inferred
    # from coverage alone.
    def _smis_component_means(upper, lower, y_values):
        if len(y_values) == 0:
            return np.full(3, np.nan)
        width, lower_miss, upper_miss = interval_score_components(
            upper, lower, y_values, alpha
        )
        return np.array([
            np.nanmean(width),
            np.nanmean(lower_miss),
            np.nanmean(upper_miss),
        ])

    smis_components_array = np.vstack([
        _smis_component_means(credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test),
        _smis_component_means(credo_CP_qnn_pred_adaptive[:, 1], credo_CP_qnn_pred_adaptive[:, 0], y_test),
        _smis_component_means(cqr_int[:, 1], cqr_int[:, 0], y_test),
        _smis_component_means(cqrr_int[:, 1], cqrr_int[:, 0], y_test),
        _smis_component_means(uacqrs_int[:, 1], uacqrs_int[:, 0], y_test_uacqr),
        _smis_component_means(uacqrp_int[:, 1], uacqrp_int[:, 0], y_test_uacqr),
        _smis_component_means(pred_epic_mdn_test[:, 1], pred_epic_mdn_test[:, 0], y_test),
    ])

    if outlier_same_time and outlier_analysis and outlier_detector == "both":
        interval_map = {
            "credo_QNN": credo_CP_qnn_pred,
            "credo_QNN_adaptive": credo_CP_qnn_pred_adaptive,
            "cqr": cqr_int,
            "cqrr": cqrr_int,
            "uacqrs": np.column_stack((
                uacqr_pred_test["UACQR-S"]["lower"],
                uacqr_pred_test["UACQR-S"]["upper"],
            )),
            "uacqrp": np.column_stack((
                uacqr_pred_test["UACQR-P"]["lower"],
                uacqr_pred_test["UACQR-P"]["upper"],
            )),
            "EPIC": pred_epic_mdn_test,
        }
        outlier_metrics = {}
        for detector in outlier_detectors:
            detector_embedding = resolve_outlier_embedding(
                dataset, detector, outlier_embedding
            )
            outlier_metrics[detector] = compute_outlier_metrics_for_detector(
                X_test,
                y_test,
                interval_map,
                detector,
                inlier_size,
                n_neighbors,
                contamination,
                tsne_random_state,
                n_components,
                detector_embedding,
                iforest_n_estimators,
                iforest_max_samples,
                iforest_max_features,
                iforest_bootstrap,
                i,
            )
        return (
            cover_array,
            isl_array,
            length_array,
            scarcity_cover_arrays,
            scarcity_worst_cover_arrays,
            smis_components_array,
            outlier_metrics,
        )

    if outlier_same_time and outlier_analysis:
        outlier_space = "t-SNE" if outlier_embedding == "tsne" else "standardized original feature space"
        print(f"Performing outlier detection with {outlier_space} and {outlier_detector}")
        outlier_indexes, most_inlier_idxs = select_outlier_inlier_indices(
            X_test,
            y_test,
            outlier_detector=outlier_detector,
            outlier_embedding=outlier_embedding,
            contamination=contamination,
            inlier_size=inlier_size,
            n_neighbors=n_neighbors,
            n_components=n_components,
            tsne_random_state=tsne_random_state,
            iforest_n_estimators=iforest_n_estimators,
            iforest_max_samples=iforest_max_samples,
            iforest_max_features=iforest_max_features,
            iforest_bootstrap=iforest_bootstrap,
            random_state=i,
        )

        # selecting prediction intervals for inliers and outliers
        credo_qnn_outliers = credo_CP_qnn_pred[outlier_indexes]
        credo_qnn_adaptive_outliers = credo_CP_qnn_pred_adaptive[outlier_indexes]
        cqr_outliers = cqr_int[outlier_indexes]
        cqrr_outliers = cqrr_int[outlier_indexes]
        epic_mdn_outliers = pred_epic_mdn_test[outlier_indexes]
        y_test_out = y_test[outlier_indexes]

        credo_qnn_inliers = credo_CP_qnn_pred[most_inlier_idxs]
        credo_qnn_adaptive_inliers = credo_CP_qnn_pred_adaptive[most_inlier_idxs]
        cqr_inliers = cqr_int[most_inlier_idxs]
        cqrr_inliers = cqrr_int[most_inlier_idxs]
        epic_mdn_inliers = pred_epic_mdn_test[most_inlier_idxs]
        y_test_in = y_test[most_inlier_idxs]

        uacqrs_outliers = np.empty((0, 2))
        uacqrp_outliers = np.empty((0, 2))
        uacqrs_inliers = np.empty((0, 2))
        uacqrp_inliers = np.empty((0, 2))
        y_test_out_uacqrs = np.asarray([])
        y_test_out_uacqrp = np.asarray([])
        cover_uacqrs_out = np.nan
        cover_uacqrp_out = np.nan
        
        if not n_removed == n_total:
            uacqrs_outliers = uacqrs_int[outlier_indexes]
            uacqrp_outliers = uacqrp_int[outlier_indexes]
            
            uacqrs_inliers = uacqrs_int[most_inlier_idxs]
            uacqrp_inliers = uacqrp_int[most_inlier_idxs]

            # checking if there are any infinite bounds in UACQR-S or UACQR-P and removing 
            # those indices from all methods to ensure fair comparison
            lower_s = np.asarray(uacqr_pred_test["UACQR-S"]["lower"])
            upper_s = np.asarray(uacqr_pred_test["UACQR-S"]["upper"])
            finite_s = np.isfinite(lower_s) & np.isfinite(upper_s)
    
            lower_p = np.asarray(uacqr_pred_test["UACQR-P"]["lower"])
            upper_p = np.asarray(uacqr_pred_test["UACQR-P"]["upper"])
            finite_p = np.isfinite(lower_p) & np.isfinite(upper_p)

            combined_idxs = np.concatenate([outlier_indexes, most_inlier_idxs])
            finite_s_combined = finite_s[combined_idxs]
            finite_p_combined = finite_p[combined_idxs]

            n_total_combined = combined_idxs.shape[0]
            n_removed_s = int((~finite_s_combined).sum())
            n_removed_p = int((~finite_p_combined).sum())
            print(
                "UACQR finite-bound check on selected points: "
                f"UACQR-S removes {n_removed_s}/{n_total_combined}; "
                f"UACQR-P removes {n_removed_p}/{n_total_combined}"
            )

            valid_s_idxs = combined_idxs[finite_s_combined]
            valid_p_idxs = combined_idxs[finite_p_combined]

            outlier_keep_pos_s = np.isin(outlier_indexes, valid_s_idxs)
            inlier_keep_pos_s = np.isin(most_inlier_idxs, valid_s_idxs)
            outlier_keep_pos_p = np.isin(outlier_indexes, valid_p_idxs)
            inlier_keep_pos_p = np.isin(most_inlier_idxs, valid_p_idxs)

            uacqrs_outliers = uacqrs_outliers[outlier_keep_pos_s]
            y_test_out_uacqrs = y_test_out[outlier_keep_pos_s]
            uacqrs_inliers = uacqrs_inliers[inlier_keep_pos_s]

            uacqrp_outliers = uacqrp_outliers[outlier_keep_pos_p]
            y_test_out_uacqrp = y_test_out[outlier_keep_pos_p]
            uacqrp_inliers = uacqrp_inliers[inlier_keep_pos_p]

            if len(uacqrs_outliers) > 0:
                cover_uacqrs_out = average_coverage(
                    uacqrs_outliers[:, 1], uacqrs_outliers[:, 0], y_test_out_uacqrs
                )
            else:
                cover_uacqrs_out = np.nan

            if len(uacqrp_outliers) > 0:
                cover_uacqrp_out = average_coverage(
                    uacqrp_outliers[:, 1], uacqrp_outliers[:, 0], y_test_out_uacqrp
                )
            else:
                cover_uacqrp_out = np.nan

        if len(uacqrs_outliers) > 0 and len(uacqrs_inliers) > 0:
            uacqrs_ratio = np.mean(
                compute_interval_length(uacqrs_outliers[:, 1], uacqrs_outliers[:, 0])
            ) / np.mean(
                compute_interval_length(uacqrs_inliers[:, 1], uacqrs_inliers[:, 0])
            )
        else:
            uacqrs_ratio = np.nan

        if len(uacqrp_outliers) > 0 and len(uacqrp_inliers) > 0:
            uacqrp_ratio = np.mean(
                compute_interval_length(uacqrp_outliers[:, 1], uacqrp_outliers[:, 0])
            ) / np.mean(
                compute_interval_length(uacqrp_inliers[:, 1], uacqrp_inliers[:, 0])
            )
        else:
            uacqrp_ratio = np.nan
    
          
        del uacqr_results
        gc.collect()
        
        # evaluating metrics of interest
        # coverage for outliers
        cover_credo_qnn_out = average_coverage(
            credo_CP_qnn_pred[outlier_indexes][:, 1], 
            credo_CP_qnn_pred[outlier_indexes][:, 0], 
            y_test_out
        )
        cover_credo_qnn_adaptive_out = average_coverage(
            credo_CP_qnn_pred_adaptive[outlier_indexes][:, 1],
            credo_CP_qnn_pred_adaptive[outlier_indexes][:, 0],
            y_test_out
        )
        cover_cqr_out = average_coverage(
            cqr_outliers[:, 1], cqr_outliers[:, 0], y_test_out
        )
        cover_cqrr_out = average_coverage(
            cqrr_outliers[:, 1], cqrr_outliers[:, 0], y_test_out
        )
        cover_epic_mdn_out = average_coverage(
            epic_mdn_outliers[:, 1], epic_mdn_outliers[:, 0],
            y_test_out
        )
        
        # Interval length ratio
        credo_qnn_ratio = np.mean(
                compute_interval_length(
                    credo_qnn_outliers[:, 1],
                    credo_qnn_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    credo_qnn_inliers[:, 1],
                    credo_qnn_inliers[:, 0]
                )
            )
        credo_qnn_adaptive_ratio = np.mean(
                compute_interval_length(
                    credo_qnn_adaptive_outliers[:, 1],
                    credo_qnn_adaptive_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    credo_qnn_adaptive_inliers[:, 1],
                    credo_qnn_adaptive_inliers[:, 0]
                )
            )
        cqr_ratio = np.mean(
                compute_interval_length(
                    cqr_outliers[:, 1], cqr_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    cqr_inliers[:, 1], cqr_inliers[:, 0]
                )
            )
        cqrr_ratio = np.mean(
                compute_interval_length(
                    cqrr_outliers[:, 1], cqrr_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    cqrr_inliers[:, 1], cqrr_inliers[:, 0]
                )
            )
        epic_mdn_ratio = np.mean(
                compute_interval_length(
                    epic_mdn_outliers[:, 1], epic_mdn_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    epic_mdn_inliers[:, 1], epic_mdn_inliers[:, 0]
                )
            )
          
        cover_outlier_array = np.array([
            cover_credo_qnn_out,
            cover_credo_qnn_adaptive_out,
            cover_cqr_out,
            cover_cqrr_out,
            cover_uacqrs_out,
            cover_uacqrp_out,
            cover_epic_mdn_out,
        ])
        ratio_array = np.array([
            credo_qnn_ratio,
            credo_qnn_adaptive_ratio,
            cqr_ratio,
            cqrr_ratio,
            uacqrs_ratio,
            uacqrp_ratio,
            epic_mdn_ratio,
        ])

        return (
            cover_array,
            isl_array,
            length_array,
            scarcity_cover_array,
            scarcity_worst_cover_array,
            smis_components_array,
            cover_outlier_array,
            ratio_array,
        )

    return (
        cover_array,
        isl_array,
        length_array,
        scarcity_cover_array,
        scarcity_worst_cover_array,
        smis_components_array,
    )

def fit_methods_outlier(
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        mdn_params,
        i,
        batch_size = 32,
        scale_y = False,
        inlier_size = 0.2,
        n_neighbors = 15,
        contamination = 0.05,
        tsne_random_state=120,
        n_components = 2,
        outlier_detector = "lof",
        outlier_embedding = "tsne",
        iforest_n_estimators = 200,
        iforest_max_samples = "auto",
        iforest_max_features = 1.0,
        iforest_bootstrap = False,
): 
    if scale_y:
        y_scaler = StandardScaler().set_output(transform="pandas")
        y_train = y_scaler.fit_transform(y_train.to_frame())
        y_calib = y_scaler.transform(y_calib.to_frame())
        y_test = y_scaler.transform(y_test.to_frame())

    # Fitting UACQR
    print(f"Fitting UACQR")
    if uacqr_model == "rfqr":
        rfqr_params = {
        "n_estimators": 300,
        "max_features" : "sqrt",
    }
        uacqr_params = {
        "model_type": "rfqr",
        "B": 100, 
        "uacqrs_agg": "std",
        "base_model_type": "Quantile",
        }

        uacqr_results = uacqr(
        rfqr_params,
        bootstrapping_for_uacqrp=True,
        uacqrs_bagging=True,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        alpha = alpha,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=i,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
     )
    elif uacqr_model == "catboost":
        catboost_params = {
                "iterations": 1000,
                "depth": 6,
                "l2_leaf_reg": 3, 
                "random_strength": 1,
                "random_strength": 1,
                "bagging_temperature": 1,
                # auto-tuned learning rate
        }

        uacqr_params = {
        "model_type": "catboost",
        "B": 1000,
        "uacqrs_agg": "std",
        "base_model_type": "Quantile",
        }

        uacqr_results = uacqr(
        catboost_params,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=i,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
        )
    elif uacqr_model in ["neural_net", "qnn"]:
        uacqr_params = {
        "model_type": "neural_net",
        "B": 100,
        "uacqrs_agg": "std",
        "base_model_type": "Quantile",
        }
        qnn_params = make_competitor_qnn_params(batch_size, B=uacqr_params["B"])

        uacqr_results = uacqr(
        qnn_params,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=i,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
        bootstrapping_for_uacqrp=False,
        )
    else:
        raise ValueError(f"Unknown uacqr_model={uacqr_model}")
    
    uacqr_results.fit(X_train, y_train)
    uacqr_results.calibrate(X_calib, y_calib)
    uacqr_pred_test = uacqr_results.predict_uacqr(X_test)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_calib = X_calib.to_numpy()
    y_calib = y_calib.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Fitting EPICSCORE
    print(f"Fitting EPICSCORE")
    epic_obj = EPIC_split(
            QuantileScore,
            uacqr_results,
            alpha=alpha,
            is_fitted=True,
            base_model_type=uacqr_params["base_model_type"],
        )
    epic_obj.fit(X_train, y_train)
    epic_obj.calib(
        X_calib,
        y_calib,
        num_components=mdn_params["num_components"],
        dropout_rate=mdn_params["dropout_rate"],
        hidden_layers=mdn_params["hidden_layers"],
        patience=mdn_params["patience"],
        epochs=mdn_params["epochs"],
        normalize_y=mdn_params["normalize_y"],
        scale=mdn_params["scale"],
        batch_size=mdn_params["batch_size"],
        verbose=mdn_params["verbose"],
        type=mdn_params["type"],
        ensemble=False,
    )
    pred_epic_mdn_test = epic_obj.predict(X_test)
    del epic_obj
    gc.collect()

    print(f"Fitting vanilla CREDO with QNN")
    # Fitting CREDO with QNN
    credal_CP_qnn = CredalCPRegressor(
        nc_type = 'Quantile',
        base_model = "QNN",
        alpha = alpha,
        adaptive_gamma = False,
        gamma = gamma,
        )
    
    credal_CP_qnn.fit(
        X_train, 
        y_train,
        weight_decay=1e-6,
        step_size=10,
        gamma=0.99,
        hidden_layers=QNN_HIDDEN_LAYERS,
        dropout=QNN_DROPOUT_CREDO,
        epochs=2000,
        patience=50,
        lr=1e-3, 
        batch_size=batch_size,
        verbose=1,
        random_seed_fit=i,
    )

    credal_CP_qnn.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)
    credo_CP_qnn_pred = credal_CP_qnn.predict(X_test)

    print(f"Fitting adaptive CREDO with QNN")
    credal_CP_qnn_adaptive = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "QNN",
    alpha = alpha,
    adaptive_gamma = True,
    gamma = gamma,
    is_fitted = True,
    )

    credal_CP_qnn_adaptive.fit(
         X_train, 
        y_train,
        weight_decay=1e-6,
        step_size=5,
        gamma=0.99,
        hidden_layers=QNN_HIDDEN_LAYERS,
        dropout=QNN_DROPOUT_CREDO,
        epochs=2000,
        patience=50,
        lr=1e-3, 
        batch_size=batch_size,
        verbose=1,
        random_seed_fit=i,
        heuristic_gamma=heuristic_gamma,
        k=k_gamma,
    )
    credal_CP_qnn_adaptive.calibrate(X_calib, 
                                  y_calib,
                                  N_samples_MC=n_MCMC, 
                                  gamma_max=gamma_max,
                                  gamma_min=gamma_min,
                                  tau=tau_gamma,
                                  )
    credo_CP_qnn_pred_adaptive = credal_CP_qnn_adaptive.predict(X_test)
    del credal_CP_qnn
    del credal_CP_qnn_adaptive
    gc.collect()


    lower_cqr = uacqr_pred_test["CQR"]["lower"]
    upper_cqr = uacqr_pred_test["CQR"]["upper"]
    cqr_int = np.column_stack((lower_cqr, upper_cqr))

    lower_cqrr = uacqr_pred_test["CQR-r"]["lower"]
    upper_cqrr = uacqr_pred_test["CQR-r"]["upper"]
    cqrr_int = np.column_stack((lower_cqrr, upper_cqrr))

    lower_uacqrs = uacqr_pred_test["UACQR-S"]["lower"]
    upper_uacqrs = uacqr_pred_test["UACQR-S"]["upper"]
    uacqrs_int = np.column_stack((lower_uacqrs, upper_uacqrs))

    lower_uacqrp = uacqr_pred_test["UACQR-P"]["lower"]
    upper_uacqrp = uacqr_pred_test["UACQR-P"]["upper"]
    uacqrp_int = np.column_stack((lower_uacqrp, upper_uacqrp))

    outlier_space = "t-SNE" if outlier_embedding == "tsne" else "standardized original feature space"
    print(f"Performing outlier detection with {outlier_space} and {outlier_detector}")
    outlier_indexes, most_inlier_idxs = select_outlier_inlier_indices(
        X_test,
        y_test,
        outlier_detector=outlier_detector,
        outlier_embedding=outlier_embedding,
        contamination=contamination,
        inlier_size=inlier_size,
        n_neighbors=n_neighbors,
        n_components=n_components,
        tsne_random_state=tsne_random_state,
        iforest_n_estimators=iforest_n_estimators,
        iforest_max_samples=iforest_max_samples,
        iforest_max_features=iforest_max_features,
        iforest_bootstrap=iforest_bootstrap,
        random_state=i,
    )

    # selecting prediction intervals for inliers and outliers
    credo_qnn_outliers = credo_CP_qnn_pred[outlier_indexes]
    credo_qnn_adaptive_outliers = credo_CP_qnn_pred_adaptive[outlier_indexes]
    cqr_outliers = cqr_int[outlier_indexes]
    cqrr_outliers = cqrr_int[outlier_indexes]
    uacqrs_outliers = uacqrs_int[outlier_indexes]
    uacqrp_outliers = uacqrp_int[outlier_indexes]
    epic_mdn_outliers = pred_epic_mdn_test[outlier_indexes]
    y_test_out = y_test[outlier_indexes]

    credo_qnn_inliers = credo_CP_qnn_pred[most_inlier_idxs]
    credo_qnn_adaptive_inliers = credo_CP_qnn_pred_adaptive[most_inlier_idxs]
    cqr_inliers = cqr_int[most_inlier_idxs]
    cqrr_inliers = cqrr_int[most_inlier_idxs]
    uacqrs_inliers = uacqrs_int[most_inlier_idxs]
    uacqrp_inliers = uacqrp_int[most_inlier_idxs]
    epic_mdn_inliers = pred_epic_mdn_test[most_inlier_idxs]
    y_test_in = y_test[most_inlier_idxs]

    # checking if there are any infinite bounds in UACQR-S or UACQR-P and removing 
    # those indices from all methods to ensure fair comparison
    lower_s = np.asarray(uacqr_pred_test["UACQR-S"]["lower"])
    upper_s = np.asarray(uacqr_pred_test["UACQR-S"]["upper"])
    finite_s = np.isfinite(lower_s) & np.isfinite(upper_s)

    lower_p = np.asarray(uacqr_pred_test["UACQR-P"]["lower"])
    upper_p = np.asarray(uacqr_pred_test["UACQR-P"]["upper"])
    finite_p = np.isfinite(lower_p) & np.isfinite(upper_p)

    combined_idxs = np.concatenate([outlier_indexes, most_inlier_idxs])
    finite_s_combined = finite_s[combined_idxs]
    finite_p_combined = finite_p[combined_idxs]

    n_total_combined = combined_idxs.shape[0]
    n_removed_s = int((~finite_s_combined).sum())
    n_removed_p = int((~finite_p_combined).sum())
    print(
        "UACQR finite-bound check on selected points: "
        f"UACQR-S removes {n_removed_s}/{n_total_combined}; "
        f"UACQR-P removes {n_removed_p}/{n_total_combined}"
    )

    valid_s_idxs = combined_idxs[finite_s_combined]
    valid_p_idxs = combined_idxs[finite_p_combined]

    outlier_keep_pos_s = np.isin(outlier_indexes, valid_s_idxs)
    inlier_keep_pos_s = np.isin(most_inlier_idxs, valid_s_idxs)
    outlier_keep_pos_p = np.isin(outlier_indexes, valid_p_idxs)
    inlier_keep_pos_p = np.isin(most_inlier_idxs, valid_p_idxs)

    uacqrs_outliers = uacqrs_outliers[outlier_keep_pos_s]
    y_test_out_uacqrs = y_test_out[outlier_keep_pos_s]
    uacqrs_inliers = uacqrs_inliers[inlier_keep_pos_s]

    uacqrp_outliers = uacqrp_outliers[outlier_keep_pos_p]
    y_test_out_uacqrp = y_test_out[outlier_keep_pos_p]
    uacqrp_inliers = uacqrp_inliers[inlier_keep_pos_p]
    del uacqr_results
    gc.collect()
    
    # evaluating metrics of interest
    # coverage for outliers
    cover_credo_qnn_out = average_coverage(
        credo_CP_qnn_pred[outlier_indexes][:, 1], 
        credo_CP_qnn_pred[outlier_indexes][:, 0], 
        y_test_out
    )
    cover_credo_qnn_adaptive_out = average_coverage(
        credo_CP_qnn_pred_adaptive[outlier_indexes][:, 1],
        credo_CP_qnn_pred_adaptive[outlier_indexes][:, 0],
        y_test_out
    )
    cover_cqr_out = average_coverage(
        cqr_outliers[:, 1], cqr_outliers[:, 0], y_test_out
    )
    cover_cqrr_out = average_coverage(
        cqrr_outliers[:, 1], cqrr_outliers[:, 0], y_test_out
    )
    if len(uacqrs_outliers) > 0:
        cover_uacqrs_out = average_coverage(
            uacqrs_outliers[:, 1], uacqrs_outliers[:, 0],
            y_test_out_uacqrs
        )
    else:
        cover_uacqrs_out = np.nan
    if len(uacqrp_outliers) > 0:
        cover_uacqrp_out = average_coverage(
            uacqrp_outliers[:, 1], uacqrp_outliers[:, 0],
            y_test_out_uacqrp
        )
    else:
        cover_uacqrp_out = np.nan
    cover_epic_mdn_out = average_coverage(
        epic_mdn_outliers[:, 1], epic_mdn_outliers[:, 0],
        y_test_out
    )

    # Interval length ratio
    credo_qnn_ratio = np.mean(
            compute_interval_length(
                credo_qnn_outliers[:, 1],
                credo_qnn_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                credo_qnn_inliers[:, 1],
                credo_qnn_inliers[:, 0]
            )
        )
    credo_qnn_adaptive_ratio = np.mean(
            compute_interval_length(
                credo_qnn_adaptive_outliers[:, 1],
                credo_qnn_adaptive_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                credo_qnn_adaptive_inliers[:, 1],
                credo_qnn_adaptive_inliers[:, 0]
            )
        )
    cqr_ratio = np.mean(
            compute_interval_length(
                cqr_outliers[:, 1], cqr_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                cqr_inliers[:, 1], cqr_inliers[:, 0]
            )
        )
    cqrr_ratio = np.mean(
            compute_interval_length(
                cqrr_outliers[:, 1], cqrr_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                cqrr_inliers[:, 1], cqrr_inliers[:, 0]
            )
        )
    epic_mdn_ratio = np.mean(
            compute_interval_length(
                epic_mdn_outliers[:, 1], epic_mdn_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                epic_mdn_inliers[:, 1], epic_mdn_inliers[:, 0]
            )
        )
    if len(uacqrs_outliers) > 0 and len(uacqrs_inliers) > 0:
        uacqrs_ratio = np.mean(
                compute_interval_length(
                    uacqrs_outliers[:, 1], uacqrs_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    uacqrs_inliers[:, 1], uacqrs_inliers[:, 0]
                )
            )
    else:
        uacqrs_ratio = np.nan

    if len(uacqrp_outliers) > 0 and len(uacqrp_inliers) > 0:
        uacqrp_ratio = np.mean(
                compute_interval_length(
                    uacqrp_outliers[:, 1], uacqrp_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    uacqrp_inliers[:, 1], uacqrp_inliers[:, 0]
                )
            )
    else:
        uacqrp_ratio = np.nan
    
    cover_array = np.array([
        cover_credo_qnn_out,
        cover_credo_qnn_adaptive_out,
        cover_cqr_out,
        cover_cqrr_out,
        cover_uacqrs_out,
        cover_uacqrp_out,
        cover_epic_mdn_out,
    ])
    ratio_array = np.array([
        credo_qnn_ratio,
        credo_qnn_adaptive_ratio,
        cqr_ratio,
        cqrr_ratio,
        uacqrs_ratio,
        uacqrp_ratio,
        epic_mdn_ratio,
    ])

    return cover_array, ratio_array

def run_experiment_outlier(
    dataset,
    n_rep,
    target_column,
    prop_test = 0.2,
    inlier_size=0.2,
    contamination=0.05,
    n_neighbors=15,
    n_components=2,
    tsne_random_state=120,
    seed_initial=125,
    checkpoint_flag = False,
    checkpoint_data = None,
    scale_y = False,
    outlier_detector = "lof",
    outlier_embedding = "tsne",
    iforest_n_estimators = 200,
    iforest_max_samples = "auto",
    iforest_max_features = 1.0,
    iforest_bootstrap = False,
):
    data = pd.read_csv(os.path.join(DATA_PATH, f"{dataset}.csv"))

    # EPICSCORE params
    mdn_params = {
    "num_components": 5,
    "dropout_rate": 0.5,
    "epistemic_model": "MC_dropout",
    "hidden_layers": [64, 64],
    "patience": 50,
    "epochs": 2000,
    "scale": True,
    "batch_size": 40,
    "normalize_y": True,
    "verbose": 0,
    "type": "gaussian",
    }

    batch_size = 32

    if data.shape[0] > 10000:
        mdn_params["batch_size"] = 120
        batch_size = 125
    if dataset == "WEC":
        mdn_params["batch_size"] = 250
        batch_size = 250
    iforest_max_samples = resolve_iforest_max_samples(
        dataset,
        outlier_detector,
        iforest_max_samples,
    )
    if outlier_detector == "isolation_forest":
        outlier_embedding = resolve_outlier_embedding(
            dataset,
            outlier_detector,
            outlier_embedding,
        )

    if checkpoint_flag:
        resume_from = int(checkpoint_data.get("iteration", -1)) + 1
        ratio_results = checkpoint_data.get("ratio_results", [])
        coverage_results = checkpoint_data.get("coverage_results", [])
        seeds = checkpoint_data.get("seeds", None)
        print(f"Resuming from iteration {resume_from}. Loaded {len(coverage_results)} results so far.")
    else:
        resume_from = 0
        seeds = generate_seeds(seed_initial, n_rep)
        coverage_results = []
        ratio_results = []

    for i in tqdm(range(resume_from, n_rep), desc = f"Running methods for dataset: {dataset}"):
        print(f"Repetition {i+1}/{n_rep}")
        seed = seeds[i]
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train_calib, X_test, y_train_calib, y_test = train_test_split(
        X, y, test_size=prop_test, random_state=seed
    )
        prop_train = 0.7
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_calib, y_train_calib, test_size=1-prop_train, random_state=seed
        )

        if dataset in ["blog"]:
            scale_y_current = True
        else:
            scale_y_current = scale_y

        cover_array, ratio_array = fit_methods_outlier(
            X_train,
            y_train,
            X_calib,
            y_calib,
            X_test,
            y_test,
            mdn_params,
            i,
            batch_size = batch_size,
            scale_y = scale_y_current,
            inlier_size = inlier_size,
            n_neighbors = n_neighbors,
            contamination = contamination,
            tsne_random_state=tsne_random_state,
            n_components = n_components,
            outlier_detector = outlier_detector,
            outlier_embedding = outlier_embedding,
            iforest_n_estimators = iforest_n_estimators,
            iforest_max_samples = iforest_max_samples,
            iforest_max_features = iforest_max_features,
            iforest_bootstrap = iforest_bootstrap,
        )
        coverage_results.append(cover_array)
        ratio_results.append(ratio_array)

        def save_checkpoint(iteration, seeds):
            try:
                checkpoint = {
                    "coverage_results": coverage_results,
                    "ratio_results": ratio_results,
                    "iteration": iteration,
                    "seeds": seeds,
                    "alpha": alpha,
                    "gamma": gamma,
                    "dataset": dataset,
                    "base_model": base_model_label,
                    "outlier_detector": outlier_detector,
                }
                chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
                os.makedirs(chk_dir, exist_ok=True)
                filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}_outlier{outlier_result_suffix}.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"Failed saving checkpoint at iter {iteration+1}: {e}")

        # save checkpoint after each repetition
        save_checkpoint(i, seeds)
        # summarize results: convert lists to arrays and compute mean and sd (sample sd if n_rep>1)
    coverage_results = np.array(coverage_results)
    ratio_results = np.array(ratio_results)

    def mean_sd(arr):
        mean = np.nanmean(arr, axis=0, )
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        return mean, sd

    methods = [
        "credo_QNN",
        "credo_QNN_adaptive",
        "cqr",
        "cqrr",
        "uacqrs",
        "uacqrp",
        "EPIC"]

    cover_mean, cover_sd = mean_sd(coverage_results)
    ratio_mean, ratio_sd = mean_sd(ratio_results)

    # create summary dataframes and save to CSV
    df_cover = pd.DataFrame({"base_model": base_model_label, "methods": methods ,"mean": cover_mean, "sd": cover_sd})
    df_ratio = pd.DataFrame({"base_model": base_model_label, "methods": methods ,"mean": ratio_mean, "sd": ratio_sd})

    data_dir = os.path.join(RESULTS_PATH, f"{dataset}_{base_model_slug}_summary")
    os.makedirs(data_dir, exist_ok=True)

    suffix = detector_suffix(outlier_detector)
    df_cover.to_csv(os.path.join(data_dir, f"{dataset}_coverage_outlier{suffix}_summary.csv"))
    df_ratio.to_csv(os.path.join(data_dir, f"{dataset}_ratio_outlier{suffix}_summary.csv"))
    return np.array(coverage_results), np.array(ratio_results)

def run_experiment(dataset, 
                   n_rep, 
                   target_column, 
                   prop_test = 0.2,
                   checkpoint_flag = False,
                   checkpoint_data = None,
                   checkpoint_data_outlier = None,
                   outlier_same_time = False,
                   outlier_analysis = False,
                   outlier_detector = "lof",
                   outlier_embedding = "tsne",
                   inlier_size = 0.2,
                   contamination = 0.05,
                   n_neighbors = 15,
                   n_components = 2,
                   tsne_random_state = 120,
                   iforest_n_estimators = 200,
                   iforest_max_samples = "auto",
                   iforest_max_features = 1.0,
                   iforest_bootstrap = False,
):
    data = pd.read_csv(os.path.join(DATA_PATH, f"{dataset}.csv"))

    # EPICSCORE params
    mdn_params = {
    "num_components": 5,
    "dropout_rate": 0.5,
    "epistemic_model": "MC_dropout",
    "hidden_layers": [64, 64],
    "patience": 50,
    "epochs": 2000,
    "scale": True,
    "batch_size": 40,
    "normalize_y": True,
    "verbose": 0,
    "type": "gaussian",
    }
    
    batch_size = 32
    
    if data.shape[0] > 10000:
        mdn_params["batch_size"] = 120
        batch_size = 125
    if dataset == "WEC":
        mdn_params["batch_size"] = 250
        batch_size = 250
    if outlier_analysis:
        if "isolation_forest" in outlier_detectors:
            if dataset == "WEC" and str(iforest_max_samples).lower() == "auto":
                iforest_max_samples = 2048
    

    if checkpoint_flag:
        local_diagnostics_restart = False
        resume_from = int(checkpoint_data.get("iteration", -1)) + 1
        cover_results = checkpoint_data.get("cover_results", [])
        isl_results = checkpoint_data.get("isl_results", [])
        length_results = checkpoint_data.get("length_results", [])
        smis_components_results = checkpoint_data.get("smis_components_results", None)
        scarcity_coverage_results = checkpoint_data.get("scarcity_coverage_results", None)
        scarcity_worst_coverage_results = checkpoint_data.get("scarcity_worst_coverage_results", None)
        seeds = checkpoint_data.get("seeds", None)
        if (
            scarcity_coverage_results is None
            or scarcity_worst_coverage_results is None
        ):
            print("Checkpoint does not include local coverage diagnostics. Restarting this run to keep summaries aligned.")
            local_diagnostics_restart = True
            resume_from = 0
            cover_results = []
            isl_results = []
            length_results = []
            smis_components_results = []
            scarcity_coverage_results = {method: [] for method in scarcity_methods}
            scarcity_worst_coverage_results = {method: [] for method in scarcity_methods}
            if seeds is None:
                seeds = generate_seeds(seed_initial, n_rep)
        print(f"Resuming from iteration {resume_from}. Loaded {len(cover_results)} results so far.")
        if outlier_same_time and outlier_analysis:
            if local_diagnostics_restart:
                ratio_results = {detector: [] for detector in outlier_detectors}
                coverage_outlier_results = {detector: [] for detector in outlier_detectors}
            else:
                resume_from = int(checkpoint_data_outlier.get("iteration", -1)) + 1
                ratio_results = checkpoint_data_outlier.get("ratio_results", [])
                coverage_outlier_results = checkpoint_data_outlier.get("coverage_results", [])
                if not isinstance(ratio_results, dict):
                    ratio_results = {outlier_detector: ratio_results}
                    coverage_outlier_results = {outlier_detector: coverage_outlier_results}
                seeds = checkpoint_data_outlier.get("seeds", None)
            print(f"Resuming from iteration {resume_from}. Loaded {len(coverage_outlier_results)} results so far.")

        if smis_components_results is None:
            print("Checkpoint does not include SMIS components. Restarting to collect width/miscoverage diagnostics.")
            resume_from = 0
            cover_results = []
            isl_results = []
            length_results = []
            smis_components_results = []
            scarcity_coverage_results = {method: [] for method in scarcity_methods}
            scarcity_worst_coverage_results = {method: [] for method in scarcity_methods}
            if outlier_same_time and outlier_analysis:
                ratio_results = {detector: [] for detector in outlier_detectors}
                coverage_outlier_results = {detector: [] for detector in outlier_detectors}

    else:
        resume_from = 0
        seeds = generate_seeds(seed_initial, n_rep)
        cover_results = []
        isl_results = []
        length_results = []
        smis_components_results = []
        scarcity_coverage_results = {method: [] for method in scarcity_methods}
        scarcity_worst_coverage_results = {method: [] for method in scarcity_methods}
        if outlier_same_time and outlier_analysis:
            ratio_results = {detector: [] for detector in outlier_detectors}
            coverage_outlier_results = {detector: [] for detector in outlier_detectors}

    for i in tqdm(range(resume_from, n_rep), desc = f"Running methods for dataset: {dataset}"):
        print(f"Repetition {i+1}/{n_rep}")
        seed = seeds[i]
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train_calib, X_test, y_train_calib, y_test = train_test_split(
            X, y, test_size=prop_test, random_state=seed
        )
        
        prop_train = 0.7
        
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_calib, y_train_calib, test_size=1-prop_train, random_state=seed
        )

        if dataset in ["blog"]:
            scale_y = True
        else:
            scale_y = False

        if not outlier_same_time:
            cover_array, isl_array, length_array, scarcity_cover_array, scarcity_worst_cover_array, smis_components_array = fit_methods(
                X_train,
                y_train,
                X_calib,
                y_calib,
                X_test,
                y_test,
                mdn_params,
                i,
                batch_size = batch_size,
                scale_y = scale_y,
                scarcity_method = scarcity_method,
                scarcity_iforest_n_estimators = scarcity_iforest_n_estimators,
                scarcity_iforest_max_samples = scarcity_iforest_max_samples,
                scarcity_iforest_max_features = scarcity_iforest_max_features,
                scarcity_iforest_bootstrap = scarcity_iforest_bootstrap,
                credo_dropout_training = credo_dropout_training,
            )
            cover_results.append(cover_array)
            isl_results.append(isl_array)
            length_results.append(length_array)
            smis_components_results.append(smis_components_array)
            if not isinstance(scarcity_cover_array, dict):
                scarcity_cover_array = {scarcity_method: scarcity_cover_array}
                scarcity_worst_cover_array = {scarcity_method: scarcity_worst_cover_array}
            for method in scarcity_methods:
                scarcity_coverage_results[method].append(scarcity_cover_array[method])
                scarcity_worst_coverage_results[method].append(scarcity_worst_cover_array[method])

            def save_checkpoint(iteration, seeds):
                try:
                    checkpoint = {
                        "cover_results": cover_results,
                        "isl_results": isl_results,
                        "length_results": length_results,
                        "smis_components_results": smis_components_results,
                        "scarcity_coverage_results": scarcity_coverage_results,
                        "scarcity_worst_coverage_results": scarcity_worst_coverage_results,
                        "iteration": iteration,
                        "seeds": seeds,
                        "alpha": alpha,
                        "gamma": gamma,
                        "dataset": dataset,
                        "base_model": base_model_label,
                        "shared_qnn_backbone": shared_qnn_backbone,
                        "credo_first_qnn_backbone": credo_first_qnn_backbone,
                        "credo_dropout_training": credo_dropout_training,
                        "scarcity_method": scarcity_method,
                        "outlier_detector": outlier_detector,
                        "protocol_version": 3,
                    }
                    chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
                    os.makedirs(chk_dir, exist_ok=True)
                    filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}.pkl")
                    with open(filepath, "wb") as f:
                        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Failed saving checkpoint at iter {iteration+1}: {e}")

            # save checkpoint after each repetition
            save_checkpoint(i, seeds)
        elif outlier_same_time and outlier_analysis:
            cover_array, isl_array, length_array, scarcity_cover_array, scarcity_worst_cover_array, smis_components_array, outlier_metrics = fit_methods(
            X_train,
            y_train, 
            X_calib, 
            y_calib, 
            X_test, 
            y_test,
            mdn_params, 
            i, 
            batch_size = batch_size,
            scale_y = scale_y,
            scarcity_method = scarcity_method,
            scarcity_iforest_n_estimators = scarcity_iforest_n_estimators,
            scarcity_iforest_max_samples = scarcity_iforest_max_samples,
            scarcity_iforest_max_features = scarcity_iforest_max_features,
            scarcity_iforest_bootstrap = scarcity_iforest_bootstrap,
            credo_dropout_training = credo_dropout_training,
            outlier_same_time = outlier_same_time, 
            inlier_size = inlier_size,
            n_neighbors = n_neighbors,
            contamination = contamination,
            tsne_random_state = tsne_random_state,
            n_components = n_components,
            outlier_detector = outlier_detector,
            outlier_embedding = outlier_embedding,
            iforest_n_estimators = iforest_n_estimators,
            iforest_max_samples = iforest_max_samples,
            iforest_max_features = iforest_max_features,
            iforest_bootstrap = iforest_bootstrap,
            )
            cover_results.append(cover_array)
            isl_results.append(isl_array)
            length_results.append(length_array)
            smis_components_results.append(smis_components_array)
            if not isinstance(scarcity_cover_array, dict):
                scarcity_cover_array = {scarcity_method: scarcity_cover_array}
                scarcity_worst_cover_array = {scarcity_method: scarcity_worst_cover_array}
            for method in scarcity_methods:
                scarcity_coverage_results[method].append(scarcity_cover_array[method])
                scarcity_worst_coverage_results[method].append(scarcity_worst_cover_array[method])
            for detector in outlier_detectors:
                coverage_outlier_results[detector].append(outlier_metrics[detector][0])
                ratio_results[detector].append(outlier_metrics[detector][1])

            def save_checkpoint(iteration, seeds):
                try:
                    checkpoint = {
                        "cover_results": cover_results,
                        "isl_results": isl_results,
                        "length_results": length_results,
                        "smis_components_results": smis_components_results,
                        "scarcity_coverage_results": scarcity_coverage_results,
                        "scarcity_worst_coverage_results": scarcity_worst_coverage_results,
                        "iteration": iteration,
                        "seeds": seeds,
                        "alpha": alpha,
                        "gamma": gamma,
                        "dataset": dataset,
                        "base_model": base_model_label,
                        "outlier_detector": outlier_detector,
                        "shared_qnn_backbone": shared_qnn_backbone,
                        "credo_first_qnn_backbone": credo_first_qnn_backbone,
                        "credo_dropout_training": credo_dropout_training,
                        "scarcity_method": scarcity_method,
                        "protocol_version": 3,
                    }
                    chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
                    os.makedirs(chk_dir, exist_ok=True)
                    filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}.pkl")
                    with open(filepath, "wb") as f:
                        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
               
                    checkpoint_outlier = {
                        "coverage_results": coverage_outlier_results,
                        "ratio_results": ratio_results,
                        "iteration": iteration,
                        "seeds": seeds,
                        "alpha": alpha,
                        "gamma": gamma,
                        "dataset": dataset,
                        "base_model": base_model_label,
                        "outlier_detector": outlier_detector,
                        "shared_qnn_backbone": shared_qnn_backbone,
                        "credo_first_qnn_backbone": credo_first_qnn_backbone,
                        "credo_dropout_training": credo_dropout_training,
                        "scarcity_method": scarcity_method,
                        "protocol_version": 3,
                    }
                    filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}_outlier{outlier_result_suffix}.pkl")
                    with open(filepath, "wb") as f:
                        pickle.dump(checkpoint_outlier, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
                except Exception as e:
                    print(f"Failed saving checkpoint at iter {iteration+1}: {e}")
            save_checkpoint(i, seeds)

    
    # summarize results: convert lists to arrays and compute mean and sd (sample sd if n_rep>1)
    cover_results = np.array(cover_results)
    isl_results = np.array(isl_results)
    length_results = np.array(length_results)
    smis_components_results = np.array(smis_components_results)

    def mean_sd(arr):
        mean = np.nanmean(arr, axis=0, )
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        return mean, sd

    methods = [
        "credo_QNN",
        "credo_QNN_adaptive",
        "cqr",
        "cqrr",
        "uacqrs",
        "uacqrp",
        "EPIC"]

    cover_mean, cover_sd = mean_sd(cover_results)
    isl_mean, isl_sd = mean_sd(isl_results)
    length_mean, length_sd = mean_sd(length_results)
    smis_components_mean, smis_components_sd = mean_sd(smis_components_results)

    # create summary dataframes and save to CSV
    df_cover = pd.DataFrame({"base_model": base_model_label, "methods": methods ,"mean": cover_mean, "sd": cover_sd})
    df_isl = pd.DataFrame({"base_model": base_model_label, "methods": methods ,"mean": isl_mean, "sd": isl_sd})
    df_length = pd.DataFrame({"base_model": base_model_label, "methods": methods ,"mean": length_mean, "sd": length_sd})
    df_smis_components = pd.DataFrame({
        "base_model": base_model_label,
        "methods": methods,
        "width_mean": smis_components_mean[:, 0],
        "width_sd": smis_components_sd[:, 0],
        "lower_miscoverage_mean": smis_components_mean[:, 1],
        "lower_miscoverage_sd": smis_components_sd[:, 1],
        "upper_miscoverage_mean": smis_components_mean[:, 2],
        "upper_miscoverage_sd": smis_components_sd[:, 2],
        "smis_mean": np.sum(smis_components_mean, axis=1),
    })
    data_dir = os.path.join(RESULTS_PATH, f"{dataset}_{base_model_slug}_summary")
    os.makedirs(data_dir, exist_ok=True)

    df_cover.to_csv(os.path.join(data_dir, f"{dataset}_coverage_summary.csv"))
    df_isl.to_csv(os.path.join(data_dir, f"{dataset}_isl_summary.csv"))
    df_length.to_csv(os.path.join(data_dir, f"{dataset}_interval_length_summary.csv"))
    df_smis_components.to_csv(os.path.join(data_dir, f"{dataset}_smis_components_summary.csv"))

    for scarcity_name in scarcity_methods:
        scarcity_coverage_array = np.asarray(scarcity_coverage_results[scarcity_name])
        scarcity_worst_array = np.asarray(scarcity_worst_coverage_results[scarcity_name])
        scarcity_worst_mean, scarcity_worst_sd = mean_sd(scarcity_worst_array)
        df_scarcity_worst = pd.DataFrame({
            "base_model": base_model_label,
            "methods": methods,
            "mean": scarcity_worst_mean,
            "sd": scarcity_worst_sd,
            "scarcity_method": scarcity_name,
            "n_bins": scarcity_bins,
            "scarcity_k": scarcity_k,
            "scarcity_heuristic": scarcity_heuristic,
            "scarcity_iforest_n_estimators": scarcity_iforest_n_estimators,
            "scarcity_iforest_max_samples": scarcity_iforest_max_samples,
            "scarcity_iforest_max_features": scarcity_iforest_max_features,
            "scarcity_iforest_bootstrap": scarcity_iforest_bootstrap,
        })
        scarcity_mean, scarcity_sd = mean_sd(scarcity_coverage_array)
        scarcity_rows = []
        for method_idx, method in enumerate(methods):
            for bin_idx in range(scarcity_mean.shape[1]):
                scarcity_rows.append({
                    "base_model": base_model_label,
                    "methods": method,
                    "scarcity_bin": f"Q{bin_idx + 1}",
                    "mean": scarcity_mean[method_idx, bin_idx],
                    "sd": scarcity_sd[method_idx, bin_idx],
                    "scarcity_method": scarcity_name,
                    "n_bins": scarcity_bins,
                    "scarcity_k": scarcity_k,
                    "scarcity_heuristic": scarcity_heuristic,
                    "scarcity_iforest_n_estimators": scarcity_iforest_n_estimators,
                    "scarcity_iforest_max_samples": scarcity_iforest_max_samples,
                    "scarcity_iforest_max_features": scarcity_iforest_max_features,
                    "scarcity_iforest_bootstrap": scarcity_iforest_bootstrap,
                })
        suffix = scarcity_result_suffixes[scarcity_name]
        df_scarcity_worst.to_csv(os.path.join(data_dir, f"{dataset}_scarcity_worst_coverage{suffix}_summary.csv"))
        pd.DataFrame(scarcity_rows).to_csv(os.path.join(data_dir, f"{dataset}_scarcity_coverage{suffix}_summary.csv"))

    if outlier_same_time and outlier_analysis:
        print("Saving outliers df")
        for detector in outlier_detectors:
            detector_cover = np.asarray(coverage_outlier_results[detector])
            detector_ratio = np.asarray(ratio_results[detector])
            cover_mean_outlier, cover_sd_outlier = mean_sd(detector_cover)
            ratio_mean_outlier, ratio_sd_outlier = mean_sd(detector_ratio)
            df_cover_out = pd.DataFrame({"base_model": base_model_label, "methods": methods ,"mean": cover_mean_outlier, "sd": cover_sd_outlier})
            df_ratio_out = pd.DataFrame({"base_model": base_model_label, "methods": methods ,"mean": ratio_mean_outlier, "sd": ratio_sd_outlier})
            suffix = detector_suffix(detector)
            df_cover_out.to_csv(os.path.join(data_dir, f"{dataset}_coverage_outlier{suffix}_summary.csv"))
            df_ratio_out.to_csv(os.path.join(data_dir, f"{dataset}_ratio_outlier{suffix}_summary.csv"))


        return np.array(cover_results), np.array(isl_results), \
                np.array(length_results), \
                scarcity_coverage_results, scarcity_worst_coverage_results, \
                np.array(smis_components_results), \
                coverage_outlier_results, ratio_results

    return np.array(cover_results), np.array(isl_results), \
            np.array(length_results), \
            scarcity_coverage_results, scarcity_worst_coverage_results, \
            np.array(smis_components_results)


if __name__ == "__main__":
    if kernel == "RBF":
        kernel_gp = gpx.kernels.RBF()
    elif kernel == "Matern32":
        kernel_gp = gpx.kernels.Matern32()
    elif kernel == "Matern52":
        kernel_gp = gpx.kernels.Matern52()
    elif kernel == "RationalQuadratic":
        kernel_gp = gpx.kernels.RationalQuadratic()
    elif kernel == "RBF + Matern52":
        kernel_gp = gpx.kernels.RBF() + gpx.kernels.Matern52()
    else:
        kernel_gp = (
        gpx.kernels.RBF() + 
        gpx.kernels.Matern32()+
        gpx.kernels.Matern52(lengthscale=0.12)
    )
    
    if kernel_noise == "RBF":
        kernel_noise_gp = gpx.kernels.RBF()
    elif kernel_noise == "Matern32":
        kernel_noise_gp = gpx.kernels.Matern32()
    else:
        kernel_noise_gp = gpx.kernels.RationalQuadratic()
    
    DATA_PATH = os.path.join(original_path , "data")
    RESULTS_PATH = os.path.join(original_path , "results")
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # fixing random generator and torch seeds
    rng = np.random.default_rng(seed_initial)
    torch.manual_seed(seed_initial)
    torch.cuda.manual_seed(seed_initial)
    
    # Check for an existing checkpoint to optionally resume the experiment
    chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
    if outlier_analysis and not(outlier_same_time):
        chk_file = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}_outlier{outlier_result_suffix}.pkl")
    elif outlier_same_time and outlier_analysis:    
        chk_file = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}.pkl")
        chk_file_outlier = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}_outlier{outlier_result_suffix}.pkl")
    else:
        chk_file = os.path.join(chk_dir, f"{dataset}_checkpoint_{base_model_slug}.pkl")
    resume_from = 0
    checkpoint_data = None
    loaded_cover = loaded_isl = None
    loaded_seeds_so_far = None
    
    if os.path.exists(chk_file):
        try:
            with open(chk_file, "rb") as f:
                checkpoint_data = pickle.load(f)
            checkpoint_flag = True
            print(f"Found checkpoint for dataset '{dataset}'. Resuming from iteration {resume_from}.")
        except Exception as e:
            print(f"Failed to load checkpoint '{chk_file}': {e}")
            checkpoint_data = None
            checkpoint_flag = False
        if outlier_same_time and outlier_analysis:
            try:
                with open(chk_file_outlier, "rb") as f:
                    checkpoint_data_outlier = pickle.load(f)
                checkpoint_flag = True
                print(f"Found outlier checkpoint for dataset '{dataset}'. Resuming from iteration {resume_from}.")
            except Exception as e: 
                print(f"Failed to load outlier checkpoint '{chk_file_outlier}': {e}") 
                checkpoint_data_outlier = None 
                checkpoint_flag = False
        else:
            checkpoint_data_outlier = None
    else:
        print(f"No checkpoint found at '{chk_file}'. Starting a new run.")
        checkpoint_flag = False
        checkpoint_data_outlier = None

    def checkpoint_has_enough_seeds(checkpoint, required_n_rep):
        if checkpoint is None:
            return False
        seeds = checkpoint.get("seeds")
        return seeds is not None and len(seeds) >= required_n_rep

    def checkpoint_matches_protocol(checkpoint):
        """Do not resume a checkpoint produced with a different backbone/diagnostic setup."""
        if checkpoint is None:
            return False
        return (
            checkpoint.get("protocol_version") == 3
            and checkpoint.get("shared_qnn_backbone") == shared_qnn_backbone
            and checkpoint.get("credo_first_qnn_backbone") == credo_first_qnn_backbone
            and checkpoint.get("credo_dropout_training") == credo_dropout_training
            and checkpoint.get("scarcity_method") == scarcity_method
            and checkpoint.get("outlier_detector") == outlier_detector
        )

    if checkpoint_flag and not checkpoint_matches_protocol(checkpoint_data):
        print(
            f"Ignoring checkpoint '{chk_file}' because it was produced with "
            "a different QNN-sharing or diagnostic configuration."
        )
        checkpoint_flag = False
        checkpoint_data = None
        checkpoint_data_outlier = None
    elif (
        checkpoint_flag
        and outlier_same_time
        and outlier_analysis
        and not checkpoint_matches_protocol(checkpoint_data_outlier)
    ):
        print(
            f"Ignoring outlier checkpoint '{chk_file_outlier}' because it was produced with "
            "a different QNN-sharing or diagnostic configuration."
        )
        checkpoint_flag = False
        checkpoint_data = None
        checkpoint_data_outlier = None

    if checkpoint_flag and not checkpoint_has_enough_seeds(checkpoint_data, n_rep):
        print(
            f"Ignoring checkpoint '{chk_file}' because it has fewer seeds than n_rep={n_rep}. "
            "Starting a new run."
        )
        checkpoint_flag = False
        checkpoint_data = None
        checkpoint_data_outlier = None
    elif (
        checkpoint_flag
        and outlier_same_time
        and outlier_analysis
        and not checkpoint_has_enough_seeds(checkpoint_data_outlier, n_rep)
    ):
        print(
            f"Ignoring outlier checkpoint '{chk_file_outlier}' because it has fewer seeds than n_rep={n_rep}. "
            "Starting a new run."
        )
        checkpoint_flag = False
        checkpoint_data = None
        checkpoint_data_outlier = None

    if not outlier_analysis:
        cover, isl, length, scarcity_cover, scarcity_worst_cover, smis_components = run_experiment(
            dataset = dataset, 
            n_rep = n_rep, 
            target_column = "target",
            checkpoint_flag = checkpoint_flag,
            checkpoint_data = checkpoint_data,
            )
    
        raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}")
        os.makedirs(raw_dir, exist_ok=True)
    
        to_save = {
            "cover": cover,
            "isl": isl,
            "length": length,
            "smis_components": smis_components,
        }
        for name, arr in to_save.items():
            filepath = os.path.join(raw_dir, f"{dataset}_{name}_{base_model_slug}_raw.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        for scarcity_name in scarcity_methods:
            suffix = scarcity_result_suffixes[scarcity_name]
            for name, values in [
                ("scarcity_cover", scarcity_cover[scarcity_name]),
                ("scarcity_worst_cover", scarcity_worst_cover[scarcity_name]),
            ]:
                filepath = os.path.join(raw_dir, f"{dataset}_{name}{suffix}_{base_model_slug}_raw.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        chk_file = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_{base_model_slug}.pkl")
        try:
            if os.path.exists(chk_file):
                os.remove(chk_file)
                chk_dir = os.path.dirname(chk_file)
                if os.path.isdir(chk_dir) and not os.listdir(chk_dir):
                    os.rmdir(chk_dir)
        except Exception as e:
            print(f"Failed to delete checkpoint {chk_file}: {e}")
    
    elif outlier_analysis and outlier_same_time:
        print("Running main experiment with outlier analysis at the same time")
        cover, isl, length, scarcity_cover, scarcity_worst_cover, smis_components, cover_out, ratio_out = run_experiment(
            dataset = dataset, 
            n_rep = n_rep, 
            target_column = "target", 
            checkpoint_flag = checkpoint_flag, 
            checkpoint_data = checkpoint_data, 
            checkpoint_data_outlier=checkpoint_data_outlier, 
            outlier_same_time=outlier_same_time, 
            outlier_analysis=outlier_analysis,
            outlier_detector=outlier_detector,
            inlier_size=outlier_inlier_size,
            contamination=outlier_contamination,
            n_neighbors=outlier_neighbors,
            n_components=outlier_tsne_components,
            tsne_random_state=outlier_tsne_random_state,
            )
            
        raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}")
        os.makedirs(raw_dir, exist_ok=True)
        to_save = {"cover": cover, 
                   "isl": isl, 
                   "length": length,
                   "smis_components": smis_components,
                   }
        for name, arr in to_save.items(): 
            filepath = os.path.join(
                raw_dir, f"{dataset}_{name}_{base_model_slug}_raw.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        for scarcity_name in scarcity_methods:
            suffix = scarcity_result_suffixes[scarcity_name]
            for name, values in [
                ("scarcity_cover", scarcity_cover[scarcity_name]),
                ("scarcity_worst_cover", scarcity_worst_cover[scarcity_name]),
            ]:
                filepath = os.path.join(raw_dir, f"{dataset}_{name}{suffix}_{base_model_slug}_raw.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        for detector in outlier_detectors:
            suffix = detector_suffix(detector)
            for name, values in [
                ("cover_out", cover_out[detector]),
                ("ratio_out", ratio_out[detector]),
            ]:
                filepath = os.path.join(raw_dir, f"{dataset}_{name}{suffix}_{base_model_slug}_raw.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        chk_file = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_{base_model_slug}.pkl")
        chk_file_out = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_{base_model_slug}_outlier{outlier_result_suffix}.pkl")
        try:
            if os.path.exists(chk_file):
                os.remove(chk_file)
                chk_dir = os.path.dirname(chk_file)
                if os.path.isdir(chk_dir) and not os.listdir(chk_dir):
                    os.rmdir(chk_dir)
        except Exception as e:
            print(f"Failed to delete checkpoint {chk_file}: {e}")
        
        try:
            if os.path.exists(chk_file_out): 
                os.remove(chk_file_out)
                chk_dir = os.path.dirname(chk_file_out)
            if os.path.isdir(chk_dir) and not os.listdir(chk_dir):
                os.rmdir(chk_dir) 
        except Exception as e: 
            print(f"Failed to delete checkpoint {chk_file_out}: {e}")
            
    else:
        cover_out, ratio_out = run_experiment_outlier(
            dataset = dataset, 
            n_rep = n_rep, 
            target_column = "target",
            checkpoint_flag = checkpoint_flag,
            checkpoint_data = checkpoint_data,
            outlier_detector=outlier_detector,
            inlier_size=outlier_inlier_size,
            contamination=outlier_contamination,
            n_neighbors=outlier_neighbors,
            n_components=outlier_tsne_components,
            tsne_random_state=outlier_tsne_random_state,
            seed_initial=seed_initial,
            iforest_n_estimators=iforest_n_estimators,
            iforest_max_samples=iforest_max_samples,
            iforest_max_features=iforest_max_features,
            iforest_bootstrap=iforest_bootstrap,
        )
    
        raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}_outlier{outlier_result_suffix}")
        os.makedirs(raw_dir, exist_ok=True)
    
        to_save = {"cover_out": cover_out, "ratio_out": ratio_out}
        for name, arr in to_save.items():
            filepath = os.path.join(raw_dir, f"{dataset}_{name}{outlier_result_suffix}_{base_model_slug}_raw.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        chk_file = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_{base_model_slug}_outlier{outlier_result_suffix}.pkl")
        try:
            if os.path.exists(chk_file):
                os.remove(chk_file)
                chk_dir = os.path.dirname(chk_file)
                if os.path.isdir(chk_dir) and not os.listdir(chk_dir):
                    os.rmdir(chk_dir)
        except Exception as e:
            print(f"Failed to delete checkpoint {chk_file}: {e}")
