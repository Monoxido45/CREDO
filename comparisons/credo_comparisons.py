import os
import numpy as np
import pandas as pd
import traceback
import torch
from argparse import ArgumentParser
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

original_path = os.getcwd()
os.chdir(os.path.join(original_path, "comparisons"))

from credal_cp.credal_cp import CredalCPRegressor
from credal_cp.utils import (
    average_interval_score_loss,
    average_coverage,
    corr_coverage_widths,
    compute_interval_length,
)
# uacqr part
from uacqr import uacqr
# EPIC part
from epic import QuantileScore, EPIC_split
import pickle
import os

# Importing outlier to inlier ratio auxiliary functions
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE

os.chdir(original_path)
print(original_path)

parser = ArgumentParser()
parser.add_argument("-alpha", "--alpha",type=float, default=0.1, help="miscoverage level for conformal prediction")
parser.add_argument("-gamma","--gamma", type=float, default=0.1, help="adaptive gamma parameter")
parser.add_argument("-n_rep", "--n_rep", type=int, default=30, help="number of repetitions for the experiment")
parser.add_argument("-n_MCMC", "--n_MCMC", type=int, default=1000, help="number of MCMC samples for BART")
parser.add_argument("-alpha_bart", "--alpha_bart", type=float, default=0.95, help="alpha parameter for BART prior")
parser.add_argument("-seed_initial", "--seed_initial", type=int, default=125, help="initial seed for random generator to create seeds for repetitions")
parser.add_argument("-dataset", "--dataset", type=str, default="airfoil", help="dataset to use for the experiment")
parser.add_argument("-uacqr_model", "--uacqr_model", type=str, default="catboost", help="UACQR and CQR base models: 'rfqr' or 'catboost'")
parser.add_argument("-outlier_analysis", "--outlier_analysis", type=bool, help="whether to perform outlier analysis using LOF and t-SNE")
parser.add_argument("-n_cores", "--n_cores", type=int, default=4, help="number of cores to use for parallel processing")
parser.add_argument("-m_bart", "--m_bart", type=int, default=100, help="number of trees used in CREDO-BART")
args = parser.parse_args()

alpha = args.alpha
gamma = args.gamma
n_rep = args.n_rep
n_MCMC = args.n_MCMC
alpha_bart = args.alpha_bart
seed_initial = args.seed_initial
dataset = args.dataset
uacqr_model = args.uacqr_model
outlier_analysis = args.outlier_analysis
n_cores = args.n_cores
m_bart = args.m_bart

DATA_PATH = os.path.join(original_path , "data")
RESULTS_PATH = os.path.join(original_path , "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

# fixing random generator and torch seeds
rng = np.random.default_rng(seed_initial)
torch.manual_seed(seed_initial)
torch.cuda.manual_seed(seed_initial)

# Check for an existing checkpoint to optionally resume the experiment
chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
if outlier_analysis:
    chk_file = os.path.join(chk_dir, f"{dataset}_checkpoint_{uacqr_model}_outlier.pkl")
else:    
    chk_file = os.path.join(chk_dir, f"{dataset}_checkpoint_{uacqr_model}.pkl")
resume_from = 0
checkpoint_data = None
loaded_cover = loaded_isl = loaded_IL = loaded_pcorr = None
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
else:
    print(f"No checkpoint found at '{chk_file}'. Starting a new run.")
    checkpoint_flag = False


def generate_seeds(seed_initial, n_rep):
    np.random.seed(seed_initial)
    seeds = np.random.randint(0, 2**31 - 1, size=n_rep)
    return seeds

def fit_methods(
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        mdn_params,
        i,
        scale_y = False,
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

    # Fitting CREDO with BART
    print(f"Fitting CREDO with BART")
    credal_CP_bart_adaptive = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "BART",
    alpha = alpha,
    adaptive_gamma = True,
    gamma = gamma,
    )
    
    credal_CP_bart_adaptive.fit(
        X_train, 
        y_train,
        progressbar = True,
        n_cores = n_cores,
        n_MCMC = n_MCMC,
        normalize_y=True,
        alpha_bart = alpha_bart,
        random_seed_fit=i,
        m=m_bart,
    )
    credal_CP_bart_adaptive.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)

    credal_CP_bart_fixed = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = credal_CP_bart_adaptive.base_model,
    alpha = alpha,
    adaptive_gamma = False,
    gamma = gamma,
    is_fitted = True,
    )

    credal_CP_bart_fixed.fit(
        X_train, 
        y_train,
        base_model_type ="BART",
    )
    
    credal_CP_bart_fixed.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)

    credo_adaptive_int = credal_CP_bart_adaptive.predict(X_test)
    credo_fixed_int = credal_CP_bart_fixed.predict(X_test)

    # deleting CREDO models to free memory
    del credal_CP_bart_adaptive
    del credal_CP_bart_fixed
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

    # checking if there are any infinite bounds in UACQR-S or UACQR-P and removing 
    # those indices from all methods to ensure fair comparison
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

    del uacqr_results
    gc.collect()
    
    # evaluating metrics of interest
    # marginal coverage
    cover_credo_adap = average_coverage(
        credo_adaptive_int[:, 1], credo_adaptive_int[:, 0], y_test
    )
    cover_credo_fixed = average_coverage(
        credo_fixed_int[:, 1], credo_fixed_int[:, 0], y_test
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
    isl_credo_adap = average_interval_score_loss(
        credo_adaptive_int[:, 1], credo_adaptive_int[:, 0], y_test, alpha
    )
    isl_credo_fixed = average_interval_score_loss(
        credo_fixed_int[:, 1], credo_fixed_int[:, 0], y_test, alpha
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

    # IL
    IL_credo_adap = np.mean(compute_interval_length(credo_adaptive_int[:, 1], credo_adaptive_int[:, 0]))
    IL_credo_fixed = np.mean(compute_interval_length(credo_fixed_int[:, 1], credo_fixed_int[:, 0]))
    IL_cqr = np.mean(compute_interval_length(cqr_int[:, 1], cqr_int[:, 0]))
    IL_cqrr = np.mean(compute_interval_length(cqrr_int[:, 1], cqrr_int[:, 0]))
    IL_uacqrs = np.mean(compute_interval_length(uacqrs_int[:, 1], uacqrs_int[:, 0]))
    IL_uacqrp = np.mean(compute_interval_length(uacqrp_int[:, 1], uacqrp_int[:, 0]))
    IL_epic_mdn = np.mean(compute_interval_length(pred_epic_mdn_test[:, 1], pred_epic_mdn_test[:, 0]))

    # pcorr
    pcorr_credo_adap = corr_coverage_widths(
        credo_adaptive_int[:, 1], credo_adaptive_int[:, 0], y_test
    )
    pcorr_credo_fixed = corr_coverage_widths(
        credo_fixed_int[:, 1], credo_fixed_int[:, 0], y_test
    )
    pcorr_cqr = corr_coverage_widths(
        cqr_int[:, 1], cqr_int[:, 0], y_test
    )
    pcorr_cqrr = corr_coverage_widths(
        cqrr_int[:, 1], cqrr_int[:, 0], y_test
    )
    if not (n_removed == n_total):
      pcorr_uacqrs = corr_coverage_widths(
          uacqrs_int[:, 1], uacqrs_int[:, 0],
          y_test_uacqr
      )
      pcorr_uacqrp = corr_coverage_widths(
          uacqrp_int[:, 1], uacqrp_int[:, 0],
          y_test_uacqr
      )
    pcorr_epic_mdn = corr_coverage_widths(
        pred_epic_mdn_test[:, 1], pred_epic_mdn_test[:, 0],
        y_test
    )
    
    if n_removed == n_total:
      IL_uacqrs, IL_uacqrp = np.nan, np.nan
      isl_uacqrs, isl_uacqrp= np.nan, np.nan
      cover_uacqrs, cover_uacqrp = np.nan, np.nan
      pcorr_uacqrs, pcorr_uacqrp = np.nan, np.nan
    
    IL_array = np.array([
        IL_credo_adap,
        IL_credo_fixed,
        IL_cqr,
        IL_cqrr,
        IL_uacqrs,
        IL_uacqrp,
        IL_epic_mdn,
    ])
    isl_array = np.array([
        isl_credo_adap,
        isl_credo_fixed,
        isl_cqr,
        isl_cqrr,
        isl_uacqrs,
        isl_uacqrp,
        isl_epic_mdn,
    ])
    cover_array = np.array([
        cover_credo_adap,
        cover_credo_fixed,
        cover_cqr,
        cover_cqrr,
        cover_uacqrs,
        cover_uacqrp,
        cover_epic_mdn,
    ])
    pcorr_array = np.array([
        pcorr_credo_adap,
        pcorr_credo_fixed,
        pcorr_cqr,
        pcorr_cqrr,
        pcorr_uacqrs,
        pcorr_uacqrp,
        pcorr_epic_mdn,
    ])

    return cover_array, isl_array, IL_array, pcorr_array

def fit_methods_outlier(
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        mdn_params,
        i,
        scale_y = False,
        inlier_size = 0.2,
        n_neighbors = 15,
        contamination = 0.05,
        tsne_random_state=120,
        n_components = 2,
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

    # Fitting CREDO with BART
    print(f"Fitting CREDO with BART")
    credal_CP_bart_adaptive = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = "BART",
    alpha = alpha,
    adaptive_gamma = True,
    gamma = gamma,
    )
    
    credal_CP_bart_adaptive.fit(
        X_train, 
        y_train,
        progressbar = True,
        n_cores = n_cores,
        n_MCMC = n_MCMC,
        normalize_y=True,
        alpha_bart = alpha_bart,
        random_seed_fit=i,
    )
    credal_CP_bart_adaptive.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)

    credal_CP_bart_fixed = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = credal_CP_bart_adaptive.base_model,
    alpha = alpha,
    adaptive_gamma = False,
    gamma = gamma,
    is_fitted = True,
    )

    credal_CP_bart_fixed.fit(
        X_train, 
        y_train,
        base_model_type ="BART",
    )
    
    credal_CP_bart_fixed.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)

    credo_adaptive_int = credal_CP_bart_adaptive.predict(X_test)
    credo_fixed_int = credal_CP_bart_fixed.predict(X_test)

    # deleting CREDO models to free memory
    del credal_CP_bart_adaptive
    del credal_CP_bart_fixed
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

    # Detecting outliers using t-SNE and Local Outlier Factor
    print(f"Performing outlier detection with t-SNE and Local Outlier Factor")
    tsne = TSNE(n_components=n_components, random_state=tsne_random_state)
    X_tsne_test = tsne.fit_transform(X_test)

    # Standardize the features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_tsne_test)
    # Use Local Outlier Factor for anomaly detection on scaled data
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    out_pred = lof.fit_predict(X_test_scaled)

    outlier_obs = y_test[out_pred == -1]
    outlier_indexes = np.where(out_pred == -1)[0]
    # selecting 15% top inliers
    inlier_indexes = np.setdiff1d(np.arange(len(y_test)), outlier_indexes)
    inlier_scores = lof.negative_outlier_factor_[inlier_indexes]
    # computing inlier scores
    size = int((y_test.shape[0] - outlier_obs.shape[0]) * inlier_size)
    most_inlier_idxs = inlier_indexes[np.argsort(inlier_scores)[::-1][:size]]

    # selecting prediction intervals for inliers and outliers
    credo_adapt_outliers = credo_adaptive_int[outlier_indexes]
    credo_fixed_outliers = credo_fixed_int[outlier_indexes]
    cqr_outliers = cqr_int[outlier_indexes]
    cqrr_outliers = cqrr_int[outlier_indexes]
    uacqrs_outliers = uacqrs_int[outlier_indexes]
    uacqrp_outliers = uacqrp_int[outlier_indexes]
    epic_mdn_outliers = pred_epic_mdn_test[outlier_indexes]
    y_test_out = y_test[outlier_indexes]

    credo_adapt_inliers = credo_adaptive_int[most_inlier_idxs]
    credo_fixed_inliers = credo_fixed_int[most_inlier_idxs]
    cqr_inliers = cqr_int[most_inlier_idxs]
    cqrr_inliers = cqrr_int[most_inlier_idxs]
    uacqrs_inliers = uacqrs_int[most_inlier_idxs]
    uacqrp_inliers = uacqrp_int[most_inlier_idxs]
    epic_mdn_inliers = pred_epic_mdn_test[most_inlier_idxs]
    y_test_in = y_test[most_inlier_idxs]


    # checking if there are any infinite bounds in UACQR-S or UACQR-P and removing 
    # those indices from all methods to ensure fair comparison
    # check finite bounds but only for the selected outlier + inlier indices
    lower_s = np.asarray(uacqr_pred_test["UACQR-S"]["lower"])
    upper_s = np.asarray(uacqr_pred_test["UACQR-S"]["upper"])
    finite_s = np.isfinite(lower_s) & np.isfinite(upper_s)

    lower_p = np.asarray(uacqr_pred_test["UACQR-P"]["lower"])
    upper_p = np.asarray(uacqr_pred_test["UACQR-P"]["upper"])
    finite_p = np.isfinite(lower_p) & np.isfinite(upper_p)

    # combined indices of interest (outliers + selected inliers)
    combined_idxs = np.concatenate([outlier_indexes, most_inlier_idxs])
    finite_s_combined = finite_s[combined_idxs]
    finite_p_combined = finite_p[combined_idxs]
    good_combined_mask = finite_s_combined & finite_p_combined

    n_total_combined = combined_idxs.shape[0]
    n_removed_combined = int((~good_combined_mask).sum())
    print(f"Combined removal: excluding {n_removed_combined} of {n_total_combined} selected points with infinite bounds in either UACQR-S or UACQR-P")

    # valid combined indices (in the original test-set indexing)
    valid_combined_idxs = combined_idxs[good_combined_mask]

    # filter the previously sliced outlier / inlier arrays to keep only valid entries
    outlier_keep_pos = np.isin(outlier_indexes, valid_combined_idxs)
    inlier_keep_pos = np.isin(most_inlier_idxs, valid_combined_idxs)

    uacqrs_outliers = uacqrs_outliers[outlier_keep_pos]
    uacqrp_outliers = uacqrp_outliers[outlier_keep_pos]
    y_test_out_uacqr = y_test_out[outlier_keep_pos]

    uacqrs_inliers = uacqrs_inliers[inlier_keep_pos]
    uacqrp_inliers = uacqrp_inliers[inlier_keep_pos]
    y_test_in_uacqr = y_test_in[inlier_keep_pos]
    del uacqr_results
    gc.collect()
    
    # evaluating metrics of interest
    # coverage for outliers
    cover_credo_adap_out = average_coverage(
        credo_adapt_outliers[:, 1], credo_adapt_outliers[:, 0], y_test_out
    )
    cover_credo_fixed_out = average_coverage(
        credo_fixed_outliers[:, 1], credo_fixed_outliers[:, 0], y_test_out
    )
    cover_cqr_out = average_coverage(
        cqr_outliers[:, 1], cqr_outliers[:, 0], y_test_out
    )
    cover_cqrr_out = average_coverage(
        cqrr_outliers[:, 1], cqrr_outliers[:, 0], y_test_out
    )
    cover_uacqrs_out = average_coverage(
        uacqrs_outliers[:, 1], uacqrs_outliers[:, 0],
        y_test_out_uacqr
    )
    cover_uacqrp_out = average_coverage(
        uacqrp_outliers[:, 1], uacqrp_outliers[:, 0],
        y_test_out_uacqr
    )
    cover_epic_mdn_out = average_coverage(
        epic_mdn_outliers[:, 1], epic_mdn_outliers[:, 0],
        y_test_out
    )

    # ISL on outliers
    isl_credo_adap_out = average_interval_score_loss(
        credo_adapt_outliers[:, 1], credo_adapt_outliers[:, 0], y_test_out, alpha
    )
    isl_credo_fixed_out = average_interval_score_loss(
        credo_fixed_outliers[:, 1], credo_fixed_outliers[:, 0], y_test_out, alpha
    )
    isl_cqr_out = average_interval_score_loss(
        cqr_outliers[:, 1], cqr_outliers[:, 0], y_test_out, alpha
    )
    isl_cqrr_out = average_interval_score_loss(
        cqrr_outliers[:, 1], cqrr_outliers[:, 0], y_test_out, alpha
    )
    isl_uacqrs_out = average_interval_score_loss(
        uacqrs_outliers[:, 1], uacqrs_outliers[:, 0],
        y_test_out_uacqr, alpha
    )
    isl_uacqrp_out = average_interval_score_loss(
        uacqrp_outliers[:, 1], uacqrp_outliers[:, 0],
        y_test_out_uacqr, alpha
    )
    isl_epic_mdn_out = average_interval_score_loss(
        epic_mdn_outliers[:, 1], epic_mdn_outliers[:, 0],
        y_test_out, alpha
    )

    # Interval length ratio
    credo_adap_ratio = np.mean(
            compute_interval_length(
                credo_adapt_outliers[:, 1], credo_adapt_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                credo_adapt_inliers[:, 1], credo_adapt_inliers[:, 0]
            )
        )
    credo_fixed_ratio = np.mean(
            compute_interval_length(
                credo_fixed_outliers[:, 1], credo_fixed_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                credo_fixed_inliers[:, 1], credo_fixed_inliers[:, 0]
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
    if not (n_removed_combined == n_total_combined):
        uacqrs_ratio = np.mean(
                compute_interval_length(
                    uacqrs_outliers[:, 1], uacqrs_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    uacqrs_inliers[:, 1], uacqrs_inliers[:, 0]
                )
            )
        uacqrp_ratio = np.mean(
                compute_interval_length(
                    uacqrp_outliers[:, 1], uacqrp_outliers[:, 0]
                )
            ) / np.mean(
                compute_interval_length(
                    uacqrp_inliers[:, 1], uacqrp_inliers[:, 0]
                )
            )
    if n_removed_combined == n_total_combined:
        cover_uacqrs_out, cover_uacqrp_out = np.nan, np.nan
        isl_uacqrs_out, isl_uacqrp_out = np.nan, np.nan
        uacqrs_ratio, uacqrp_ratio = np.nan, np.nan
    
    isl_array = np.array([
        isl_credo_adap_out,
        isl_credo_fixed_out,
        isl_cqr_out,
        isl_cqrr_out,
        isl_uacqrs_out,
        isl_uacqrp_out,
        isl_epic_mdn_out,
    ])
    cover_array = np.array([
        cover_credo_adap_out,
        cover_credo_fixed_out,
        cover_cqr_out,
        cover_cqrr_out,
        cover_uacqrs_out,
        cover_uacqrp_out,
        cover_epic_mdn_out,
    ])
    ratio_array = np.array([
        credo_adap_ratio,
        credo_fixed_ratio,
        cqr_ratio,
        cqrr_ratio,
        uacqrs_ratio,
        uacqrp_ratio,
        epic_mdn_ratio,
    ])

    return cover_array, isl_array, ratio_array

def run_experiment_outlier(
    dataset,
    n_rep,
    target_column,
    prop_test = 0.2,
    inlier_size=0.2,
    contamination=0.05,
    n_neighbors=20,
    n_components=2,
    tsne_random_state=120,
    seed_initial=145,
    checkpoint_flag = False,
    checkpoint_data = None,
):
    data = pd.read_csv(os.path.join(DATA_PATH, f"{dataset}.csv"))

    # EPICSCORE params
    mdn_params = {
    "num_components": 3,
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

    if data.shape[0] > 10000:
        mdn_params["batch_size"] = 125
    if dataset == "WEC":
        mdn_params["batch_size"] = 250

    if checkpoint_flag:
        resume_from = int(checkpoint_data.get("iteration", -1)) + 1
        ratio_results = checkpoint_data.get("ratio_results", [])
        coverage_results = checkpoint_data.get("coverage_results", [])
        isl_results = checkpoint_data.get("isl_results", [])
        seeds = checkpoint_data.get("seeds", None)
        print(f"Resuming from iteration {resume_from}. Loaded {len(coverage_results)} results so far.")
    else:
        resume_from = 0
        seeds = generate_seeds(seed_initial, n_rep)
        coverage_results = []
        isl_results = []
        ratio_results = []

        for i in tqdm(range(resume_from, n_rep), desc = f"Running methods for dataset: {dataset}"):
            print(f"Repetition {i+1}/{n_rep}")
            seed = seeds[i]
            X = data.drop(columns=[target_column])
            y = data[target_column]

            X_train_calib, X_test, y_train_calib, y_test = train_test_split(
            X, y, test_size=prop_test, random_state=seed
        )
            if X.shape[0] < 5000:
                prop_train = 0.5
            else:
                prop_train = 0.7
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train_calib, y_train_calib, test_size=1-prop_train, random_state=seed
            )

            if dataset in ["superconductivity"]:
                scale_y = True
            else:
                scale_y = False

            cover_array, isl_array, ratio_array = fit_methods_outlier(
                X_train,
                y_train,
                X_calib,
                y_calib,
                X_test,
                y_test,
                mdn_params,
                i,
                scale_y = scale_y,
                inlier_size = inlier_size,
                n_neighbors = n_neighbors,
                contamination = contamination,
                tsne_random_state=tsne_random_state,
                n_components = n_components,
            )
            coverage_results.append(cover_array)
            isl_results.append(isl_array)
            ratio_results.append(ratio_array)

            def save_checkpoint(iteration, seeds):
                try:
                    checkpoint = {
                        "coverage_results": coverage_results,
                        "isl_results": isl_results,
                        "ratio_results": ratio_results,
                        "iteration": iteration,
                        "seeds": seeds,
                        "alpha": alpha,
                        "gamma": gamma,
                        "dataset": dataset,
                    }
                    chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
                    os.makedirs(chk_dir, exist_ok=True)
                    filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_{uacqr_model}_outlier.pkl")
                    with open(filepath, "wb") as f:
                        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Failed saving checkpoint at iter {iteration+1}: {e}")

            # save checkpoint after each repetition
            save_checkpoint(i, seeds)
        # summarize results: convert lists to arrays and compute mean and sd (sample sd if n_rep>1)
    coverage_results = np.array(coverage_results)
    isl_results = np.array(isl_results)
    ratio_results = np.array(ratio_results)

    def mean_sd(arr):
        mean = np.nanmean(arr, axis=0, )
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        return mean, sd

    methods = ["credo_adap", "credo_fixed", "cqr", "cqrr", "uacqrs", "uacqrp", "EPIC"]

    cover_mean, cover_sd = mean_sd(coverage_results)
    isl_mean, isl_sd = mean_sd(isl_results)
    ratio_mean, ratio_sd = mean_sd(ratio_results)

    # create summary dataframes and save to CSV
    df_cover = pd.DataFrame({"methods": methods ,"mean": cover_mean, "sd": cover_sd})
    df_isl = pd.DataFrame({"methods": methods ,"mean": isl_mean, "sd": isl_sd})
    df_ratio = pd.DataFrame({"methods": methods ,"mean": ratio_mean, "sd": ratio_sd})

    data_dir = os.path.join(RESULTS_PATH, f"{dataset}_{uacqr_model}_summary")
    os.makedirs(data_dir, exist_ok=True)

    df_cover.to_csv(os.path.join(data_dir, f"{dataset}_coverage_outlier_summary.csv"))
    df_isl.to_csv(os.path.join(data_dir, f"{dataset}_isl_outlier_summary.csv"))
    df_ratio.to_csv(os.path.join(data_dir, f"{dataset}_ratio_outlier_summary.csv"))
    return np.array(coverage_results), np.array(isl_results), np.array(ratio_results)

def run_experiment(dataset, 
                   n_rep, 
                   target_column, 
                   prop_test = 0.2,
                   checkpoint_flag = False,
                   checkpoint_data = None,
):
    data = pd.read_csv(os.path.join(DATA_PATH, f"{dataset}.csv"))

    # EPICSCORE params
    mdn_params = {
    "num_components": 3,
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

    if data.shape[0] > 10000:
        mdn_params["batch_size"] = 125
    if dataset == "WEC":
        mdn_params["batch_size"] = 250

    if checkpoint_flag:
        resume_from = int(checkpoint_data.get("iteration", -1)) + 1
        cover_results = checkpoint_data.get("cover_results", [])
        isl_results = checkpoint_data.get("isl_results", [])
        IL_results = checkpoint_data.get("IL_results", [])
        pcorr_results = checkpoint_data.get("pcorr_results", [])
        seeds = checkpoint_data.get("seeds", None)
        print(f"Resuming from iteration {resume_from}. Loaded {len(cover_results)} results so far.")
    else:
        resume_from = 0
        seeds = generate_seeds(seed_initial, n_rep)
        cover_results = []
        isl_results = []
        IL_results = []
        pcorr_results = []

    for i in tqdm(range(resume_from, n_rep), desc = f"Running methods for dataset: {dataset}"):
        print(f"Repetition {i+1}/{n_rep}")
        seed = seeds[i]
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train_calib, X_test, y_train_calib, y_test = train_test_split(
            X, y, test_size=prop_test, random_state=seed
        )
        if X.shape[0] < 5000:
          prop_train = 0.5
        else:
          prop_train = 0.7
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_calib, y_train_calib, test_size=1-prop_train, random_state=seed
        )

        if dataset in ["superconductivity", "homes", "protein"]:
            scale_y = True
        else:
            scale_y = False

        cover_array, isl_array, IL_array, pcorr_array = fit_methods(
            X_train,
            y_train,
            X_calib,
            y_calib,
            X_test,
            y_test,
            mdn_params,
            i,
            scale_y = scale_y,
        )
        cover_results.append(cover_array)
        isl_results.append(isl_array)
        IL_results.append(IL_array)
        pcorr_results.append(pcorr_array)

        def save_checkpoint(iteration, seeds):
            try:
                checkpoint = {
                    "cover_results": cover_results,
                    "isl_results": isl_results,
                    "IL_results": IL_results,
                    "pcorr_results": pcorr_results,
                    "iteration": iteration,
                    "seeds": seeds,
                    "alpha": alpha,
                    "gamma": gamma,
                    "dataset": dataset,
                }
                chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
                os.makedirs(chk_dir, exist_ok=True)
                filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_{uacqr_model}.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"Failed saving checkpoint at iter {iteration+1}: {e}")

        # save checkpoint after each repetition
        save_checkpoint(i, seeds)
    
    # summarize results: convert lists to arrays and compute mean and sd (sample sd if n_rep>1)
    cover_results = np.array(cover_results)
    isl_results = np.array(isl_results)
    IL_results = np.array(IL_results)
    pcorr_results = np.array(pcorr_results)

    def mean_sd(arr):
        mean = np.nanmean(arr, axis=0, )
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        return mean, sd

    methods = ["credo_adap", "credo_fixed", "cqr", "cqrr", "uacqrs", "uacqrp", "EPIC"]

    cover_mean, cover_sd = mean_sd(cover_results)
    isl_mean, isl_sd = mean_sd(isl_results)
    IL_mean, IL_sd = mean_sd(IL_results)
    pcorr_mean, pcorr_sd = mean_sd(pcorr_results)

    # create summary dataframes and save to CSV
    df_cover = pd.DataFrame({"methods": methods ,"mean": cover_mean, "sd": cover_sd})
    df_isl = pd.DataFrame({"methods": methods ,"mean": isl_mean, "sd": isl_sd})
    df_IL = pd.DataFrame({"methods": methods ,"mean": IL_mean, "sd": IL_sd})
    df_pcorr = pd.DataFrame({"methods": methods ,"mean": pcorr_mean, "sd": pcorr_sd})

    data_dir = os.path.join(RESULTS_PATH, f"{dataset}_{uacqr_model}_summary")
    os.makedirs(data_dir, exist_ok=True)

    df_cover.to_csv(os.path.join(data_dir, f"{dataset}_coverage_summary.csv"))
    df_isl.to_csv(os.path.join(data_dir, f"{dataset}_isl_summary.csv"))
    df_IL.to_csv(os.path.join(data_dir, f"{dataset}_IL_summary.csv"))
    df_pcorr.to_csv(os.path.join(data_dir, f"{dataset}_pcorr_summary.csv"))
    return np.array(cover_results), np.array(isl_results), np.array(IL_results), np.array(pcorr_results)

if not outlier_analysis:
    cover, isl, IL, pcorr = run_experiment(
        dataset = dataset, 
        n_rep = n_rep, 
        target_column = "target",
        checkpoint_flag = checkpoint_flag,
        checkpoint_data = checkpoint_data
        )

    raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}")
    os.makedirs(raw_dir, exist_ok=True)

    to_save = {"cover": cover, "isl": isl, "IL": IL, "pcorr": pcorr}
    for name, arr in to_save.items():
        filepath = os.path.join(raw_dir, f"{dataset}_{name}_{uacqr_model}_raw.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)


    chk_file = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_{uacqr_model}.pkl")
    try:
        if os.path.exists(chk_file):
            os.remove(chk_file)
            chk_dir = os.path.dirname(chk_file)
            if os.path.isdir(chk_dir) and not os.listdir(chk_dir):
                os.rmdir(chk_dir)
    except Exception as e:
        print(f"Failed to delete checkpoint {chk_file}: {e}")
else:
    cover_out, isl_out, ratio_out = run_experiment_outlier(
        dataset = dataset, 
        n_rep = n_rep, 
        target_column = "target",
        checkpoint_flag = checkpoint_flag,
        checkpoint_data = checkpoint_data
    )

    raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}_outlier")
    os.makedirs(raw_dir, exist_ok=True)

    to_save = {"cover_out": cover_out, "isl_out": isl_out, "ratio_out": ratio_out}
    for name, arr in to_save.items():
        filepath = os.path.join(raw_dir, f"{dataset}_{name}_{uacqr_model}_raw.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    chk_file = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_{uacqr_model}_outlier.pkl")
    try:
        if os.path.exists(chk_file):
            os.remove(chk_file)
            chk_dir = os.path.dirname(chk_file)
            if os.path.isdir(chk_dir) and not os.listdir(chk_dir):
                os.rmdir(chk_dir)
    except Exception as e:
        print(f"Failed to delete checkpoint {chk_file}: {e}")

