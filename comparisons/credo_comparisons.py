import os
import numpy as np
import pandas as pd
import traceback
import torch
from argparse import ArgumentParser
import gc
from sklearn.model_selection import train_test_split
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
from uacqr import uacqr
import pickle

os.chdir(original_path)
print(original_path)

parser = ArgumentParser()
parser.add_argument("-alpha", "--alpha",type=float, default=0.1, help="miscoverage level for conformal prediction")
parser.add_argument("-gamma","--gamma", type=float, default=0.05, help="adaptive gamma parameter")
parser.add_argument("-n_rep", "--n_rep", type=int, default=10, help="number of repetitions for the experiment")
parser.add_argument("-n_MCMC", "--n_MCMC", type=int, default=1000, help="number of MCMC samples for BART")
parser.add_argument("-alpha_bart", "--alpha_bart", type=float, default=0.98, help="alpha parameter for BART prior")
parser.add_argument("-seed_initial", "--seed_initial", type=int, default=15, help="initial seed for random generator to create seeds for repetitions")
parser.add_argument("-dataset", "--dataset", type=str, default="airfoil", help="dataset to use for the experiment")
parser.add_argument("-uacqr_model", "--uacqr_model", type=str, default="rfqr", help="UACQR and CQR base models: 'rfqr' or 'catboost'")
parser.add_argument("-n_cores", "--n_cores", type=int, default=4, help="number of cores to use for parallel processing")
parser.add_argument("-uacqr_seed", "--uacqr_seed", type = int, default = 45, help="seed for the random generator used in UACQR")
args = parser.parse_args()

alpha = args.alpha
gamma = args.gamma
n_rep = args.n_rep
n_MCMC = args.n_MCMC
alpha_bart = args.alpha_bart
seed_initial = args.seed_initial
dataset = args.dataset
uacqr_model = args.uacqr_model
n_cores = args.n_cores
uacqr_seed = args.uacqr_seed

DATA_PATH = os.path.join(original_path , "data")
RESULTS_PATH = os.path.join(original_path , "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

# fixing random generator and torch seeds
rng = np.random.default_rng(seed_initial)
torch.manual_seed(seed_initial)
torch.cuda.manual_seed(seed_initial)
alpha = 0.1
gamma = 0.05

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
): 
    # Fitting CREDO with BART
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
    )
    credal_CP_bart_adaptive.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)

    credal_CP_bart_fixed = CredalCPRegressor(
    nc_type = 'Quantile',
    base_model = credal_CP_bart_adaptive.base_model,
    alpha = alpha,
    adaptive_gamma = False,
    gamma = gamma,
    )

    credal_CP_bart_fixed.fit(
        X_train, 
        y_train,
        base_model_type = credal_CP_bart_adaptive.base_model_type,
    )
    
    credal_CP_bart_fixed.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)

    credo_adaptive_int = credal_CP_bart_adaptive.predict(X_test)
    credo_fixed_int = credal_CP_bart_fixed.predict(X_test)

    # deleting CREDO models to free memory
    del credal_CP_bart_adaptive
    del credal_CP_bart_fixed
    gc.collect()

    # Fitting UACQR
    if uacqr_model == "rfqr":
        rfqr_params = {
    "n_estimators": 100,
    "max_features" : "sqrt",
    "min_samples_leaf": 5,
    }
        uacqr_params = {
    "model_type": "rfqr",
    "B": 100, 
    "uacqrs_agg": "std",
    "base_model_type": "Quantile",
        }

        uacqr_results = uacqr(
        rfqr_params,
        bootstrapping_for_uacqrp=False,
        q_lower=alpha / 2 * 100,
        q_upper=(1 - alpha / 2) * 100,
        alpha = alpha,
        model_type=uacqr_params["model_type"],
        B=uacqr_params["B"],
        random_state=uacqr_seed,
        uacqrs_bagging=False,
        uacqrs_agg=uacqr_params["uacqrs_agg"],
     )
        
    uacqr_results.fit(X_train, y_train)
    uacqr_results.calibrate(X_calib, y_calib)
    uacqr_pred_test = uacqr_results.predict(X_test)

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
        y_test
    )
    cover_uacqrp = average_coverage(
        uacqrp_int[:, 1], uacqrp_int[:, 0],
        y_test
    )
    cover_array = np.array([
        cover_credo_adap,
        cover_credo_fixed,
        cover_cqr,
        cover_cqrr,
        cover_uacqrs,
        cover_uacqrp,
    ])

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
        y_test, alpha
    )
    isl_uacqrp = average_interval_score_loss(
        uacqrp_int[:, 1], uacqrp_int[:, 0],
        y_test, alpha
    )
    isl_array = np.array([
        isl_credo_adap,
        isl_credo_fixed,
        isl_cqr,
        isl_cqrr,
        isl_uacqrs,
        isl_uacqrp,
    ])

    # IL
    IL_credo_adap = np.mean(compute_interval_length(credo_adaptive_int[:, 1], credo_adaptive_int[:, 0]))
    IL_credo_fixed = np.mean(compute_interval_length(credo_fixed_int[:, 1], credo_fixed_int[:, 0]))
    IL_cqr = np.mean(compute_interval_length(cqr_int[:, 1], cqr_int[:, 0]))
    IL_cqrr = np.mean(compute_interval_length(cqrr_int[:, 1], cqrr_int[:, 0]))
    IL_uacqrs = np.mean(compute_interval_length(uacqrs_int[:, 1], uacqrs_int[:, 0]))
    IL_uacqrp = np.mean(compute_interval_length(uacqrp_int[:, 1], uacqrp_int[:, 0]))
    IL_array = np.array([
        IL_credo_adap,
        IL_credo_fixed,
        IL_cqr,
        IL_cqrr,
        IL_uacqrs,
        IL_uacqrp,
    ])

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
    pcorr_uacqrs = corr_coverage_widths(
        uacqrs_int[:, 1], uacqrs_int[:, 0],
        y_test
    )
    pcorr_uacqrp = corr_coverage_widths(
        uacqrp_int[:, 1], uacqrp_int[:, 0],
        y_test
    )
    pcorr_array = np.array([
        pcorr_credo_adap,
        pcorr_credo_fixed,
        pcorr_cqr,
        pcorr_cqrr,
        pcorr_uacqrs,
        pcorr_uacqrp,
    ])

    return cover_array, isl_array, IL_array, pcorr_array

def run_experiment(dataset, 
                   n_rep, 
                   target_column, 
                   prop_test = 0.2, 
                   prop_train = 0.5):
    data = pd.read_csv(os.path.join(DATA_PATH, f"{dataset}.csv"))

    seeds = generate_seeds(seed_initial, n_rep)
    cover_results = []
    isl_results = []
    IL_results = []
    pcorr_results = []
    for i in tqdm(range(n_rep), desc = f"Running methods for dataset: {dataset}"):
        print(f"Repetition {i+1}/{n_rep}")
        seed = seeds[i]
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train_calib, X_test, y_train_calib, y_test = train_test_split(
            X, y, test_size=prop_test, random_state=seed
        )
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_calib, y_train_calib, test_size=prop_train, random_state=seed
        )

        
        X_train = X_train.to_numpy(dtype=np.float32)
        y_train = y_train.to_numpy()
        X_calib = X_calib.to_numpy(dtype=np.float32)
        y_calib = y_calib.to_numpy()
        X_test = X_test.to_numpy(dtype=np.float32)
        y_test = y_test.to_numpy()


        cover_array, isl_array, IL_array, pcorr_array = fit_methods(
            X_train,
            y_train,
            X_calib,
            y_calib,
            X_test,
            y_test,
        )
        cover_results.append(cover_array)
        isl_results.append(isl_array)
        IL_results.append(IL_array)
        pcorr_results.append(pcorr_array)
    
    # summarize results: convert lists to arrays and compute mean and sd (sample sd if n_rep>1)
    cover_results = np.array(cover_results)
    isl_results = np.array(isl_results)
    IL_results = np.array(IL_results)
    pcorr_results = np.array(pcorr_results)

    def mean_sd(arr):
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        return mean, sd

    methods = ["credo_adap", "credo_fixed", "cqr", "cqrr", "uacqrs", "uacqrp"]

    cover_mean, cover_sd = mean_sd(cover_results)
    isl_mean, isl_sd = mean_sd(isl_results)
    IL_mean, IL_sd = mean_sd(IL_results)
    pcorr_mean, pcorr_sd = mean_sd(pcorr_results)

    # create summary dataframes and save to CSV
    df_cover = pd.DataFrame({"methods": methods ,"mean": cover_mean, "sd": cover_sd})
    df_isl = pd.DataFrame({"methods": methods ,"mean": isl_mean, "sd": isl_sd})
    df_IL = pd.DataFrame({"methods": methods ,"mean": IL_mean, "sd": IL_sd})
    df_pcorr = pd.DataFrame({"methods": methods ,"mean": pcorr_mean, "sd": pcorr_sd})

    data_dir = os.path.join(RESULTS_PATH, f"{dataset}_summary")
    os.makedirs(data_dir, exist_ok=True)

    df_cover.to_csv(os.path.join(data_dir, f"{dataset}_coverage_summary.csv"))
    df_isl.to_csv(os.path.join(data_dir, f"{dataset}_isl_summary.csv"))
    df_IL.to_csv(os.path.join(data_dir, f"{dataset}_IL_summary.csv"))
    df_pcorr.to_csv(os.path.join(data_dir, f"{dataset}_pcorr_summary.csv"))
    return np.array(cover_results), np.array(isl_results), np.array(IL_results), np.array(pcorr_results)

cover, isl, IL, pcorr = run_experiment(
    dataset = dataset, 
    n_rep = n_rep, 
    target_column = "target",
    )

raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}")
os.makedirs(raw_dir, exist_ok=True)

to_save = {"cover": cover, "isl": isl, "IL": IL, "pcorr": pcorr}
for name, arr in to_save.items():
    filepath = os.path.join(raw_dir, f"{dataset}_{name}_raw.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)



