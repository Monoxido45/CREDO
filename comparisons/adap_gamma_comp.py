import os
import numpy as np
import pandas as pd
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
from jaxtyping import install_import_hook
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

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

parser = ArgumentParser()
parser.add_argument("-alpha", "--alpha",type=float, default=0.1, help="miscoverage level for conformal prediction")
parser.add_argument("-gamma","--gamma", type=float, default=0.1, help="adaptive gamma parameter")
parser.add_argument("-n_rep", "--n_rep", type=int, default=30, help="number of repetitions for the experiment")
parser.add_argument("-n_MCMC", "--n_MCMC", type=int, default=1000, help="number of MCMC samples")
parser.add_argument("-seed_initial", "--seed_initial", type=int, default=125,
                     help="initial seed for random generator to create seeds for repetitions")
parser.add_argument("-dataset", "--dataset", type=str, default="airfoil", help="dataset to use for the experiment")
parser.add_argument("-n_cores", "--n_cores", type=int, default=4, help="number of cores to use for parallel processing")
parser.add_argument("-kernel", "--kernel", type=str, default="RBF + Matern52", 
                    help="kernel to use for Gaussian Process regression in CREDO: 'RBF', 'Matern32', 'Matern52' or 'RationalQuadratic'")
parser.add_argument("-kernel_noise", "--kernel_noise", type=str, default="RBF", 
                    help="kernel to use for Gaussian Process noise in CREDO: 'RBF', 'Matern32', 'Matern52' or 'RationalQuadratic'")
parser.add_argument("-activation_noise", "--activation_noise", type=str, default="softplus", 
                    help="activation function for noise in Gaussian Process")
args = parser.parse_args()

alpha = args.alpha
gamma = args.gamma
n_rep = args.n_rep
n_MCMC = args.n_MCMC
seed_initial = args.seed_initial
dataset = args.dataset
n_cores = args.n_cores
kernel = args.kernel
kernel_noise = args.kernel_noise
activation_noise = args.activation_noise

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
        i,
        batch_size = 42,
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

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_calib = X_calib.to_numpy()
    y_calib = y_calib.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Fitting CREDO with GP
    print(f"Fitting CREDO adaptive with GP")
    credal_CP_gp = CredalCPRegressor(
        nc_type = 'Quantile',
        base_model = "GP",
        alpha = alpha,
        adaptive_gamma = False,
        gamma = gamma,
    )

    credal_CP_gp.fit(
        X_train,
        y_train,
        kernel = kernel_gp,
        normalize_y = True,
        scale = True,
        heteroscedastic = False,
    )
    credal_CP_gp.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)
    credo_CP_gp_pred = credal_CP_gp.predict(X_test)
#     print(credal_CP_gp.gamma_quantiles)

    del credal_CP_gp
    gc.collect()

    print(f"Fitting CREDO adaptive with QNN")
    # Fitting CREDO with QNN
    credal_CP_qnn = CredalCPRegressor(
        nc_type = 'Quantile',
        base_model = "QNN",
        alpha = alpha,
        adaptive_gamma = True,
        gamma = gamma,
        )
    
    credal_CP_qnn.fit(
        X_train, 
        y_train,
        weight_decay=1e-6,
        step_size=10,
        gamma=0.99,
        hidden_layers=[64, 64],
        dropout=0.3,
        epochs=2000,
        patience=50,
        lr=1e-3, 
        batch_size=batch_size,
        verbose=1,
        random_seed_fit=i,
    )

    credal_CP_qnn.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)
    credo_CP_qnn_pred = credal_CP_qnn.predict(X_test)

    del credal_CP_qnn
    gc.collect()

    
    # evaluating metrics of interest
    # marginal coverage
    cover_credo_gp = average_coverage(
        credo_CP_gp_pred[:, 1], credo_CP_gp_pred[:, 0], y_test
    )
    cover_credo_qnn = average_coverage(
        credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test
    )

    # ISL
    isl_credo_gp = average_interval_score_loss(
        credo_CP_gp_pred[:, 1], credo_CP_gp_pred[:, 0], y_test, alpha
    )
    isl_credo_qnn = average_interval_score_loss(
        credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test, alpha
    )

    # IL
    IL_credo_gp = np.mean(compute_interval_length(credo_CP_gp_pred[:, 1], credo_CP_gp_pred[:, 0]))
    IL_credo_qnn = np.mean(compute_interval_length(credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0]))

    # pcorr
    pcorr_credo_gp = corr_coverage_widths(
        credo_CP_gp_pred[:, 1], credo_CP_gp_pred[:, 0], y_test
    )
    pcorr_credo_qnn = corr_coverage_widths(
        credo_CP_qnn_pred[:, 1], credo_CP_qnn_pred[:, 0], y_test
    )
    
    IL_array = np.array([
        IL_credo_gp,
        IL_credo_qnn,
    ])
    isl_array = np.array([
        isl_credo_gp,
        isl_credo_qnn,
    ])
    cover_array = np.array([
        cover_credo_gp,
        cover_credo_qnn,
    ])
    pcorr_array = np.array([
        pcorr_credo_gp,
        pcorr_credo_qnn, 
    ])
    
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
    credo_gp_outliers = credo_CP_gp_pred[outlier_indexes]
    credo_qnn_outliers = credo_CP_qnn_pred[outlier_indexes]
    y_test_out = y_test[outlier_indexes]

    credo_gp_inliers = credo_CP_gp_pred[most_inlier_idxs]
    credo_qnn_inliers = credo_CP_qnn_pred[most_inlier_idxs]
    y_test_in = y_test[most_inlier_idxs]
    
    # evaluating metrics of interest
    # coverage for outliers
    cover_credo_gp_out = average_coverage(
        credo_CP_gp_pred[outlier_indexes][:, 1], 
        credo_CP_gp_pred[outlier_indexes][:, 0], 
        y_test_out
    )
    cover_credo_qnn_out = average_coverage(
        credo_CP_qnn_pred[outlier_indexes][:, 1], 
        credo_CP_qnn_pred[outlier_indexes][:, 0], 
        y_test_out
    )

    # ISL on outliers
    isl_credo_gp_out = average_interval_score_loss(
        credo_CP_gp_pred[outlier_indexes][:, 1], 
        credo_CP_gp_pred[outlier_indexes][:, 0], 
        y_test_out, alpha
    )
    isl_credo_qnn_out = average_interval_score_loss(
        credo_CP_qnn_pred[outlier_indexes][:, 1], 
        credo_CP_qnn_pred[outlier_indexes][:, 0], 
        y_test_out, alpha
    )
    
    # Interval length ratio
    credo_gp_ratio = np.mean(
            compute_interval_length(
                credo_gp_outliers[:, 1],
                credo_gp_outliers[:, 0]
            )
        ) / np.mean(
            compute_interval_length(
                credo_gp_inliers[:, 1],
                credo_gp_inliers[:, 0]
            )
        )
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
      
    
    isl_outlier_array = np.array([
        isl_credo_gp_out,
        isl_credo_qnn_out,
    ])
    cover_outlier_array = np.array([
        cover_credo_gp_out,
        cover_credo_qnn_out,
    ])
    ratio_array = np.array([
        credo_gp_ratio,
        credo_qnn_ratio,
    ])

    return cover_array, isl_array, IL_array, pcorr_array, cover_outlier_array, isl_outlier_array, ratio_array


def run_experiment(dataset, 
                   n_rep, 
                   target_column, 
                   prop_test = 0.2,
                   checkpoint_flag = False,
                   checkpoint_data = None,
                   checkpoint_data_outlier = None,
                   batch_size = 42,
):
    data = pd.read_csv(os.path.join(DATA_PATH, f"{dataset}.csv"))

    if data.shape[0] > 10000:
        batch_size = 120
    if dataset == "WEC":
        batch_size = 250

    if checkpoint_flag:
        resume_from = int(checkpoint_data.get("iteration", -1)) + 1
        cover_results = checkpoint_data.get("cover_results", [])
        isl_results = checkpoint_data.get("isl_results", [])
        IL_results = checkpoint_data.get("IL_results", [])
        pcorr_results = checkpoint_data.get("pcorr_results", [])
        seeds = checkpoint_data.get("seeds", None)
        
        ratio_results = checkpoint_data_outlier.get("ratio_results", [])
        coverage_outlier_results = checkpoint_data_outlier.get("coverage_results", [])
        isl_outlier_results = checkpoint_data_outlier.get("isl_results", [])
        print(f"Resuming from iteration {resume_from}. Loaded {len(coverage_outlier_results)} results so far.")

    else:
        resume_from = 0
        seeds = generate_seeds(seed_initial, n_rep)
        cover_results = []
        isl_results = []
        IL_results = []
        pcorr_results = []
        ratio_results = []
        coverage_outlier_results = []
        isl_outlier_results = []

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

        if dataset in ["blog"]:
            scale_y = True
        else:
            scale_y = False

        cover_array, isl_array, IL_array, pcorr_array, cover_outlier_array, isl_outlier_array, ratio_array = fit_methods(
        X_train,
        y_train, 
        X_calib, 
        y_calib, 
        X_test, 
        y_test,
        i,
        batch_size = batch_size,
        scale_y = scale_y,
        )
        cover_results.append(cover_array)
        isl_results.append(isl_array)
        IL_results.append(IL_array)
        pcorr_results.append(pcorr_array)
        coverage_outlier_results.append(cover_outlier_array)
        isl_outlier_results.append(isl_outlier_array)
        ratio_results.append(ratio_array)

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
                filepath = os.path.join(chk_dir,  f"{dataset}_checkpoint_gamma_adaptive.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
                checkpoint_outlier = {
                    "coverage_results": coverage_outlier_results,
                    "isl_results": isl_outlier_results,
                    "ratio_results": ratio_results,
                    "iteration": iteration,
                    "seeds": seeds,
                    "alpha": alpha,
                    "gamma": gamma,
                    "dataset": dataset,
                }
                filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_gamma_adaptive_outlier.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(checkpoint_outlier, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            except Exception as e:
                print(f"Failed saving checkpoint at iter {iteration+1}: {e}")
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

    methods = [
        "credo_GP",
        "credo_QNN",
        ]

    cover_mean, cover_sd = mean_sd(cover_results)
    isl_mean, isl_sd = mean_sd(isl_results)
    IL_mean, IL_sd = mean_sd(IL_results)
    pcorr_mean, pcorr_sd = mean_sd(pcorr_results)

    # create summary dataframes and save to CSV
    df_cover = pd.DataFrame({"methods": methods ,"mean": cover_mean, "sd": cover_sd})
    df_isl = pd.DataFrame({"methods": methods ,"mean": isl_mean, "sd": isl_sd})
    df_IL = pd.DataFrame({"methods": methods ,"mean": IL_mean, "sd": IL_sd})
    df_pcorr = pd.DataFrame({"methods": methods ,"mean": pcorr_mean, "sd": pcorr_sd})

    data_dir = os.path.join(RESULTS_PATH, f"{dataset}_gamma_adaptive_summary")
    os.makedirs(data_dir, exist_ok=True)

    df_cover.to_csv(os.path.join(data_dir, f"{dataset}_coverage_summary.csv"))
    df_isl.to_csv(os.path.join(data_dir, f"{dataset}_isl_summary.csv"))
    df_IL.to_csv(os.path.join(data_dir, f"{dataset}_IL_summary.csv"))
    df_pcorr.to_csv(os.path.join(data_dir, f"{dataset}_pcorr_summary.csv"))

    coverage_outlier_results = np.array(coverage_outlier_results)
    isl_outlier_results = np.array(isl_outlier_results)
    ratio_results = np.array(ratio_results)
    
    print("Saving outliers df")
    cover_mean_outlier, cover_sd_outlier = mean_sd(coverage_outlier_results)
    isl_mean_outlier, isl_sd_outlier = mean_sd(isl_outlier_results)
    ratio_mean_outlier, ratio_sd_outlier = mean_sd(ratio_results)

    df_cover_out = pd.DataFrame({"methods": methods ,"mean": cover_mean_outlier, "sd": cover_sd_outlier})
    df_isl_out = pd.DataFrame({"methods": methods ,"mean": isl_mean_outlier, "sd": isl_sd_outlier})
    df_ratio_out = pd.DataFrame({"methods": methods ,"mean": ratio_mean_outlier, "sd": ratio_sd_outlier})

    df_cover_out.to_csv(os.path.join(data_dir, f"{dataset}_coverage_outlier_summary.csv"))
    df_isl_out.to_csv(os.path.join(data_dir, f"{dataset}_isl_outlier_summary.csv"))
    df_ratio_out.to_csv(os.path.join(data_dir, f"{dataset}_ratio_outlier_summary.csv"))


    return np.array(cover_results), np.array(isl_results), \
        np.array(IL_results), np.array(pcorr_results), \
            np.array(coverage_outlier_results), np.array(isl_outlier_results), \
            np.array(ratio_results)


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
        kernel = (
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
    chk_file = os.path.join(chk_dir, f"{dataset}_checkpoint_gamma_adaptive.pkl")
    chk_file_outlier = os.path.join(chk_dir, f"{dataset}_checkpoint_gamma_adaptive_outlier.pkl")
    
    resume_from = 0
    checkpoint_data = None
    loaded_cover = loaded_isl = loaded_IL = loaded_pcorr = None
    loaded_seeds_so_far = None
    
    if os.path.exists(chk_file):
        # loading main experiment checkpoint
        try:
            with open(chk_file, "rb") as f:
                checkpoint_data = pickle.load(f)
            checkpoint_flag = True
            print(f"Found checkpoint for dataset '{dataset}'. Resuming from iteration {resume_from}.")
        except Exception as e:
            print(f"Failed to load checkpoint '{chk_file}': {e}")
            checkpoint_data = None
            checkpoint_flag = False

        # loading outlier checkpoint
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
        print(f"No checkpoint found at '{chk_file}'. Starting a new run.")
        checkpoint_flag = False
        checkpoint_data_outlier = None
    
    print("Running main experiment with outlier analysis at the same time")
    cover, isl, IL, pcorr, cover_out, isl_out, ratio_out = run_experiment(
        dataset = dataset, 
        n_rep = n_rep, 
        target_column = "target", 
        checkpoint_flag = checkpoint_flag, 
        checkpoint_data = checkpoint_data, 
        checkpoint_data_outlier=checkpoint_data_outlier, 
        )
            
    raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}")
    os.makedirs(raw_dir, exist_ok=True)
    to_save = {"cover": cover, 
                "isl": isl, 
                "IL": IL, 
                "pcorr": pcorr,
                "cover_out": cover_out,
                "isl_out": isl_out,
                "ratio_out": ratio_out
                }
    for name, arr in to_save.items(): 
        filepath = os.path.join(
            raw_dir, f"{dataset}_{name}_gamma_adaptive_raw.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    chk_file = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_gamma_adaptive.pkl")
    chk_file_out = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_gamma_adaptive_outlier.pkl")
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

