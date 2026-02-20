import os
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import multiprocessing as mp

original_path = os.getcwd()
os.chdir(os.path.join(original_path, "comparisons"))

from credal_cp.credal_cp import CredalCPRegressor
from jaxtyping import install_import_hook
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

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

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_calib = X_calib.to_numpy()
    y_calib = y_calib.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Fitting CREDO with GP
    print(f"Fitting CREDO vanilla with GP")
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

    print(f"Fitting CREDO vanilla with QNN")
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
        hidden_layers=[64, 64],
        dropout=0.3,
        epochs=2000,
        patience=50,
        lr=1e-3, 
        batch_size=mdn_params["batch_size"],
        verbose=1,
        random_seed_fit=i,
    )

    credal_CP_qnn.calibrate(X_calib, y_calib, N_samples_MC=n_MCMC)

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

    print("Disentangling uncertainties for inliers and outliers for both methods")
    _, aleat_unc_inlier_gp, epis_unc_inlier_gp = credal_CP_gp.predict(X_test[most_inlier_idxs], disentangle=True)
    _, aleat_unc_outlier_gp, epis_unc_outlier_gp = credal_CP_gp.predict(X_test[outlier_indexes], disentangle=True)

    _, aleat_unc_inlier_qnn, epis_unc_inlier_qnn = credal_CP_qnn.predict(X_test[most_inlier_idxs], disentangle=True)
    _, aleat_unc_outlier_qnn, epis_unc_outlier_qnn = credal_CP_qnn.predict(X_test[outlier_indexes], disentangle=True)

    del credal_CP_gp, credal_CP_qnn
    gc.collect()

    # normalizing uncertainties to be able to compare inliers and outliers
    total_gp = aleat_unc_inlier_gp + epis_unc_inlier_gp
    epis_unc_inlier_gp /= total_gp

    total_gp_outlier = aleat_unc_outlier_gp + epis_unc_outlier_gp
    epis_unc_outlier_gp /= total_gp_outlier

    total_qnn = aleat_unc_inlier_qnn + epis_unc_inlier_qnn
    epis_unc_inlier_qnn /= total_qnn

    total_qnn_outlier = aleat_unc_outlier_qnn + epis_unc_outlier_qnn
    epis_unc_outlier_qnn /= total_qnn_outlier

    # returning the mean epistemic uncertainty for inliers and outliers for both models to be able to compare them
    print("Summarizing epistemic uncertainty over inliers and outliers for both methods")
    epis_unc_inlier_gp_mean, epis_unc_outlier_gp_mean = np.mean(epis_unc_inlier_gp), np.mean(epis_unc_outlier_gp)
    epis_unc_inlier_qnn_mean, epis_unc_outlier_qnn_mean = np.mean(epis_unc_inlier_qnn), np.mean(epis_unc_outlier_qnn)

    return epis_unc_inlier_gp_mean, epis_unc_outlier_gp_mean, epis_unc_inlier_qnn_mean, epis_unc_outlier_qnn_mean

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
        mdn_params["batch_size"] = 120
    if dataset == "WEC":
        mdn_params["batch_size"] = 250

    if checkpoint_flag:
        resume_from = int(checkpoint_data.get("iteration", -1)) + 1
        epis_unc_inlier_gp_results = checkpoint_data.get("epis_unc_inlier_gp", [])
        epis_unc_outlier_gp_results = checkpoint_data.get("epis_unc_outlier_gp", [])
        epis_unc_inlier_qnn_results = checkpoint_data.get("epis_unc_inlier_qnn", [])
        epis_unc_outlier_qnn_results = checkpoint_data.get("epis_unc_outlier_qnn", [])
        seeds = checkpoint_data.get("seeds", None)
        print(f"Resuming from iteration {resume_from}. Loaded {len(epis_unc_inlier_gp_results)} results so far.")
    else:
        resume_from = 0
        seeds = generate_seeds(seed_initial, n_rep)
        epis_unc_inlier_gp_results = []
        epis_unc_outlier_gp_results = []
        epis_unc_inlier_qnn_results = []
        epis_unc_outlier_qnn_results = []


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

        epis_unc_inlier_gp, epis_unc_outlier_gp, epis_unc_inlier_qnn, epis_unc_outlier_qnn = fit_methods(
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
        epis_unc_inlier_gp_results.append(epis_unc_inlier_gp)
        epis_unc_outlier_gp_results.append(epis_unc_outlier_gp)
        epis_unc_inlier_qnn_results.append(epis_unc_inlier_qnn)
        epis_unc_outlier_qnn_results.append(epis_unc_outlier_qnn)

        def save_checkpoint(iteration, seeds):
            try:
                checkpoint = {
                    "epis_unc_inlier_gp": epis_unc_inlier_gp_results,
                    "epis_unc_outlier_gp": epis_unc_outlier_gp_results,
                    "epis_unc_inlier_qnn": epis_unc_inlier_qnn_results,
                    "epis_unc_outlier_qnn": epis_unc_outlier_qnn_results,
                    "iteration": iteration,
                    "seeds": seeds,
                    "alpha": alpha,
                    "gamma": gamma,
                    "dataset": dataset,
                }
                chk_dir = os.path.join(RESULTS_PATH, "checkpoints")
                os.makedirs(chk_dir, exist_ok=True)
                filepath = os.path.join(chk_dir, f"{dataset}_checkpoint_unc.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"Failed saving checkpoint at iter {iteration+1}: {e}")

        # save checkpoint after each repetition
        save_checkpoint(i, seeds)
    
    # summarize results: convert lists to arrays and compute mean and sd (sample sd if n_rep>1)
    epis_unc_inlier_gp_results = np.array(epis_unc_inlier_gp_results)
    epis_unc_outlier_gp_results = np.array(epis_unc_outlier_gp_results)
    epis_unc_inlier_qnn_results = np.array(epis_unc_inlier_qnn_results)
    epis_unc_outlier_qnn_results = np.array(epis_unc_outlier_qnn_results)

    def mean_sd(arr):
        mean = np.nanmean(arr, axis=0, )
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        return mean, sd

    methods = [
        "credo_GP",
        "credo_GP",
        "credo_QNN",
        "credo_QNN",
        ]
    
    types = [
        "inliers",
        "outliers",
        "inliers",
        "outliers",
    ]

    epis_unc_inlier_gp_mean, epis_unc_inlier_gp_sd = mean_sd(epis_unc_inlier_gp_results)
    epis_unc_outlier_gp_mean, epis_unc_outlier_gp_sd = mean_sd(epis_unc_outlier_gp_results)
    epis_unc_inlier_qnn_mean, epis_unc_inlier_qnn_sd = mean_sd(epis_unc_inlier_qnn_results)
    epis_unc_outlier_qnn_mean, epis_unc_outlier_qnn_sd = mean_sd(epis_unc_outlier_qnn_results)

    # concatenate inlier/outlier results into flat arrays
    mean_all_combined = np.concatenate((
        epis_unc_inlier_gp_mean,
        epis_unc_outlier_gp_mean,
        epis_unc_inlier_qnn_mean,
        epis_unc_outlier_qnn_mean,
    ))

    sd_all_combined = np.concatenate((
        epis_unc_inlier_gp_sd,
        epis_unc_outlier_gp_sd,
        epis_unc_inlier_qnn_sd,
        epis_unc_outlier_qnn_sd,
    ))

    # create summary dataframes and save to CSV
    general_df = pd.DataFrame({"methods": methods ,
                          "mean": mean_all_combined, 
                          "sd": sd_all_combined,
                          "type": types,
                          })
    data_dir = os.path.join(RESULTS_PATH, f"{dataset}_unc_summary")
    os.makedirs(data_dir, exist_ok=True)

    general_df.to_csv(os.path.join(data_dir, f"{dataset}_general_summary.csv"))
    
    return np.array(epis_unc_inlier_gp_results), np.array(epis_unc_outlier_gp_results), \
            np.array(epis_unc_inlier_qnn_results), np.array(epis_unc_outlier_qnn_results)



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
    chk_file = os.path.join(chk_dir, f"{dataset}_checkpoint_unc.pkl")

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
        checkpoint_data = None
        checkpoint_flag = False
    
    # Run the experiment
    epis_unc_inlier_gp, epis_unc_outlier_gp, epis_unc_inlier_qnn, epis_unc_outlier_qnn = run_experiment(
            dataset = dataset, 
            n_rep = n_rep, 
            target_column = "target",
            checkpoint_flag = checkpoint_flag,
            checkpoint_data = checkpoint_data,
            )
    
    raw_dir = os.path.join(RESULTS_PATH, f"raw/{dataset}")
    os.makedirs(raw_dir, exist_ok=True)

    to_save = {"epis_unc_inlier_gp": epis_unc_inlier_gp, "epis_unc_outlier_gp": epis_unc_outlier_gp,
               "epis_unc_inlier_qnn": epis_unc_inlier_qnn, "epis_unc_outlier_qnn": epis_unc_outlier_qnn}
    for name, arr in to_save.items():
        filepath = os.path.join(raw_dir, f"{dataset}_{name}_raw_unc.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    chk_file = os.path.join(RESULTS_PATH, "checkpoints", f"{dataset}_checkpoint_unc.pkl")
    try:
        if os.path.exists(chk_file):
            os.remove(chk_file)
            chk_dir = os.path.dirname(chk_file)
            if os.path.isdir(chk_dir) and not os.listdir(chk_dir):
                os.rmdir(chk_dir)
    except Exception as e:
        print(f"Failed to delete checkpoint {chk_file}: {e}")



