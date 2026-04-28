import argparse
import gc
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from credo.credal_cp import CredalCPRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
RESULTS_PATH = ROOT / "results"

DEFAULT_DATASETS = [
    "concrete",
    "airfoil",
    "winewhite",
    "star",
    "winered",
    "cycle",
    "electric",
    "meps19",
]

QNN_HIDDEN_LAYERS = [128, 128, 64]
QNN_DROPOUT_CREDO = 0.3
METHODS = {
    "credo_QNN": {"adaptive_gamma": False},
    "credo_QNN_adaptive": {"adaptive_gamma": True},
}


def generate_seeds(seed_initial, n_rep):
    np.random.seed(seed_initial)
    return np.random.randint(0, 2**31 - 1, size=n_rep)


def dataset_batch_size(dataset, n_rows):
    if dataset == "WEC":
        return 250
    if n_rows > 10000:
        return 125
    return 32


def split_dataset(data, target_column, seed, prop_test=0.2, prop_train=0.7):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train_calib, X_test, y_train_calib, y_test = train_test_split(
        X, y, test_size=prop_test, random_state=seed
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_calib,
        y_train_calib,
        test_size=1 - prop_train,
        random_state=seed,
    )
    return (
        X_train.to_numpy(),
        X_calib.to_numpy(),
        X_test.to_numpy(),
        y_train.to_numpy(),
        y_calib.to_numpy(),
        y_test.to_numpy(),
    )


def maybe_scale_y(dataset, y_train, y_calib, y_test):
    if dataset not in {"blog"}:
        return y_train, y_calib, y_test
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_calib = scaler.transform(y_calib.reshape(-1, 1)).ravel()
    y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()
    return y_train, y_calib, y_test


def fit_credo_qnn(
    X_train,
    y_train,
    alpha,
    batch_size,
    fit_seed,
    adaptive_gamma,
    gamma,
    heuristic_gamma,
    k_gamma,
    epochs,
    patience,
    step_size,
    verbose,
):
    model = CredalCPRegressor(
        nc_type="Quantile",
        base_model="QNN",
        alpha=alpha,
        adaptive_gamma=adaptive_gamma,
        gamma=gamma,
    )
    model.fit(
        X_train,
        y_train,
        weight_decay=1e-6,
        step_size=step_size,
        gamma=0.99,
        hidden_layers=QNN_HIDDEN_LAYERS,
        dropout=QNN_DROPOUT_CREDO,
        epochs=epochs,
        patience=patience,
        lr=1e-3,
        batch_size=batch_size,
        verbose=verbose,
        random_seed_fit=int(fit_seed),
        heuristic_gamma=heuristic_gamma,
        k=k_gamma,
    )
    return model


def ratio_from_model(model, X_test, n_predict_samples, eps):
    _, _, epistemic_unc = model.predict(
        X_test,
        n_samples=n_predict_samples,
        disentangle=True,
    )
    epistemic_unc = np.asarray(epistemic_unc, dtype=float).ravel()
    denom = np.maximum(epistemic_unc, eps)

    cutoff = np.asarray(model.cutoff, dtype=float).ravel()
    if cutoff.size == 1:
        cutoff_vec = np.full_like(denom, cutoff.item(), dtype=float)
        cutoff_value = cutoff.item()
    else:
        cutoff_vec = cutoff[: denom.shape[0]]
        cutoff_value = float(np.nanmean(cutoff_vec))

    return cutoff_vec / denom, epistemic_unc, cutoff_value


def align_vectors(vectors):
    arrays = [np.asarray(vector, dtype=float).ravel() for vector in vectors]
    if not arrays:
        return np.empty((0, 0))
    min_len = min(array.shape[0] for array in arrays)
    if any(array.shape[0] != min_len for array in arrays):
        print(f"Warning: unequal test vector lengths; truncating to {min_len}.")
    return np.vstack([array[:min_len] for array in arrays])


def summarize_method(dataset, method, ratio_means, cutoff_values, epistemic_means, ratio_vectors):
    ratio_means = np.asarray(ratio_means, dtype=float)
    cutoff_values = np.asarray(cutoff_values, dtype=float)
    epistemic_means = np.asarray(epistemic_means, dtype=float)
    ratio_matrix = align_vectors(ratio_vectors)

    n_rep = ratio_means.shape[0]
    ratio_sd = np.nanstd(ratio_means, ddof=1) if n_rep > 1 else 0.0
    cutoff_sd = np.nanstd(cutoff_values, ddof=1) if n_rep > 1 else 0.0
    epistemic_sd = np.nanstd(epistemic_means, ddof=1) if n_rep > 1 else 0.0

    return {
        "dataset": dataset,
        "method": method,
        "mean": float(np.nanmean(ratio_means)),
        "sd": float(ratio_sd),
        "se2": float(2 * ratio_sd / np.sqrt(n_rep)) if n_rep > 0 else np.nan,
        "cutoff_mean": float(np.nanmean(cutoff_values)),
        "cutoff_sd": float(cutoff_sd),
        "epistemic_uncertainty_mean": float(np.nanmean(epistemic_means)),
        "epistemic_uncertainty_sd": float(epistemic_sd),
        "n_rep_completed": int(n_rep),
        "ratio_obs_mean": np.nanmean(ratio_matrix, axis=0) if ratio_matrix.size else np.array([]),
        "ratio_matrix": ratio_matrix,
    }


def save_dataset_results(dataset, summaries, raw_payload):
    summary_dir = RESULTS_PATH / f"{dataset}_cutoff_epistemic_ratio_summary"
    obs_dir = RESULTS_PATH / f"{dataset}_cutoff_epistemic_ratio_by_observation"
    raw_dir = RESULTS_PATH / "raw" / dataset
    summary_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for summary in summaries:
        method = summary["method"]
        obs_df = pd.DataFrame({"cutoff_epistemic_ratio": summary["ratio_obs_mean"]})
        obs_df.to_csv(
            obs_dir / f"{dataset}_{method}_cutoff_epistemic_ratio_obs_mean.csv",
            index=False,
        )

        row = {key: value for key, value in summary.items() if key not in {"ratio_obs_mean", "ratio_matrix"}}
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        summary_dir / f"{dataset}_cutoff_epistemic_ratio_summary.csv",
        index=False,
    )

    with open(raw_dir / f"{dataset}_cutoff_epistemic_ratio_raw.pkl", "wb") as f:
        pickle.dump(raw_payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_dataset(dataset, args, seeds):
    data = pd.read_csv(DATA_PATH / f"{dataset}.csv")
    batch_size = dataset_batch_size(dataset, data.shape[0])
    method_results = {
        method: {
            "ratio_means": [],
            "ratio_vectors": [],
            "cutoff_values": [],
            "epistemic_means": [],
        }
        for method in METHODS
    }

    for rep_idx, seed in enumerate(tqdm(seeds, desc=f"{dataset}: cutoff/epistemic ratio")):
        X_train, X_calib, X_test, y_train, y_calib, y_test = split_dataset(
            data,
            args.target_column,
            int(seed),
            prop_test=args.prop_test,
            prop_train=args.prop_train,
        )
        y_train, y_calib, _ = maybe_scale_y(dataset, y_train, y_calib, y_test)

        for method_idx, (method, config) in enumerate(METHODS.items()):
            model = fit_credo_qnn(
                X_train,
                y_train,
                alpha=args.alpha,
                batch_size=batch_size,
                fit_seed=rep_idx,
                adaptive_gamma=config["adaptive_gamma"],
                gamma=args.gamma,
                heuristic_gamma=args.heuristic_gamma,
                k_gamma=args.k_gamma,
                epochs=args.epochs,
                patience=args.patience,
                step_size=args.step_size,
                verbose=args.verbose,
            )
            model.calibrate(
                X_calib,
                y_calib,
                N_samples_MC=args.n_mcmc,
                gamma_max=args.gamma_max,
                gamma_min=args.gamma_min if config["adaptive_gamma"] else None,
                tau=args.tau_gamma,
            )
            ratio, epistemic_unc, cutoff_value = ratio_from_model(
                model,
                X_test,
                n_predict_samples=args.n_predict_samples,
                eps=args.eps,
            )

            method_results[method]["ratio_means"].append(float(np.nanmean(ratio)))
            method_results[method]["ratio_vectors"].append(ratio)
            method_results[method]["cutoff_values"].append(float(cutoff_value))
            method_results[method]["epistemic_means"].append(float(np.nanmean(epistemic_unc)))

            del model
            gc.collect()

    summaries = [
        summarize_method(
            dataset,
            method,
            values["ratio_means"],
            values["cutoff_values"],
            values["epistemic_means"],
            values["ratio_vectors"],
        )
        for method, values in method_results.items()
    ]

    raw_payload = {
        "dataset": dataset,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "gamma_min": args.gamma_min,
        "gamma_max": args.gamma_max,
        "tau_gamma": args.tau_gamma,
        "eps": args.eps,
        "seeds": seeds,
        "run_config": {
            "qnn_hidden_layers": QNN_HIDDEN_LAYERS,
            "qnn_dropout_credo": QNN_DROPOUT_CREDO,
            "weight_decay": 1e-6,
            "step_size": args.step_size,
            "scheduler_gamma": 0.99,
            "lr": 1e-3,
            "epochs": args.epochs,
            "patience": args.patience,
            "n_mcmc": args.n_mcmc,
            "n_predict_samples": args.n_predict_samples,
            "prop_test": args.prop_test,
            "prop_train": args.prop_train,
            "seed_initial": args.seed_initial,
            "seed_generator": "np.random.seed(seed_initial); np.random.randint",
        },
        "results": method_results,
    }
    save_dataset_results(dataset, summaries, raw_payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute estimated cutoff / epistemic uncertainty ratios for CREDO QNN."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--target_column", default="target")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--gamma_min", type=float, default=None)
    parser.add_argument("--gamma_max", type=float, default=0.75)
    parser.add_argument("--tau_gamma", type=float, default=1.0)
    parser.add_argument("--n_rep", type=int, default=30)
    parser.add_argument("--n_mcmc", type=int, default=1000)
    parser.add_argument("--n_predict_samples", type=int, default=500)
    parser.add_argument("--seed_initial", type=int, default=125)
    parser.add_argument("--prop_test", type=float, default=0.2)
    parser.add_argument("--prop_train", type=float, default=0.7)
    parser.add_argument("--heuristic_gamma", choices=["log", "exp"], default="log")
    parser.add_argument("--k_gamma", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed_initial)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed_initial)
    seeds = generate_seeds(args.seed_initial, args.n_rep)
    for dataset in args.datasets:
        if not (DATA_PATH / f"{dataset}.csv").exists():
            print(f"Skipping {dataset}: data file not found.")
            continue
        run_dataset(dataset, args, seeds)


if __name__ == "__main__":
    main()
