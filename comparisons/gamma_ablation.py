import argparse
import gc
import itertools
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from credo.credal_cp import CredalCPRegressor
from credo.utils import (
    average_coverage,
    average_interval_score_loss,
    average_interval_width,
    compute_interval_length,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
RESULTS_PATH = ROOT / "results"

DEFAULT_DATASETS = ["airfoil", "concrete", "winered", "winewhite"]
HIGH_DIM_DATASETS = ["meps19"]

QNN_HIDDEN_LAYERS = [64, 64, 32]
QNN_DROPOUT_CREDO = 0.1


def parse_float_grid(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def generate_seeds(seed_initial, n_rep):
    rng = np.random.default_rng(seed_initial)
    return rng.integers(0, 2**31 - 1, size=n_rep)


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
        step_size=10,
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


def outlier_slices(
    X_test,
    y_test,
    n_neighbors=15,
    contamination=0.05,
    inlier_size=0.2,
    n_components=2,
    tsne_random_state=120,
):
    tsne = TSNE(n_components=n_components, random_state=tsne_random_state)
    X_embedded = tsne.fit_transform(X_test)
    X_scaled = StandardScaler().fit_transform(X_embedded)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    out_pred = lof.fit_predict(X_scaled)

    outlier_indexes = np.where(out_pred == -1)[0]
    inlier_indexes = np.setdiff1d(np.arange(len(y_test)), outlier_indexes)
    inlier_scores = lof.negative_outlier_factor_[inlier_indexes]
    size = max(1, int((y_test.shape[0] - outlier_indexes.shape[0]) * inlier_size))
    most_inlier_idxs = inlier_indexes[np.argsort(inlier_scores)[::-1][:size]]
    return outlier_indexes, most_inlier_idxs


def evaluate_prediction(pred, y_test, outlier_indexes, inlier_indexes, alpha):
    high = pred[:, 1]
    low = pred[:, 0]
    outlier_high = high[outlier_indexes]
    outlier_low = low[outlier_indexes]
    inlier_high = high[inlier_indexes]
    inlier_low = low[inlier_indexes]

    outlier_length = np.mean(compute_interval_length(outlier_high, outlier_low))
    inlier_length = np.mean(compute_interval_length(inlier_high, inlier_low))
    ratio = outlier_length / inlier_length if inlier_length > 0 else np.nan

    return {
        "coverage": average_coverage(high, low, y_test),
        "smis": average_interval_score_loss(high, low, y_test, alpha),
        "interval_length": average_interval_width(high, low),
        "outlier_coverage": average_coverage(
            outlier_high,
            outlier_low,
            y_test[outlier_indexes],
        ),
        "outlier_inlier_ratio": ratio,
        "n_outliers": len(outlier_indexes),
        "n_inliers": len(inlier_indexes),
    }


def gamma_summary(model, X, gamma_max=None, gamma_min=None, tau=None):
    if not model.adaptive_gamma:
        return {}
    gamma_values = model.compute_gamma(
        X,
        gamma_max=gamma_max,
        gamma_min=gamma_min,
        tau=tau,
    )
    return {
        "gamma_x_mean": np.mean(gamma_values),
        "gamma_x_sd": np.std(gamma_values, ddof=1) if gamma_values.shape[0] > 1 else 0.0,
        "gamma_x_min": np.min(gamma_values),
        "gamma_x_max": np.max(gamma_values),
    }


def summarize(raw_df, group_columns):
    metric_columns = [
        "coverage",
        "smis",
        "interval_length",
        "outlier_coverage",
        "outlier_inlier_ratio",
        "gamma_x_mean",
        "gamma_x_sd",
        "gamma_x_min",
        "gamma_x_max",
    ]
    present_metrics = [col for col in metric_columns if col in raw_df.columns]
    grouped = raw_df.groupby(group_columns, dropna=False)
    mean_df = grouped[present_metrics].mean().add_suffix("_mean")
    sd_df = grouped[present_metrics].std(ddof=1).add_suffix("_sd")
    n_df = grouped.size().rename("n_rep_completed")
    return pd.concat([mean_df, sd_df, n_df], axis=1).reset_index()


def save_results(dataset, study, raw_rows, group_columns):
    output_dir = RESULTS_PATH / f"{dataset}_ablation_gamma_{study}_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df, group_columns)
    raw_df.to_csv(output_dir / f"{dataset}_ablation_gamma_{study}_raw.csv", index=False)
    summary_df.to_csv(output_dir / f"{dataset}_ablation_gamma_{study}_summary.csv", index=False)
    with open(output_dir / f"{dataset}_ablation_gamma_{study}_raw.pkl", "wb") as f:
        pickle.dump(raw_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    return raw_df, summary_df


def run_fixed_ablation(data, dataset, args, seeds):
    raw_rows = []
    batch_size = dataset_batch_size(dataset, data.shape[0])

    for rep_idx, seed in enumerate(tqdm(seeds, desc=f"{dataset}: fixed gamma")):
        torch.manual_seed(int(seed))
        X_train, X_calib, X_test, y_train, y_calib, y_test = split_dataset(
            data,
            args.target_column,
            int(seed),
            prop_test=args.prop_test,
            prop_train=args.prop_train,
        )
        outlier_indexes, inlier_indexes = outlier_slices(
            X_test,
            y_test,
            n_neighbors=args.outlier_neighbors,
            contamination=args.outlier_contamination,
            inlier_size=args.inlier_size,
            n_components=args.tsne_components,
            tsne_random_state=args.tsne_random_state,
        )

        model = fit_credo_qnn(
            X_train,
            y_train,
            alpha=args.alpha,
            batch_size=batch_size,
            fit_seed=rep_idx,
            adaptive_gamma=False,
            gamma=args.gamma_grid[0],
            heuristic_gamma=args.heuristic_gamma,
            k_gamma=args.k_gamma,
            epochs=args.epochs,
            patience=args.patience,
            verbose=args.verbose,
        )

        for config_idx, gamma in enumerate(args.gamma_grid):
            torch.manual_seed(int(seed) + config_idx)
            model.gamma = gamma
            model.calibrate(
                X_calib,
                y_calib,
                random_seed_calib=int(seed),
                N_samples_MC=args.n_mcmc,
            )
            torch.manual_seed(int(seed) + config_idx)
            pred = model.predict(X_test, n_samples=args.n_predict_samples)
            metrics = evaluate_prediction(pred, y_test, outlier_indexes, inlier_indexes, args.alpha)
            raw_rows.append(
                {
                    "dataset": dataset,
                    "study": "fixed",
                    "method": "credo_QNN_fixed",
                    "rep": rep_idx,
                    "seed": int(seed),
                    "gamma": gamma,
                    **metrics,
                }
            )

        del model
        gc.collect()

    return save_results(dataset, "fixed", raw_rows, ["dataset", "study", "method", "gamma"])


def run_adaptive_ablation(data, dataset, args, seeds):
    raw_rows = []
    batch_size = dataset_batch_size(dataset, data.shape[0])
    grid = list(itertools.product(args.gamma_min_grid, args.gamma_max_grid, args.tau_grid))
    grid = [
        (gamma_min, gamma_max, tau)
        for gamma_min, gamma_max, tau in grid
        if gamma_min < gamma_max
    ]

    for rep_idx, seed in enumerate(tqdm(seeds, desc=f"{dataset}: adaptive gamma")):
        torch.manual_seed(int(seed))
        X_train, X_calib, X_test, y_train, y_calib, y_test = split_dataset(
            data,
            args.target_column,
            int(seed),
            prop_test=args.prop_test,
            prop_train=args.prop_train,
        )
        outlier_indexes, inlier_indexes = outlier_slices(
            X_test,
            y_test,
            n_neighbors=args.outlier_neighbors,
            contamination=args.outlier_contamination,
            inlier_size=args.inlier_size,
            n_components=args.tsne_components,
            tsne_random_state=args.tsne_random_state,
        )

        base_gamma = min(args.gamma_min_grid)
        model = fit_credo_qnn(
            X_train,
            y_train,
            alpha=args.alpha,
            batch_size=batch_size,
            fit_seed=rep_idx,
            adaptive_gamma=True,
            gamma=base_gamma,
            heuristic_gamma=args.heuristic_gamma,
            k_gamma=args.k_gamma,
            epochs=args.epochs,
            patience=args.patience,
            verbose=args.verbose,
        )

        for config_idx, (gamma_min, gamma_max, tau) in enumerate(grid):
            torch.manual_seed(int(seed) + config_idx)
            model.gamma = gamma_min
            model.calibrate(
                X_calib,
                y_calib,
                random_seed_calib=int(seed),
                N_samples_MC=args.n_mcmc,
                gamma_max=gamma_max,
                gamma_min=gamma_min,
                tau=tau,
            )
            torch.manual_seed(int(seed) + config_idx)
            pred = model.predict(X_test, n_samples=args.n_predict_samples)
            metrics = evaluate_prediction(pred, y_test, outlier_indexes, inlier_indexes, args.alpha)
            raw_rows.append(
                {
                    "dataset": dataset,
                    "study": "adaptive",
                    "method": "credo_QNN_adaptive",
                    "rep": rep_idx,
                    "seed": int(seed),
                    "gamma_min": gamma_min,
                    "gamma_max": gamma_max,
                    "tau_gamma": tau,
                    **gamma_summary(model, X_test, gamma_max=gamma_max, gamma_min=gamma_min, tau=tau),
                    **metrics,
                }
            )

        del model
        gc.collect()

    return save_results(
        dataset,
        "adaptive",
        raw_rows,
        ["dataset", "study", "method", "gamma_min", "gamma_max", "tau_gamma"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gamma fixed/adaptive ablation studies for CREDO QNN."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--target_column", default="target")
    parser.add_argument("--mode", choices=["fixed", "adaptive", "both"], default="both")
    parser.add_argument("--include_high_dim", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_rep", type=int, default=10)
    parser.add_argument("--n_mcmc", type=int, default=1000)
    parser.add_argument("--n_predict_samples", type=int, default=500)
    parser.add_argument("--seed_initial", type=int, default=125)
    parser.add_argument("--prop_test", type=float, default=0.2)
    parser.add_argument("--prop_train", type=float, default=0.7)
    parser.add_argument(
        "--gamma_grid",
        type=parse_float_grid,
        default=parse_float_grid("0.01,0.025,0.05,0.1,0.2,0.35,0.5,0.75"),
    )
    parser.add_argument(
        "--gamma_min_grid",
        type=parse_float_grid,
        default=parse_float_grid("0.01, 0.05,0.1"),
    )
    parser.add_argument(
        "--gamma_max_grid",
        type=parse_float_grid,
        default=parse_float_grid("0.5,0.75,0.9"),
    )
    parser.add_argument(
        "--tau_grid",
        type=parse_float_grid,
        default=parse_float_grid("0.5,1.0,2.0"),
    )
    parser.add_argument("--heuristic_gamma", choices=["log", "exp"], default="log")
    parser.add_argument("--k_gamma", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--outlier_neighbors", type=int, default=15)
    parser.add_argument("--outlier_contamination", type=float, default=0.05)
    parser.add_argument("--inlier_size", type=float, default=0.2)
    parser.add_argument("--tsne_components", type=int, default=2)
    parser.add_argument("--tsne_random_state", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.include_high_dim:
        args.datasets = list(dict.fromkeys(args.datasets + HIGH_DIM_DATASETS))

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    seeds = generate_seeds(args.seed_initial, args.n_rep)

    for dataset in args.datasets:
        data_file = DATA_PATH / f"{dataset}.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {data_file}")
        data = pd.read_csv(data_file)

        if args.mode in ["fixed", "both"]:
            run_fixed_ablation(data, dataset, args, seeds)
        if args.mode in ["adaptive", "both"]:
            run_adaptive_ablation(data, dataset, args, seeds)


if __name__ == "__main__":
    main()
