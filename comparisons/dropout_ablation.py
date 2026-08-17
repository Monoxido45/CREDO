"""Sensitivity analysis for the CREDO MC-dropout envelope.

The QNN architecture, data splits, and training protocol are fixed. The
dropout rate is varied only in the posterior-sampling/envelope model, with
dropout explicitly disabled during QNN training.
"""

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from credo.credal_cp import CredalCPRegressor
from credo.utils import (
    average_coverage,
    average_interval_score_loss,
    average_interval_width,
    compute_interval_length,
)
from gamma_ablation import (
    DATA_PATH,
    DEFAULT_DATASETS,
    QNN_HIDDEN_LAYERS,
    dataset_batch_size,
    generate_seeds,
    split_dataset,
)
from outlier_detection import select_outlier_inlier_indices


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results"


def parse_float_grid(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def fit_credo_qnn(
    X_train,
    y_train,
    alpha,
    batch_size,
    fit_seed,
    adaptive_gamma,
    gamma,
    dropout,
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
        dropout=dropout,
        epochs=epochs,
        patience=patience,
        lr=1e-3,
        batch_size=batch_size,
        verbose=verbose,
        random_seed_fit=int(fit_seed),
        heuristic_gamma=heuristic_gamma,
        k=k_gamma,
        dropout_during_fit=False,
    )
    return model


def evaluate_outlier_metrics(pred, y_test, outlier_indexes, inlier_indexes, alpha):
    high = pred[:, 1]
    low = pred[:, 0]
    outlier_high = high[outlier_indexes]
    outlier_low = low[outlier_indexes]
    inlier_high = high[inlier_indexes]
    inlier_low = low[inlier_indexes]
    outlier_length = np.mean(compute_interval_length(outlier_high, outlier_low))
    inlier_length = np.mean(compute_interval_length(inlier_high, inlier_low))
    return {
        "smis": average_interval_score_loss(high, low, y_test, alpha),
        "interval_length": average_interval_width(high, low),
        "outlier_coverage": average_coverage(
            outlier_high, outlier_low, y_test[outlier_indexes]
        ),
        "outlier_inlier_ratio": (
            outlier_length / inlier_length if inlier_length > 0 else np.nan
        ),
    }


def summarize(raw_df):
    group_columns = ["dataset", "study", "detector", "dropout"]
    metric_columns = [
        "smis",
        "interval_length",
        "outlier_coverage",
        "outlier_inlier_ratio",
    ]
    grouped = raw_df.groupby(group_columns, dropna=False)
    means = grouped[metric_columns].mean().add_suffix("_mean")
    sds = grouped[metric_columns].std(ddof=1).add_suffix("_sd")
    counts = grouped.size().rename("n_rep_completed")
    return pd.concat([means, sds, counts], axis=1).reset_index()


def save_results(dataset, study, raw_rows):
    output_dir = RESULTS_PATH / f"{dataset}_ablation_dropout_{study}_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df)
    raw_df.to_csv(output_dir / f"{dataset}_ablation_dropout_{study}_raw.csv", index=False)
    summary_df.to_csv(output_dir / f"{dataset}_ablation_dropout_{study}_summary.csv", index=False)
    return raw_df, summary_df


def detector_indices(X_test, y_test, args, detector, seed):
    return select_outlier_inlier_indices(
        X_test,
        y_test,
        outlier_detector=detector,
        outlier_embedding=args.outlier_embedding,
        contamination=args.outlier_contamination,
        inlier_size=args.inlier_size,
        n_neighbors=args.outlier_neighbors,
        n_components=args.tsne_components,
        tsne_random_state=args.tsne_random_state,
        iforest_n_estimators=args.iforest_n_estimators,
        iforest_max_samples=args.iforest_max_samples,
        iforest_max_features=args.iforest_max_features,
        iforest_bootstrap=args.iforest_bootstrap,
        random_state=int(seed),
    )


def run_study(data, dataset, args, seeds, adaptive_gamma):
    raw_rows = []
    study = "adaptive" if adaptive_gamma else "fixed"
    batch_size = dataset_batch_size(dataset, data.shape[0])

    for rep_idx, seed in enumerate(tqdm(seeds, desc=f"{dataset}: dropout {study}")):
        torch.manual_seed(int(seed))
        X_train, X_calib, X_test, y_train, y_calib, y_test = split_dataset(
            data,
            args.target_column,
            int(seed),
            prop_test=args.prop_test,
            prop_train=args.prop_train,
        )
        detector_slices = {
            detector: detector_indices(X_test, y_test, args, detector, seed)
            for detector in args.detectors
        }

        for dropout_idx, dropout in enumerate(args.dropout_grid):
            model = fit_credo_qnn(
                X_train,
                y_train,
                alpha=args.alpha,
                batch_size=batch_size,
                fit_seed=rep_idx,
                adaptive_gamma=adaptive_gamma,
                gamma=args.fixed_gamma if not adaptive_gamma else args.gamma_min,
                dropout=dropout,
                heuristic_gamma=args.heuristic_gamma,
                k_gamma=args.k_gamma,
                epochs=args.epochs,
                patience=args.patience,
                verbose=args.verbose,
            )
            torch.manual_seed(int(seed) + dropout_idx)
            if adaptive_gamma:
                model.calibrate(
                    X_calib,
                    y_calib,
                    random_seed_calib=int(seed),
                    N_samples_MC=args.n_mcmc,
                    gamma_max=args.gamma_max,
                    gamma_min=args.gamma_min,
                    tau=args.tau_gamma,
                )
            else:
                model.gamma = args.fixed_gamma
                model.calibrate(
                    X_calib,
                    y_calib,
                    random_seed_calib=int(seed),
                    N_samples_MC=args.n_mcmc,
                )
            torch.manual_seed(int(seed) + dropout_idx)
            pred = model.predict(X_test, n_samples=args.n_predict_samples)

            for detector, (outlier_indexes, inlier_indexes) in detector_slices.items():
                metrics = evaluate_outlier_metrics(
                    pred,
                    y_test,
                    outlier_indexes,
                    inlier_indexes,
                    args.alpha,
                )
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "study": study,
                        "detector": detector,
                        "rep": rep_idx,
                        "seed": int(seed),
                        "dropout": dropout,
                        "architecture": "64-64-32",
                        "dropout_during_fit": False,
                        "gamma": args.fixed_gamma if not adaptive_gamma else np.nan,
                        "gamma_min": args.gamma_min if adaptive_gamma else np.nan,
                        "gamma_max": args.gamma_max if adaptive_gamma else np.nan,
                        "tau_gamma": args.tau_gamma if adaptive_gamma else np.nan,
                        **metrics,
                    }
                )
            del model
            gc.collect()

    return save_results(dataset, study, raw_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="CREDO dropout sensitivity analysis.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS + ["meps19"])
    parser.add_argument("--target_column", default="target")
    parser.add_argument("--mode", choices=["fixed", "adaptive", "both"], default="both")
    parser.add_argument("--dropout_grid", type=parse_float_grid, default=parse_float_grid("0.05,0.1,0.2,0.3,0.5"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_rep", type=int, default=15)
    parser.add_argument("--n_mcmc", type=int, default=1000)
    parser.add_argument("--n_predict_samples", type=int, default=500)
    parser.add_argument("--seed_initial", type=int, default=125)
    parser.add_argument("--prop_test", type=float, default=0.2)
    parser.add_argument("--prop_train", type=float, default=0.7)
    parser.add_argument("--fixed_gamma", type=float, default=0.2)
    parser.add_argument("--gamma_min", type=float, default=0.1)
    parser.add_argument("--gamma_max", type=float, default=0.9)
    parser.add_argument("--tau_gamma", type=float, default=1.0)
    parser.add_argument("--heuristic_gamma", choices=["log", "exp"], default="log")
    parser.add_argument("--k_gamma", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--detectors", nargs="+", choices=["lof", "isolation_forest"], default=["lof", "isolation_forest"])
    parser.add_argument("--outlier_embedding", choices=["tsne", "original"], default="tsne")
    parser.add_argument("--outlier_neighbors", type=int, default=15)
    parser.add_argument("--outlier_contamination", type=float, default=0.05)
    parser.add_argument("--inlier_size", type=float, default=0.2)
    parser.add_argument("--tsne_components", type=int, default=2)
    parser.add_argument("--tsne_random_state", type=int, default=120)
    parser.add_argument("--iforest_n_estimators", type=int, default=200)
    parser.add_argument("--iforest_max_samples", default="auto")
    parser.add_argument("--iforest_max_features", type=float, default=1.0)
    parser.add_argument("--iforest_bootstrap", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    seeds = generate_seeds(args.seed_initial, args.n_rep)
    for dataset in args.datasets:
        data_file = DATA_PATH / f"{dataset}.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {data_file}")
        data = pd.read_csv(data_file)
        if args.mode in ["fixed", "both"]:
            run_study(data, dataset, args, seeds, adaptive_gamma=False)
        if args.mode in ["adaptive", "both"]:
            run_study(data, dataset, args, seeds, adaptive_gamma=True)


if __name__ == "__main__":
    main()
