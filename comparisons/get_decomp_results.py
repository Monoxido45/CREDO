# Code for generating disentanglement result plots.
from argparse import ArgumentParser
from math import ceil
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from get_results import QNN_DATASET_ORDER
except ImportError:
    from comparisons.get_results import QNN_DATASET_ORDER


ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "paper_results" / "figures"
DEFAULT_DATASETS = QNN_DATASET_ORDER


def detector_suffix(outlier_detector):
    return "" if outlier_detector == "lof" else f"_{outlier_detector}"


def available_disentanglement_datasets(outlier_detector="lof"):
    datasets = []
    suffix = detector_suffix(outlier_detector)
    for path in RESULTS_DIR.glob(f"*_unc_summary{suffix}"):
        if path.is_dir():
            datasets.append(path.name.removesuffix(f"_unc_summary{suffix}"))
    ordered = [dataset for dataset in DEFAULT_DATASETS if dataset in datasets]
    ordered.extend(sorted(set(datasets) - set(ordered)))
    return ordered


def observation_file(dataset, kind, outlier_detector="lof"):
    suffix = detector_suffix(outlier_detector)
    return (
        RESULTS_DIR
        / f"{dataset}_unc_by_observation{suffix}"
        / f"{dataset}_epis_unc_{kind}_obs_mean.csv"
    )


def datasets_with_disentanglement_results(datasets, outlier_detector="lof"):
    valid_datasets = []
    missing_datasets = []
    for dataset in datasets:
        required_files = [
            observation_file(dataset, "inlier", outlier_detector),
            observation_file(dataset, "outlier", outlier_detector),
        ]
        if all(path.exists() for path in required_files):
            valid_datasets.append(dataset)
        else:
            missing_datasets.append(dataset)
    return valid_datasets, missing_datasets


def read_metrics_files(datasets, outlier_detector="lof"):
    data_dict_boxplot = {}
    for dataset in datasets:
        inlier_obs = pd.read_csv(observation_file(dataset, "inlier", outlier_detector)).iloc[:, 0].values
        outlier_obs = pd.read_csv(observation_file(dataset, "outlier", outlier_detector)).iloc[:, 0].values

        data_dict_boxplot[dataset] = pd.DataFrame(
            {
                "epistemic_uncertainty": np.concatenate([inlier_obs, outlier_obs]),
                "type": np.repeat(["inlier", "outlier"], [len(inlier_obs), len(outlier_obs)]),
            }
        )

    return data_dict_boxplot


def subplot_grid(n_plots, max_cols=4):
    n_cols = min(max_cols, n_plots)
    n_rows = ceil(n_plots / n_cols)
    return n_rows, n_cols


def prepare_axes(n_plots, figsize_per_panel=(4.5, 3.8), max_cols=4):
    n_rows, n_cols = subplot_grid(n_plots, max_cols=max_cols)
    figsize = (figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    return fig, axes.flatten(), n_rows, n_cols


def display_dataset_name(name):
    return name.replace("_", " ")


def style_plot_fonts():
    plt.rcParams.update(
        {
            "font.size": 19,
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 19,
        }
    )


def plot_boxplots(data_boxplot, output_path=None, show=True, max_cols=4):
    fig, axes, _, n_cols = prepare_axes(
        len(data_boxplot),
        figsize_per_panel=(5.65, 4.45),
        max_cols=max_cols,
    )
    colors = ["C0", "C1"]
    show_ytick_idxs = {idx for idx in range(0, len(data_boxplot), n_cols)}

    for i, (name, df) in enumerate(data_boxplot.items()):
        ax = axes[i]
        inlier = df.loc[df["type"] == "inlier", "epistemic_uncertainty"].astype(float).values
        outlier = df.loc[df["type"] == "outlier", "epistemic_uncertainty"].astype(float).values
        data = [
            inlier if inlier.size > 0 else np.array([np.nan]),
            outlier if outlier.size > 0 else np.array([np.nan]),
        ]

        boxplot = ax.boxplot(
            data,
            vert=False,
            tick_labels=["Inlier", "Outlier"],
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_linewidth(1.2)
        for median in boxplot["medians"]:
            median.set_color("k")
            median.set_linewidth(1.6)

        ax.set_ylim(0.5, 2.5)
        ax.set_title(display_dataset_name(name), pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        if i in show_ytick_idxs:
            ax.set_yticks([1, 2])
            ax.set_yticklabels(["Inlier", "Outlier"], fontsize=18)
        else:
            ax.set_yticks([])

    for ax in axes[len(data_boxplot) :]:
        ax.axis("off")

    fig.text(0.5, 0.035, "Epistemic uncertainty", ha="center", va="center", fontsize=22)
    fig.tight_layout(rect=[0.02, 0.065, 1.0, 1.0])
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to plot. Defaults to the QNN table dataset order.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all datasets with disentanglement result files.",
    )
    parser.add_argument("--outlier_detector", choices=["lof", "isolation_forest"], default="lof")
    parser.add_argument("--max_cols", type=int, default=5)
    parser.add_argument("--save_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    style_plot_fonts()

    if args.all:
        datasets = available_disentanglement_datasets(args.outlier_detector)
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = DEFAULT_DATASETS

    datasets, missing_datasets = datasets_with_disentanglement_results(datasets, args.outlier_detector)
    if missing_datasets:
        print(
            "Skipping datasets without complete disentanglement results: "
            + ", ".join(missing_datasets)
        )
    if not datasets:
        raise FileNotFoundError("No complete disentanglement results found.")

    print("Plotting disentanglement results for: " + ", ".join(datasets))
    data_boxplot = read_metrics_files(
        datasets,
        outlier_detector=args.outlier_detector,
    )

    boxplot_path = None
    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        suffix = detector_suffix(args.outlier_detector)
        boxplot_path = args.save_dir / f"disentanglement_boxplots{suffix}.png"

    plot_boxplots(
        data_boxplot,
        output_path=boxplot_path,
        show=not args.no_show,
        max_cols=args.max_cols,
    )


if __name__ == "__main__":
    main()
