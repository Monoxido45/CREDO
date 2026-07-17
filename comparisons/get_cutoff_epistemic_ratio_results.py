from argparse import ArgumentParser
from math import ceil
import os
import pickle
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
METHOD_LABELS = {
    "credo_QNN": "CREDO fixed",
    "credo_QNN_adaptive": "CREDO adaptive",
}
METHOD_ORDER = list(METHOD_LABELS.keys())
METHOD_COLORS = ["#009E73", "#6A3D9A"]
RATIO_LABEL = "Cutoff / epi. unc."


def summary_file(dataset):
    return (
        RESULTS_DIR
        / f"{dataset}_cutoff_epistemic_ratio_summary"
        / f"{dataset}_cutoff_epistemic_ratio_summary.csv"
    )


def observation_file(dataset, method):
    return (
        RESULTS_DIR
        / f"{dataset}_cutoff_epistemic_ratio_by_observation"
        / f"{dataset}_{method}_cutoff_epistemic_ratio_obs_mean.csv"
    )


def raw_file(dataset):
    return RESULTS_DIR / "raw" / dataset / f"{dataset}_cutoff_epistemic_ratio_raw.pkl"


def align_vectors(vectors):
    arrays = [np.asarray(vector, dtype=float).ravel() for vector in vectors]
    if not arrays:
        return np.empty((0, 0))
    min_len = min(array.shape[0] for array in arrays)
    return np.vstack([array[:min_len] for array in arrays])


def read_raw_absolute_results(dataset):
    path = raw_file(dataset)
    if not path.exists():
        return None
    with path.open("rb") as f:
        raw = pickle.load(f)

    results = {}
    for method in METHOD_ORDER:
        method_data = raw.get("results", {}).get(method)
        if not method_data:
            continue
        ratio_vectors = method_data.get("ratio_vectors", [])
        ratio_matrix = align_vectors(ratio_vectors)
        if ratio_matrix.size == 0:
            continue
        abs_matrix = np.abs(ratio_matrix)
        rep_means = np.nanmean(abs_matrix, axis=1)
        n_rep = rep_means.shape[0]
        sd = np.nanstd(rep_means, ddof=1) if n_rep > 1 else 0.0
        results[method] = {
            "obs_mean": np.nanmean(abs_matrix, axis=0),
            "mean": float(np.nanmean(rep_means)),
            "se2": float(2 * sd / np.sqrt(n_rep)) if n_rep > 0 else np.nan,
        }
    return results


def available_datasets():
    datasets = []
    for path in RESULTS_DIR.glob("*_cutoff_epistemic_ratio_summary"):
        if path.is_dir():
            datasets.append(path.name.removesuffix("_cutoff_epistemic_ratio_summary"))
    ordered = [dataset for dataset in DEFAULT_DATASETS if dataset in datasets]
    ordered.extend(sorted(set(datasets) - set(ordered)))
    return ordered


def datasets_with_results(datasets):
    valid, missing = [], []
    for dataset in datasets:
        required = [summary_file(dataset)]
        required.extend(observation_file(dataset, method) for method in METHOD_ORDER)
        if all(path.exists() for path in required):
            valid.append(dataset)
        else:
            missing.append(dataset)
    return valid, missing


def read_metrics_files(datasets, bar_stat="median"):
    boxplot_data = {}
    barplot_data = {}

    for dataset in datasets:
        raw_abs = read_raw_absolute_results(dataset)
        summary = pd.read_csv(summary_file(dataset))
        summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
        summary["method"] = pd.Categorical(
            summary["method"],
            categories=METHOD_ORDER,
            ordered=True,
        )
        summary = summary.sort_values("method")

        box_rows = []
        for method in METHOD_ORDER:
            if raw_abs and method in raw_abs:
                obs = raw_abs[method]["obs_mean"]
            else:
                obs = np.abs(
                    pd.read_csv(observation_file(dataset, method))
                    .iloc[:, 0]
                    .astype(float)
                    .values
                )
            box_rows.append(
                pd.DataFrame(
                    {
                        "method": METHOD_LABELS[method],
                        "cutoff_epistemic_ratio": obs,
                    }
                )
            )
        boxplot_data[dataset] = pd.concat(box_rows, ignore_index=True)

        centers = []
        err_low = []
        err_high = []
        for method in summary["method"].astype(str):
            label = METHOD_LABELS[method]
            obs_values = boxplot_data[dataset].loc[
                boxplot_data[dataset]["method"] == label,
                "cutoff_epistemic_ratio",
            ].astype(float).to_numpy()
            if bar_stat == "median":
                center = float(np.nanmedian(obs_values))
                q25, q75 = np.nanpercentile(obs_values, [25, 75])
                centers.append(center)
                err_low.append(center - float(q25))
                err_high.append(float(q75) - center)
            elif raw_abs and method in raw_abs:
                centers.append(raw_abs[method]["mean"])
                err_low.append(raw_abs[method]["se2"])
                err_high.append(raw_abs[method]["se2"])
            else:
                method_row = summary[summary["method"].astype(str) == method].iloc[0]
                centers.append(abs(float(method_row["mean"])))
                err_low.append(float(method_row["se2"]))
                err_high.append(float(method_row["se2"]))

        barplot_data[dataset] = pd.DataFrame(
            {
                "method": [METHOD_LABELS[method] for method in summary["method"].astype(str)],
                "center": centers,
                "err_low": err_low,
                "err_high": err_high,
                "cutoff_mean": summary["cutoff_mean"].astype(float).to_numpy(),
                "epistemic_uncertainty_mean": summary["epistemic_uncertainty_mean"].astype(float).to_numpy(),
            }
        )

    return boxplot_data, barplot_data


def subplot_grid(n_plots, max_cols=4):
    n_cols = min(max_cols, n_plots)
    n_rows = ceil(n_plots / n_cols)
    return n_rows, n_cols


def prepare_axes(n_plots, figsize_per_panel=(4.6, 3.5), max_cols=4, sharex=False, sharey=False):
    n_rows, n_cols = subplot_grid(n_plots, max_cols=max_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return fig, axes.flatten(), n_rows, n_cols


def style_plot_fonts():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
        }
    )


def plot_barplots(data_barplot, output_path=None, show=True, max_cols=4):
    fig, axes, _, _ = prepare_axes(
        len(data_barplot),
        figsize_per_panel=(4.5, 3.3),
        max_cols=max_cols,
    )
    colors = METHOD_COLORS

    for idx, (dataset, df) in enumerate(data_barplot.items()):
        ax = axes[idx]
        positions = np.arange(len(df))
        centers = df["center"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                df["err_low"].to_numpy(dtype=float),
                df["err_high"].to_numpy(dtype=float),
            ]
        )

        ax.bar(positions, centers, color=colors[: len(df)], width=0.45, alpha=0.8)
        ax.errorbar(positions, centers, yerr=yerr, fmt="none", ecolor="k", capsize=5)
        ax.set_xticks(positions)
        ax.set_xticklabels(df["method"], rotation=20, ha="right")
        local_high = float((df["center"].astype(float) + df["err_high"].astype(float)).max())
        ax.set_ylim(0, local_high * 1.12 if local_high > 0 else 1.0)
        ax.set_title(dataset)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(data_barplot) :]:
        ax.axis("off")

    fig.supylabel(RATIO_LABEL, x=0.025)
    fig.tight_layout(rect=[0.07, 0.04, 1.0, 1.0])
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_boxplots(data_boxplot, output_path=None, show=True, max_cols=4):
    fig, axes, _, n_cols = prepare_axes(
        len(data_boxplot),
        figsize_per_panel=(4.5, 3.4),
        max_cols=max_cols,
    )
    colors = METHOD_COLORS
    show_ytick_idxs = {idx for idx in range(0, len(data_boxplot), n_cols)}

    for idx, (dataset, df) in enumerate(data_boxplot.items()):
        ax = axes[idx]
        data = []
        labels = []
        for method in METHOD_ORDER:
            label = METHOD_LABELS[method]
            values = df.loc[
                df["method"] == label,
                "cutoff_epistemic_ratio",
            ].astype(float).values
            data.append(values if values.size else np.array([np.nan]))
            labels.append(label)

        boxplot = ax.boxplot(
            data,
            tick_labels=labels,
            vert=False,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_linewidth(1.2)
        for median in boxplot["medians"]:
            median.set_color("k")
            median.set_linewidth(1.6)

        ax.set_title(dataset)
        if idx not in show_ytick_idxs:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    for ax in axes[len(data_boxplot) :]:
        ax.axis("off")

    fig.supxlabel(RATIO_LABEL, y=0.02)
    fig.supylabel("Method", x=0.025)
    fig.tight_layout(rect=[0.07, 0.06, 1.0, 1.0])
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max_cols", type=int, default=4)
    parser.add_argument("--save_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--bar_stat", choices=["median", "mean"], default="median")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    style_plot_fonts()

    if args.all:
        datasets = available_datasets()
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = DEFAULT_DATASETS

    datasets, missing = datasets_with_results(datasets)
    if missing:
        print(
            "Skipping datasets without complete cutoff/epistemic ratio results: "
            + ", ".join(missing)
        )
    if not datasets:
        raise FileNotFoundError("No complete cutoff/epistemic ratio results found.")

    print("Plotting cutoff/epistemic ratio results for: " + ", ".join(datasets))
    boxplot_data, barplot_data = read_metrics_files(datasets, bar_stat=args.bar_stat)

    barplot_path = None
    boxplot_path = None
    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        barplot_path = args.save_dir / "cutoff_epistemic_ratio_barplots.png"
        boxplot_path = args.save_dir / "cutoff_epistemic_ratio_boxplots.png"

    plot_barplots(
        barplot_data,
        output_path=barplot_path,
        show=not args.no_show,
        max_cols=args.max_cols,
    )
    plot_boxplots(
        boxplot_data,
        output_path=boxplot_path,
        show=not args.no_show,
        max_cols=args.max_cols,
    )


if __name__ == "__main__":
    main()
