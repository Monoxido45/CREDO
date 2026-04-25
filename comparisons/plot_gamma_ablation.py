import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results"
PLOTS_PATH = RESULTS_PATH / "gamma_ablation_plots"

DEFAULT_DATASETS = ["airfoil", "concrete", "winered", "winewhite"]
METRICS = [
    ("smis", "SMIS"),
    ("outlier_coverage", "Outlier coverage"),
    ("outlier_inlier_ratio", "Outlier/inlier ratio"),
]


def read_summary(dataset, study):
    path = (
        RESULTS_PATH
        / f"{dataset}_ablation_gamma_{study}_summary"
        / f"{dataset}_ablation_gamma_{study}_summary.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)


def metric_interval(row, metric):
    mean = row[f"{metric}_mean"]
    sd = row.get(f"{metric}_sd", np.nan)
    n_rep = row.get("n_rep_completed", np.nan)
    if pd.isna(sd) or pd.isna(n_rep) or n_rep <= 0:
        return mean, mean
    half_width = 2 * sd / np.sqrt(n_rep)
    return mean - half_width, mean + half_width


def finish_figure(fig, output_name):
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_PATH / f"{output_name}.png"
    pdf_path = PLOTS_PATH / f"{output_name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_fixed(datasets):
    summaries = {dataset: read_summary(dataset, "fixed") for dataset in datasets}
    summaries = {dataset: df for dataset, df in summaries.items() if df is not None}
    if not summaries:
        raise FileNotFoundError("No fixed-gamma ablation summaries found.")

    fig, axes = plt.subplots(
        len(METRICS),
        len(summaries),
        figsize=(4.2 * len(summaries), 8.5),
        sharex=False,
    )
    if len(summaries) == 1:
        axes = np.asarray(axes).reshape(len(METRICS), 1)

    for col, (dataset, df) in enumerate(summaries.items()):
        df = df.sort_values("gamma")
        x = df["gamma"].to_numpy()
        for row_idx, (metric, label) in enumerate(METRICS):
            ax = axes[row_idx, col]
            y = df[f"{metric}_mean"].to_numpy()
            lows, highs = [], []
            for _, row in df.iterrows():
                low, high = metric_interval(row, metric)
                lows.append(low)
                highs.append(high)
            ax.plot(x, y, marker="o", linewidth=2, color="C0")
            ax.fill_between(x, lows, highs, color="C0", alpha=0.15, linewidth=0)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(dataset)
            if col == 0:
                ax.set_ylabel(label)
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel(r"Fixed $\gamma$")

    finish_figure(fig, "gamma_fixed_ablation_curves")


def plot_adaptive(datasets):
    summaries = {dataset: read_summary(dataset, "adaptive") for dataset in datasets}
    summaries = {dataset: df for dataset, df in summaries.items() if df is not None}
    if not summaries:
        raise FileNotFoundError("No adaptive-gamma ablation summaries found.")

    fig, axes = plt.subplots(
        len(METRICS),
        len(summaries),
        figsize=(4.6 * len(summaries), 8.5),
        sharex=False,
    )
    if len(summaries) == 1:
        axes = np.asarray(axes).reshape(len(METRICS), 1)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col, (dataset, df) in enumerate(summaries.items()):
        df = df.sort_values(["gamma_min", "gamma_max", "tau_gamma"])
        configs = list(df[["gamma_min", "gamma_max"]].drop_duplicates().itertuples(index=False, name=None))
        for row_idx, (metric, label) in enumerate(METRICS):
            ax = axes[row_idx, col]
            for config_idx, (gamma_min, gamma_max) in enumerate(configs):
                subset = df[
                    (df["gamma_min"] == gamma_min)
                    & (df["gamma_max"] == gamma_max)
                ].sort_values("tau_gamma")
                x = subset["tau_gamma"].to_numpy()
                y = subset[f"{metric}_mean"].to_numpy()
                color = colors[config_idx % len(colors)]
                linestyle = "-" if gamma_min == min(df["gamma_min"]) else "--"
                label_text = rf"$\gamma_{{min}}={gamma_min}$, $\gamma_{{max}}={gamma_max}$"
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    linestyle=linestyle,
                    color=color,
                    label=label_text,
                )
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(dataset)
            if col == 0:
                ax.set_ylabel(label)
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel(r"$\tau_\gamma$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    finish_figure(fig, "gamma_adaptive_ablation_curves")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot gamma ablation summaries.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--study", choices=["fixed", "adaptive", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.study in ["fixed", "both"]:
        plot_fixed(args.datasets)
    if args.study in ["adaptive", "both"]:
        plot_adaptive(args.datasets)


if __name__ == "__main__":
    main()
