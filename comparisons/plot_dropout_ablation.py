"""Plot fixed and adaptive CREDO dropout sensitivity curves."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results"
PLOTS_PATH = RESULTS_PATH / "dropout_ablation_figures"
DEFAULT_DATASETS = ["airfoil", "concrete", "winered", "winewhite", "meps19"]
DEFAULT_DROPOUT = 0.1
METRICS = [
    ("smis", "SMIS"),
    ("outlier_coverage", "Outlier coverage"),
    ("outlier_inlier_ratio", "Outlier/inlier ratio"),
]
TITLE_FONTSIZE = 17
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 13


def format_value(value):
    return f"{value:g}"


def read_summary(dataset, study):
    path = (
        RESULTS_PATH
        / f"{dataset}_ablation_dropout_{study}_summary"
        / f"{dataset}_ablation_dropout_{study}_summary.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)


def interval(row, metric):
    mean = row[f"{metric}_mean"]
    sd = row.get(f"{metric}_sd", np.nan)
    n = row.get("n_rep_completed", np.nan)
    if pd.isna(sd) or pd.isna(n) or n <= 0:
        return mean, mean
    half_width = 2 * sd / np.sqrt(n)
    return mean - half_width, mean + half_width


def save_figure(fig, detector, study_selection):
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    suffix = "" if study_selection == "both" else f"_{study_selection}"
    stem = PLOTS_PATH / f"dropout_ablation_curves_{detector}{suffix}"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"Saved {stem}.png")
    print(f"Saved {stem}.pdf")


def plot_detector(datasets, detector, study_selection):
    studies_to_plot = ("fixed", "adaptive") if study_selection == "both" else (study_selection,)
    summaries = {
        dataset: {
            study: read_summary(dataset, study) for study in studies_to_plot
        }
        for dataset in datasets
    }
    summaries = {
        dataset: studies
        for dataset, studies in summaries.items()
        if any(frame is not None for frame in studies.values())
    }
    if not summaries:
        raise FileNotFoundError(f"No dropout-ablation summaries found for {detector}.")

    fig, axes = plt.subplots(
        len(studies_to_plot) * len(METRICS),
        len(summaries),
        figsize=(4.6 * len(summaries), 14.0),
        squeeze=False,
        sharex=False,
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 5))
    markers = ["o", "s", "^", "D", "P"]

    for col, (dataset, studies) in enumerate(summaries.items()):
        for study_idx, study in enumerate(studies_to_plot):
            frame = studies[study]
            if frame is None:
                continue
            frame = frame[frame["detector"] == detector].sort_values("dropout")
            if frame.empty:
                continue
            x_values = frame["dropout"].to_numpy()
            x_positions = np.arange(len(x_values))
            for metric_idx, (metric, label) in enumerate(METRICS):
                row_idx = study_idx * len(METRICS) + metric_idx
                ax = axes[row_idx, col]
                y = frame[f"{metric}_mean"].to_numpy()
                lows, highs = zip(*(interval(row, metric) for _, row in frame.iterrows()))
                ax.plot(
                    x_positions,
                    y,
                    color="#277da1",
                    linewidth=2.0,
                    marker="o",
                    markersize=5.5,
                    alpha=0.9,
                )
                ax.fill_between(
                    x_positions,
                    lows,
                    highs,
                    color="#277da1",
                    alpha=0.16,
                    linewidth=0,
                )
                default = np.where(np.isclose(x_values, DEFAULT_DROPOUT))[0]
                if len(default):
                    pos = default[0]
                    ax.axvline(pos, color="black", linestyle=":", linewidth=1.4, zorder=1)
                    ax.scatter(
                        [pos],
                        [y[pos]],
                        color="black",
                        edgecolor="white",
                        linewidth=0.7,
                        s=58,
                        zorder=5,
                    )
                ax.set_xticks(x_positions)
                ax.set_xticklabels([format_value(value) for value in x_values], rotation=30, ha="right")
                ax.set_xlim(-0.35, len(x_positions) - 0.65)
                ax.grid(True, alpha=0.25)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
                if metric_idx == 0:
                    ax.set_title(dataset, fontsize=TITLE_FONTSIZE)
                if col == 0:
                    ax.set_ylabel(label, fontsize=LABEL_FONTSIZE)
                if metric_idx == len(METRICS) - 1:
                    ax.set_xlabel("MC-dropout rate", fontsize=LABEL_FONTSIZE)
                if col == 0 and metric_idx == 0:
                    ax.text(
                        -0.42,
                        1.18,
                        "Fixed CREDO" if study == "fixed" else "Adaptive CREDO",
                        transform=ax.transAxes,
                        fontsize=LABEL_FONTSIZE + 1,
                        fontweight="bold",
                        va="bottom",
                    )

    handles = [
        Line2D([0], [0], color="#277da1", linewidth=2, marker="o"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=1.4,
            marker="o",
            markerfacecolor="black",
            markeredgecolor="white",
        ),
    ]
    labels = [
        "mean with 95% CI",
        rf"default dropout $={format_value(DEFAULT_DROPOUT)}$",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.998),
        fontsize=LEGEND_FONTSIZE,
    )
    fig.suptitle(
        f"CREDO dropout sensitivity ({detector.replace('_', ' ').title()})",
        fontsize=TITLE_FONTSIZE + 2,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save_figure(fig, detector, study_selection)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CREDO dropout sensitivity curves.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--detectors", nargs="+", choices=["lof", "isolation_forest"], default=["lof", "isolation_forest"])
    parser.add_argument("--study", choices=["fixed", "adaptive", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    for detector in args.detectors:
        plot_detector(args.datasets, detector, args.study)


if __name__ == "__main__":
    main()
