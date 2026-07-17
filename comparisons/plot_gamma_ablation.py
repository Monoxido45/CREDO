import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results"
PLOTS_PATH = ROOT / "paper_results" / "figures"

DEFAULT_DATASETS = ["airfoil", "concrete", "winered", "winewhite", "meps19"]
DEFAULT_FIXED_GAMMA = 0.2
DEFAULT_ADAPTIVE_GAMMA_MIN = 0.05
DEFAULT_ADAPTIVE_GAMMA_MAX = 0.9
DEFAULT_ADAPTIVE_TAU = 1.0
METRICS = [
    ("smis", "SMIS"),
    ("outlier_coverage", "Outlier coverage"),
    ("outlier_inlier_ratio", "Outlier/inlier ratio"),
]


def format_gamma(value):
    return f"{value:g}"


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


def finish_figure(fig, output_name, tight_layout=True):
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_PATH / f"{output_name}.png"
    pdf_path = PLOTS_PATH / f"{output_name}.pdf"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def dataset_suffix(datasets):
    if datasets == DEFAULT_DATASETS:
        return "main"
    if len(datasets) == 1:
        return datasets[0]
    return "_".join(datasets)


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
        x_values = df["gamma"].to_numpy()
        x_positions = np.arange(len(x_values))
        for row_idx, (metric, label) in enumerate(METRICS):
            ax = axes[row_idx, col]
            y = df[f"{metric}_mean"].to_numpy()
            lows, highs = [], []
            for _, row in df.iterrows():
                low, high = metric_interval(row, metric)
                lows.append(low)
                highs.append(high)
            ax.plot(x_positions, y, marker="o", linewidth=2, color="C0")
            ax.fill_between(x_positions, lows, highs, color="C0", alpha=0.15, linewidth=0)
            default_positions = np.where(np.isclose(x_values, DEFAULT_FIXED_GAMMA))[0]
            if len(default_positions) > 0:
                default_position = default_positions[0]
                ax.axvline(
                    default_position,
                    color="black",
                    linestyle=":",
                    linewidth=1.6,
                    alpha=0.9,
                    zorder=1,
                )
                default_row = df[np.isclose(df["gamma"], DEFAULT_FIXED_GAMMA)]
                if not default_row.empty:
                    ax.scatter(
                        [default_position],
                        [default_row.iloc[0][f"{metric}_mean"]],
                        color="black",
                        edgecolor="white",
                        linewidth=0.8,
                        s=64,
                        zorder=5,
                    )
            ax.set_xticks(x_positions)
            ax.set_xticklabels([format_gamma(value) for value in x_values], rotation=30, ha="right")
            ax.set_xlim(-0.35, len(x_positions) - 0.65)
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(dataset)
            if col == 0:
                ax.set_ylabel(label)
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel(r"Fixed $\gamma$")

    handles = [
        Line2D([0], [0], color="C0", linewidth=2, marker="o"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=1.6,
            marker="o",
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.8,
        ),
    ]
    labels = [
        r"tested fixed $\gamma$",
        rf"default fixed $\gamma={format_gamma(DEFAULT_FIXED_GAMMA)}$",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    finish_figure(fig, f"gamma_fixed_ablation_curves_{dataset_suffix(list(summaries.keys()))}")


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
    all_gamma_min = np.array(
        sorted({value for df in summaries.values() for value in df["gamma_min"].dropna().unique()})
    )
    all_gamma_max = np.array(
        sorted({value for df in summaries.values() for value in df["gamma_max"].dropna().unique()})
    )
    line_styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "P", "X"]
    gamma_min_style = {
        value: line_styles[idx % len(line_styles)]
        for idx, value in enumerate(all_gamma_min)
    }
    gamma_max_marker = {
        value: markers[idx % len(markers)]
        for idx, value in enumerate(all_gamma_max)
    }
    gamma_max_color = {
        value: colors[idx % len(colors)]
        for idx, value in enumerate(all_gamma_max)
    }

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
                is_default_pair = (
                    np.isclose(gamma_min, DEFAULT_ADAPTIVE_GAMMA_MIN)
                    and np.isclose(gamma_max, DEFAULT_ADAPTIVE_GAMMA_MAX)
                )
                color = gamma_max_color[gamma_max]
                linestyle = gamma_min_style[gamma_min]
                marker = gamma_max_marker[gamma_max]
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linewidth=2.8 if is_default_pair else 1.5,
                    linestyle=linestyle,
                    color="black" if is_default_pair else color,
                    alpha=1.0 if is_default_pair else 0.65,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    zorder=4 if is_default_pair else 2,
                )
                if is_default_pair and x.min() <= DEFAULT_ADAPTIVE_TAU <= x.max():
                    default_row = subset[np.isclose(subset["tau_gamma"], DEFAULT_ADAPTIVE_TAU)]
                    if not default_row.empty:
                        ax.scatter(
                            [DEFAULT_ADAPTIVE_TAU],
                            [default_row.iloc[0][f"{metric}_mean"]],
                            color="black",
                            edgecolor="white",
                            linewidth=0.8,
                            s=70,
                            zorder=6,
                        )
            if df["tau_gamma"].min() <= DEFAULT_ADAPTIVE_TAU <= df["tau_gamma"].max():
                ax.axvline(
                    DEFAULT_ADAPTIVE_TAU,
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.65,
                    zorder=1,
                )
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(dataset)
            if col == 0:
                ax.set_ylabel(label)
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel(r"$\tau_\gamma$")

    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.8,
            linestyle=gamma_min_style.get(DEFAULT_ADAPTIVE_GAMMA_MIN, "-"),
            marker=gamma_max_marker.get(DEFAULT_ADAPTIVE_GAMMA_MAX, "o"),
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.7,
        ),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.2),
    ]
    labels = [
        rf"default: $\gamma_{{min}}={format_gamma(DEFAULT_ADAPTIVE_GAMMA_MIN)}$, "
        rf"$\gamma_{{max}}={format_gamma(DEFAULT_ADAPTIVE_GAMMA_MAX)}$",
        rf"default: $\tau_\gamma={format_gamma(DEFAULT_ADAPTIVE_TAU)}$",
    ]

    for value in all_gamma_min:
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.25",
                linewidth=1.8,
                linestyle=gamma_min_style[value],
            )
        )
        labels.append(rf"$\gamma_{{min}}={format_gamma(value)}$")

    for value in all_gamma_max:
        handles.append(
            Line2D(
                [0],
                [0],
                color=gamma_max_color[value],
                linewidth=0,
                marker=gamma_max_marker[value],
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
        )
        labels.append(rf"$\gamma_{{max}}={format_gamma(value)}$")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(5, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    finish_figure(fig, f"gamma_adaptive_ablation_curves_{dataset_suffix(list(summaries.keys()))}")


def draw_default_cell(ax, x_values, y_values, tau):
    if not np.isclose(tau, DEFAULT_ADAPTIVE_TAU):
        return
    x_match = np.where(np.isclose(x_values, DEFAULT_ADAPTIVE_GAMMA_MAX))[0]
    y_match = np.where(np.isclose(y_values, DEFAULT_ADAPTIVE_GAMMA_MIN))[0]
    if len(x_match) == 0 or len(y_match) == 0:
        return
    ax.add_patch(
        Rectangle(
            (x_match[0] - 0.5, y_match[0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.2,
            zorder=5,
        )
    )


def annotate_heatmap(ax, matrix, vmin, vmax):
    span = vmax - vmin
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            normalized = 0.5 if span == 0 else (value - vmin) / span
            color = "white" if normalized < 0.45 else "black"
            ax.text(
                j,
                i,
                f"{value:.3g}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )


def plot_adaptive_heatmaps(datasets):
    summaries = {dataset: read_summary(dataset, "adaptive") for dataset in datasets}
    summaries = {dataset: df for dataset, df in summaries.items() if df is not None}
    if not summaries:
        raise FileNotFoundError("No adaptive-gamma ablation summaries found.")

    tau_values = sorted(
        {
            tau
            for df in summaries.values()
            for tau in df["tau_gamma"].dropna().unique()
        }
    )

    for metric, label in METRICS:
        fig, axes = plt.subplots(
            len(summaries),
            len(tau_values),
            figsize=(3.2 * len(tau_values), 2.7 * len(summaries)),
            sharex=False,
            sharey=False,
            layout="constrained",
        )
        if len(summaries) == 1 and len(tau_values) == 1:
            axes = np.asarray([[axes]])
        elif len(summaries) == 1:
            axes = np.asarray(axes).reshape(1, len(tau_values))
        elif len(tau_values) == 1:
            axes = np.asarray(axes).reshape(len(summaries), 1)

        for row_idx, (dataset, df) in enumerate(summaries.items()):
            x_values = np.array(sorted(df["gamma_max"].dropna().unique()))
            y_values = np.array(sorted(df["gamma_min"].dropna().unique()))
            row_values = df[f"{metric}_mean"].dropna().to_numpy()
            vmin = np.min(row_values)
            vmax = np.max(row_values)
            row_image = None
            for col_idx, tau in enumerate(tau_values):
                ax = axes[row_idx, col_idx]
                subset = df[np.isclose(df["tau_gamma"], tau)]
                pivot = subset.pivot_table(
                    index="gamma_min",
                    columns="gamma_max",
                    values=f"{metric}_mean",
                    aggfunc="mean",
                )
                matrix = pivot.reindex(index=y_values, columns=x_values).to_numpy()
                masked = np.ma.masked_invalid(matrix)
                row_image = ax.imshow(
                    masked,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
                annotate_heatmap(ax, matrix, vmin, vmax)
                draw_default_cell(ax, x_values, y_values, tau)
                ax.set_xticks(np.arange(len(x_values)))
                ax.set_xticklabels([format_gamma(value) for value in x_values])
                ax.set_yticks(np.arange(len(y_values)))
                ax.set_yticklabels([format_gamma(value) for value in y_values])
                if row_idx == 0:
                    ax.set_title(rf"$\tau_\gamma={format_gamma(tau)}$")
                if col_idx == 0:
                    ax.set_ylabel(f"{dataset}\n" + r"$\gamma_{min}$")
                if row_idx == len(summaries) - 1:
                    ax.set_xlabel(r"$\gamma_{max}$")

            if row_image is not None:
                cbar = fig.colorbar(
                    row_image,
                    ax=axes[row_idx, :].ravel().tolist(),
                    shrink=0.85,
                    pad=0.02,
                )
                cbar.set_label(label)

        fig.suptitle(
            rf"Default outlined: $\gamma_{{min}}={format_gamma(DEFAULT_ADAPTIVE_GAMMA_MIN)}$, "
            rf"$\gamma_{{max}}={format_gamma(DEFAULT_ADAPTIVE_GAMMA_MAX)}$, "
            rf"$\tau_\gamma={format_gamma(DEFAULT_ADAPTIVE_TAU)}$",
            y=1.03,
        )
        finish_figure(
            fig,
            f"gamma_adaptive_heatmap_{metric}_{dataset_suffix(list(summaries.keys()))}",
            tight_layout=False,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot gamma ablation summaries.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--study", choices=["fixed", "adaptive", "both"], default="both")
    parser.add_argument("--heatmaps", action="store_true", help="Also generate adaptive-gamma heatmaps.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.study in ["fixed", "both"]:
        plot_fixed(args.datasets)
    if args.study in ["adaptive", "both"]:
        plot_adaptive(args.datasets)
        if args.heatmaps:
            plot_adaptive_heatmaps(args.datasets)


if __name__ == "__main__":
    main()
