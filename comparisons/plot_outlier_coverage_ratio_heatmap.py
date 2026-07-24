"""Heatmaps for outlier coverage and outlier/inlier interval-length ratio.

The coverage panel highlights methods whose 95% CI reaches the target
coverage, i.e. upper CI endpoint >= target. If every method is incompatible
with the target for a dataset, it highlights the methods whose 95% CIs
intersect the closest-to-target method.

The ratio panel highlights methods with competitive outlier/inlier ratios,
independently of the coverage panel.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "paper_results" / "figures"

DATASETS = [
    "qsar_fish_toxicity",
    "concrete",
    "airfoil",
    "winered",
    "communities",
    "star",
    "abalone",
    "winewhite",
    "cycle",
    "electric",
    "meps19",
    "superconductivity",
    "homes",
    "protein",
    "WEC",
]

METHODS = [
    ("credo_QNN", "CREDO"),
    ("credo_QNN_adaptive", "CREDO adap."),
    ("cqr", "CQR"),
    ("cqrr", "CQR-r"),
    ("uacqrs", "UACQRS"),
    ("uacqrp", "UACQRP"),
    ("EPIC", "EPIC"),
]
EMPHASIZED_METHODS = {"CREDO", "CREDO adap."}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 13.2,
            "ytick.labelsize": 15,
            "legend.fontsize": 12.5,
        }
    )


def emphasize_method_tick_labels(ax: plt.Axes) -> None:
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        if tick_label.get_text() in EMPHASIZED_METHODS:
            tick_label.set_fontweight("bold")


def detector_suffix(outlier_detector: str) -> str:
    return "" if outlier_detector == "lof" else f"_{outlier_detector}"


def detector_label(outlier_detector: str) -> str:
    if outlier_detector == "lof":
        return "LOF"
    if outlier_detector == "isolation_forest":
        return "Isolation Forest"
    return outlier_detector.replace("_", " ").title()


def metric_path(dataset: str, model: str, metric: str, outlier_detector: str) -> Path:
    suffix = detector_suffix(outlier_detector)
    return (
        RESULTS_DIR
        / f"{dataset}_{model}_summary"
        / f"{dataset}_{metric}_outlier{suffix}_summary.csv"
    )


def interval_intersects(
    first_low: float,
    first_high: float,
    second_low: float,
    second_high: float,
    tol: float = 1e-12,
) -> bool:
    return first_low <= second_high + tol and second_low <= first_high + tol


def interval_distance_to_target(low: float, high: float, target: float) -> float:
    if low <= target <= high:
        return 0.0
    return min(abs(low - target), abs(high - target))


def read_dataset(dataset: str, model: str, outlier_detector: str, n_rep: int) -> pd.DataFrame | None:
    coverage_path = metric_path(dataset, model, "coverage", outlier_detector)
    ratio_path = metric_path(dataset, model, "ratio", outlier_detector)
    if not coverage_path.exists() or not ratio_path.exists():
        return None

    coverage = pd.read_csv(coverage_path)
    ratio = pd.read_csv(ratio_path)
    rows = []
    for method, label in METHODS:
        coverage_row = coverage[coverage["methods"].eq(method)]
        ratio_row = ratio[ratio["methods"].eq(method)]
        if coverage_row.empty or ratio_row.empty:
            continue
        coverage_mean = float(coverage_row.iloc[0]["mean"])
        coverage_hw = 2 * float(coverage_row.iloc[0]["sd"]) / math.sqrt(n_rep)
        ratio_mean = float(ratio_row.iloc[0]["mean"])
        ratio_hw = 2 * float(ratio_row.iloc[0]["sd"]) / math.sqrt(n_rep)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "label": label,
                "coverage_mean": coverage_mean,
                "coverage_hw": coverage_hw,
                "coverage_low": coverage_mean - coverage_hw,
                "coverage_high": coverage_mean + coverage_hw,
                "ratio_mean": ratio_mean,
                "ratio_hw": ratio_hw,
                "ratio_low": ratio_mean - ratio_hw,
                "ratio_high": ratio_mean + ratio_hw,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def mark_heatmap_cells(data: pd.DataFrame, target: float) -> tuple[pd.DataFrame, str]:
    data = data.copy()
    data["coverage_highlight"] = data["coverage_high"] >= target
    coverage_rule = "coverage 95% CI reaches target"

    if not data["coverage_highlight"].any():
        distances = data.apply(
            lambda row: interval_distance_to_target(row["coverage_low"], row["coverage_high"], target),
            axis=1,
        )
        reference_idx = distances.idxmin()
        ref_low = data.loc[reference_idx, "coverage_low"]
        ref_high = data.loc[reference_idx, "coverage_high"]
        data["coverage_highlight"] = data.apply(
            lambda row: interval_intersects(row["coverage_low"], row["coverage_high"], ref_low, ref_high),
            axis=1,
        )
        coverage_rule = "closest-to-target coverage CI"

    best_idx = data["ratio_mean"].idxmax()
    best_low = data.loc[best_idx, "ratio_low"]
    best_high = data.loc[best_idx, "ratio_high"]
    data["ratio_highlight"] = data.apply(
        lambda row: bool(
            interval_intersects(row["ratio_low"], row["ratio_high"], best_low, best_high)
        ),
        axis=1,
    )
    data["coverage_rule"] = coverage_rule
    return data, coverage_rule


def format_cell(mean: float, half_width: float) -> str:
    return f"{mean:.3f}\n({half_width:.3f})"


def draw_heatmap_panel(
    ax: plt.Axes,
    values: np.ndarray,
    half_widths: np.ndarray,
    highlights: np.ndarray,
    title: str,
    column_labels: list[str],
    row_labels: list[str],
    xlabel: str,
    ylabel: str,
    highlight_color: str,
) -> None:
    cmap = ListedColormap(["#FFFFFF", highlight_color])
    ax.imshow(highlights.astype(int), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_title(title, fontsize=19, pad=13, fontweight="bold")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    emphasize_method_tick_labels(ax)
    ax.set_xlabel(xlabel, fontsize=17, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=17, labelpad=10)

    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#555555", linestyle="-", linewidth=0.65, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                format_cell(values[row, col], half_widths[row, col]),
                ha="center",
                va="center",
                fontsize=11.0,
                color="black",
                fontweight="bold" if highlights[row, col] else "normal",
                linespacing=0.95,
            )


def plot_heatmap(
    datasets: list[str],
    model: str,
    outlier_detector: str,
    n_rep: int,
    target: float,
    layout: str | None,
    extra_formats: bool,
) -> pd.DataFrame:
    panels = []
    rows = []
    for dataset in datasets:
        data = read_dataset(dataset, model, outlier_detector, n_rep)
        if data is None:
            continue
        data, _ = mark_heatmap_cells(data, target)
        data["dataset"] = dataset
        rows.append(data)
        panels.append(dataset)

    if not rows:
        raise FileNotFoundError("No complete outlier coverage/ratio summaries found.")

    all_data = pd.concat(rows, ignore_index=True)
    method_labels = [label for _, label in METHODS]
    datasets = panels
    shape = (len(method_labels), len(datasets))
    coverage_values = np.full(shape, np.nan)
    coverage_hw = np.full(shape, np.nan)
    coverage_highlight = np.zeros(shape, dtype=bool)
    ratio_values = np.full(shape, np.nan)
    ratio_hw = np.full(shape, np.nan)
    ratio_highlight = np.zeros(shape, dtype=bool)

    method_index = {label: idx for idx, label in enumerate(method_labels)}
    dataset_index = {dataset: idx for idx, dataset in enumerate(datasets)}
    for _, row in all_data.iterrows():
        i = method_index[row["label"]]
        j = dataset_index[row["dataset"]]
        coverage_values[i, j] = row["coverage_mean"]
        coverage_hw[i, j] = row["coverage_hw"]
        coverage_highlight[i, j] = row["coverage_highlight"]
        ratio_values[i, j] = row["ratio_mean"]
        ratio_hw[i, j] = row["ratio_hw"]
        ratio_highlight[i, j] = row["ratio_highlight"]

    if layout is None:
        layout = "landscape"

    coverage_color = "#D99AA5"
    ratio_color = "#74A9CF"
    if layout == "portrait":
        fig, axes = plt.subplots(2, 1, figsize=(12.8, 17.2), constrained_layout=False)
        panel_specs = [
            (
                axes[0],
                coverage_values.T,
                coverage_hw.T,
                coverage_highlight.T,
                f"Outlier Coverage ({detector_label(outlier_detector)})",
                method_labels,
                datasets,
                "Methods",
                "Datasets",
                coverage_color,
            ),
            (
                axes[1],
                ratio_values.T,
                ratio_hw.T,
                ratio_highlight.T,
                f"Outlier/Inlier Ratio ({detector_label(outlier_detector)})",
                method_labels,
                datasets,
                "Methods",
                "Datasets",
                ratio_color,
            ),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(23.5, 8.0), constrained_layout=False)
        panel_specs = [
            (
                axes[0],
                coverage_values,
                coverage_hw,
                coverage_highlight,
                f"Outlier Coverage ({detector_label(outlier_detector)})",
                datasets,
                method_labels,
                "Datasets",
                "Methods",
                coverage_color,
            ),
            (
                axes[1],
                ratio_values,
                ratio_hw,
                ratio_highlight,
                f"Outlier/Inlier Ratio ({detector_label(outlier_detector)})",
                datasets,
                method_labels,
                "Datasets",
                "",
                ratio_color,
            ),
        ]

    for spec in panel_specs:
        draw_heatmap_panel(*spec)

    fig.tight_layout(rect=(0, 0.02, 1, 0.985), w_pad=1.5)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = detector_suffix(outlier_detector) or "_lof"
    png_path = FIGURES_DIR / f"outlier_coverage_ratio_heatmap{suffix}.png"
    pdf_path = FIGURES_DIR / f"outlier_coverage_ratio_heatmap{suffix}.pdf"
    csv_path = FIGURES_DIR / f"outlier_coverage_ratio_heatmap{suffix}.csv"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {png_path}")
    if extra_formats:
        fig.savefig(pdf_path, bbox_inches="tight")
        all_data.to_csv(csv_path, index=False)
        print(f"Saved {pdf_path}")
        print(f"Saved {csv_path}")
    plt.close(fig)
    return all_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model", default="qnn")
    parser.add_argument("--outlier-detector", choices=["lof", "isolation_forest"], default="lof")
    parser.add_argument("--n-rep", type=int, default=50)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--layout", choices=["landscape", "portrait"], default=None)
    parser.add_argument("--extra-formats", action="store_true", help="Also save PDF and CSV outputs.")
    return parser.parse_args()


def main() -> None:
    set_plot_style()
    args = parse_args()
    plot_heatmap(
        args.datasets,
        args.model,
        args.outlier_detector,
        args.n_rep,
        args.target,
        args.layout,
        args.extra_formats,
    )


if __name__ == "__main__":
    main()
