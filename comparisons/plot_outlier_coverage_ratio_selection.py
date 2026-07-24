"""Plot outlier coverage versus outlier/inlier interval-length ratio.

Each panel is one dataset.  Points are methods, x is the outlier/inlier ratio,
and y is outlier coverage.  Error bars are shown only on x, using the same
95% CI half-width used in the tables: 2 * sd / sqrt(n_rep).

Selection rule:
- Prefer methods with adequate outlier coverage: upper 95% CI endpoint >= target.
- If no method is compatible with target coverage, use the method whose coverage CI is
  closest to the target as reference and keep methods whose coverage CI
  intersects it.
- Independently, mark globally competitive ratios: the largest ratio and all
  methods whose ratio CI intersects the best-ratio CI.
- Selected methods satisfy both criteria.
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
from matplotlib.lines import Line2D
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

METHOD_COLORS = {
    "CREDO": "#0072B2",
    "CREDO adap.": "#009E73",
    "CQR": "#D55E00",
    "CQR-r": "#CC79A7",
    "UACQRS": "#E69F00",
    "UACQRP": "#56B4E9",
    "EPIC": "#6A3D9A",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.2,
            "legend.fontsize": 11.5,
        }
    )


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


def mark_selected(data: pd.DataFrame, target: float) -> tuple[pd.DataFrame, str]:
    data = data.copy()
    data["coverage_eligible"] = data["coverage_high"] >= target
    coverage_rule = "coverage 95% CI reaches target"

    if not data["coverage_eligible"].any():
        distances = data.apply(
            lambda row: interval_distance_to_target(
                row["coverage_low"],
                row["coverage_high"],
                target,
            ),
            axis=1,
        )
        reference_idx = distances.idxmin()
        ref_low = data.loc[reference_idx, "coverage_low"]
        ref_high = data.loc[reference_idx, "coverage_high"]
        data["coverage_eligible"] = data.apply(
            lambda row: interval_intersects(
                row["coverage_low"],
                row["coverage_high"],
                ref_low,
                ref_high,
            ),
            axis=1,
        )
        coverage_rule = "closest coverage CI to target"

    best_idx = data["ratio_mean"].idxmax()
    best_low = data.loc[best_idx, "ratio_low"]
    best_high = data.loc[best_idx, "ratio_high"]
    data["ratio_competitive"] = data.apply(
        lambda row: interval_intersects(
            row["ratio_low"],
            row["ratio_high"],
            best_low,
            best_high,
        ),
        axis=1,
    )
    data["selected"] = data["coverage_eligible"] & data["ratio_competitive"]
    return data, coverage_rule


def add_method_label(ax, x: float, y: float, label: str, selected: bool) -> None:
    ax.annotate(
        label,
        (x, y),
        xytext=(4, 4 if selected else 3),
        textcoords="offset points",
        fontsize=7.4,
        fontweight="bold" if selected else "normal",
        color="black" if selected else "#444444",
        alpha=0.95 if selected else 0.75,
    )


def plot_scatter(
    datasets: list[str],
    model: str,
    outlier_detector: str,
    n_rep: int,
    target: float,
    extra_formats: bool,
    save_scatter: bool,
) -> pd.DataFrame:
    panel_data = []
    selection_rows = []
    for dataset in datasets:
        data = read_dataset(dataset, model, outlier_detector, n_rep)
        if data is None:
            continue
        data, coverage_rule = mark_selected(data, target)
        data["coverage_rule"] = coverage_rule
        panel_data.append((dataset, data))
        selection_rows.append(data)

    if not panel_data:
        raise FileNotFoundError("No complete outlier coverage/ratio summaries found.")

    selected = pd.concat(selection_rows, ignore_index=True)
    if not save_scatter:
        return selected

    n_cols = 4
    n_rows = math.ceil(len(panel_data) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.15 * n_cols, 3.75 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for ax, (dataset, data) in zip(axes.ravel(), panel_data):
        x = data["ratio_mean"].to_numpy()
        y = data["coverage_mean"].to_numpy()
        x_hw = data["ratio_hw"].to_numpy()
        y_min = min(target - 0.035, float((data["coverage_mean"] - 0.01).min()))
        y_max = max(target + 0.035, float((data["coverage_mean"] + 0.01).max()))
        x_min = max(0.0, float((data["ratio_mean"] - data["ratio_hw"]).min()) - 0.04)
        x_max = float((data["ratio_mean"] + data["ratio_hw"]).max()) + 0.06

        ax.axhline(target, color="black", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.45)

        for _, row in data.iterrows():
            selected = bool(row["selected"])
            eligible = bool(row["coverage_eligible"])
            color = METHOD_COLORS[row["label"]]
            alpha = 1.0 if eligible else 0.28
            marker = "*" if selected else "o"
            size = 130 if selected else 42
            linewidth = 1.4 if selected else 0.9
            edgecolor = "black" if selected else color

            ax.errorbar(
                row["ratio_mean"],
                row["coverage_mean"],
                xerr=row["ratio_hw"],
                fmt="none",
                ecolor=color,
                elinewidth=1.1 if selected else 0.8,
                capsize=2.3,
                alpha=alpha,
                zorder=2 if not selected else 4,
            )
            ax.scatter(
                row["ratio_mean"],
                row["coverage_mean"],
                s=size,
                marker=marker,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=3 if not selected else 5,
            )
        ax.set_title(dataset, fontsize=14, pad=7)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.18)

    for ax in axes.ravel()[len(panel_data) :]:
        ax.axis("off")

    for row in range(n_rows):
        axes[row, 0].set_ylabel("Outlier coverage")
    for col in range(n_cols):
        axes[-1, col].set_xlabel("Outlier/inlier interval-length ratio")

    handles = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label=f"target coverage = {target:g}"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=0.9, label="ratio = 1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#777777", label="coverage eligible"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#BBBBBB", markeredgecolor="#BBBBBB", alpha=0.35, label="not coverage eligible"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#777777", markeredgecolor="black", markersize=12, label="coverage + competitive ratio"),
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[label],
            markeredgecolor=METHOD_COLORS[label],
            label=label,
            markersize=6,
        )
        for _, label in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.018))
    fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=len(method_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.992),
        fontsize=11.0,
    )
    fig.suptitle(
        f"Outlier Coverage vs. Ratio ({detector_label(outlier_detector)})",
        y=1.045,
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = detector_suffix(outlier_detector) or "_lof"
    png_path = FIGURES_DIR / f"outlier_coverage_ratio_selection{suffix}.png"
    pdf_path = FIGURES_DIR / f"outlier_coverage_ratio_selection{suffix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    csv_path = FIGURES_DIR / f"outlier_coverage_ratio_selection{suffix}.csv"
    print(f"Saved {png_path}")
    if extra_formats:
        fig.savefig(pdf_path, bbox_inches="tight")
        selected.to_csv(csv_path, index=False)
        print(f"Saved {pdf_path}")
        print(f"Saved {csv_path}")
    plt.close(fig)
    return selected


def plot_summary_barplot(
    selection: pd.DataFrame,
    outlier_detector: str,
    target: float,
    extra_formats: bool,
) -> None:
    method_labels = [label for _, label in METHODS]
    counts = (
        selection.loc[selection["selected"]]
        .groupby("label")["dataset"]
        .nunique()
        .reindex(method_labels, fill_value=0)
    )
    coverage_counts = (
        selection.loc[selection["coverage_eligible"]]
        .groupby("label")["dataset"]
        .nunique()
        .reindex(method_labels, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    x = np.arange(len(method_labels))
    colors = [METHOD_COLORS[label] for label in method_labels]

    ax.bar(
        x,
        coverage_counts.to_numpy(),
        color=colors,
        alpha=0.22,
        edgecolor=colors,
        linewidth=1.2,
        label="adequate coverage",
    )
    ax.bar(
        x,
        counts.to_numpy(),
        color=colors,
        alpha=0.9,
        label="adequate coverage + competitive ratio",
    )

    for idx, value in enumerate(counts.to_numpy()):
        ax.text(idx, value + 0.25, f"{int(value)}/15", ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=25, ha="right", fontsize=13)
    ax.set_ylabel("Number of datasets", fontsize=14)
    ax.set_ylim(0, max(15, int(coverage_counts.max()) + 2))
    ax.set_title(
        f"Coverage-Adjusted Outlier Ratio Summary ({detector_label(outlier_detector)})",
        fontsize=17,
        fontweight="bold",
        pad=42,
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    suffix = detector_suffix(outlier_detector) or "_lof"
    png_path = FIGURES_DIR / f"outlier_coverage_ratio_selection_summary{suffix}.png"
    pdf_path = FIGURES_DIR / f"outlier_coverage_ratio_selection_summary{suffix}.pdf"
    csv_path = FIGURES_DIR / f"outlier_coverage_ratio_selection_summary{suffix}.csv"
    summary = pd.DataFrame(
        {
            "method": method_labels,
            "coverage_eligible_datasets": coverage_counts.to_numpy(),
            "selected_datasets": counts.to_numpy(),
        }
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {png_path}")
    if extra_formats:
        summary.to_csv(csv_path, index=False)
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved {pdf_path}")
        print(f"Saved {csv_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model", default="qnn")
    parser.add_argument("--outlier-detector", choices=["lof", "isolation_forest"], default="lof")
    parser.add_argument("--n-rep", type=int, default=50)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--extra-formats", action="store_true", help="Also save PDF and CSV outputs.")
    parser.add_argument("--summary-only", action="store_true", help="Skip the scatter plot and save only the summary barplot.")
    return parser.parse_args()


def main() -> None:
    set_plot_style()
    args = parse_args()
    selected = plot_scatter(
        args.datasets,
        args.model,
        args.outlier_detector,
        args.n_rep,
        args.target,
        args.extra_formats,
        not args.summary_only,
    )
    plot_summary_barplot(selected, args.outlier_detector, args.target, args.extra_formats)
    print("\nSelected methods by dataset:")
    for dataset, data in selected.groupby("dataset", sort=False):
        labels = ", ".join(data.loc[data["selected"], "label"].tolist())
        rule = data["coverage_rule"].iloc[0]
        print(f"- {dataset}: {labels} ({rule})")


if __name__ == "__main__":
    main()
