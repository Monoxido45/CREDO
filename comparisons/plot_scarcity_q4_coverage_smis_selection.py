"""Plot scarcity-Q4 coverage and SMIS selection summaries.

The coverage panel highlights methods whose 95% CI reaches the target
coverage, i.e. upper CI endpoint >= target. If every method is incompatible
with the target for a dataset, it highlights the methods whose 95% CIs
intersect the closest-to-target method.

The SMIS panel highlights methods with globally competitive SMIS independently
of the coverage panel. The summary barplot combines both criteria: valid
scarcity-Q4 coverage and globally competitive SMIS.
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

DISPLAY_NAMES = {
    "qsar_fish_toxicity": "qsar_fish_toxicity",
    "concrete": "concrete",
    "airfoil": "airfoil",
    "winered": "winered",
    "communities": "communities",
    "star": "star",
    "abalone": "abalone",
    "winewhite": "winewhite",
    "cycle": "cycle",
    "electric": "electric",
    "meps19": "meps19",
    "superconductivity": "superconductivity",
    "homes": "homes",
    "protein": "protein",
    "WEC": "WEC",
}

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

# Display SMIS values in dataset-specific units. A label "x10^4" means the
# displayed value is the original SMIS divided by 10^4.
SMIS_SCALE_EXPONENTS = {
    "star": 2,
    "electric": -2,
    "homes": 4,
    "WEC": 4,
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 14,
            "legend.fontsize": 12.5,
        }
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


def metric_path(dataset: str, model: str, metric: str) -> Path:
    return RESULTS_DIR / f"{dataset}_{model}_summary" / f"{dataset}_{metric}_summary.csv"


def read_dataset(dataset: str, model: str, n_rep: int, target: float) -> pd.DataFrame | None:
    coverage_path = metric_path(dataset, model, "scarcity_coverage")
    smis_path = metric_path(dataset, model, "isl")
    if not coverage_path.exists() or not smis_path.exists():
        return None

    coverage = pd.read_csv(coverage_path)
    coverage = coverage[coverage["scarcity_bin"].eq("Q4")]
    smis = pd.read_csv(smis_path)

    rows = []
    for method, label in METHODS:
        coverage_row = coverage[coverage["methods"].eq(method)]
        smis_row = smis[smis["methods"].eq(method)]
        if coverage_row.empty or smis_row.empty:
            continue
        coverage_mean = float(coverage_row.iloc[0]["mean"])
        coverage_hw = 2 * float(coverage_row.iloc[0]["sd"]) / math.sqrt(n_rep)
        smis_mean = float(smis_row.iloc[0]["mean"])
        smis_hw = 2 * float(smis_row.iloc[0]["sd"]) / math.sqrt(n_rep)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "label": label,
                "q4_coverage_mean": coverage_mean,
                "q4_coverage_hw": coverage_hw,
                "q4_coverage_low": coverage_mean - coverage_hw,
                "q4_coverage_high": coverage_mean + coverage_hw,
                "smis_mean": smis_mean,
                "smis_hw": smis_hw,
                "smis_low": smis_mean - smis_hw,
                "smis_high": smis_mean + smis_hw,
            }
        )
    if not rows:
        return None

    data = pd.DataFrame(rows)
    data = mark_cells(data, target)
    return data


def mark_cells(data: pd.DataFrame, target: float) -> pd.DataFrame:
    data = data.copy()
    data["coverage_eligible"] = data["q4_coverage_high"] >= target
    coverage_rule = "Q4 coverage 95% CI reaches target"

    if not data["coverage_eligible"].any():
        distances = data.apply(
            lambda row: interval_distance_to_target(
                row["q4_coverage_low"],
                row["q4_coverage_high"],
                target,
            ),
            axis=1,
        )
        reference_idx = distances.idxmin()
        ref_low = data.loc[reference_idx, "q4_coverage_low"]
        ref_high = data.loc[reference_idx, "q4_coverage_high"]
        data["coverage_eligible"] = data.apply(
            lambda row: interval_intersects(
                row["q4_coverage_low"],
                row["q4_coverage_high"],
                ref_low,
                ref_high,
            ),
            axis=1,
        )
        coverage_rule = "closest Q4 coverage CI to target"

    best_smis_idx = data["smis_mean"].idxmin()
    best_smis_low = data.loc[best_smis_idx, "smis_low"]
    best_smis_high = data.loc[best_smis_idx, "smis_high"]
    data["smis_competitive"] = data.apply(
        lambda row: interval_intersects(
            row["smis_low"],
            row["smis_high"],
            best_smis_low,
            best_smis_high,
        ),
        axis=1,
    )

    data["selected"] = data["coverage_eligible"] & data["smis_competitive"]
    data["coverage_rule"] = coverage_rule
    return data


def format_cell(mean: float, half_width: float) -> str:
    return f"{mean:.3f}\n({half_width:.3f})"


def scaled_smis_values(dataset: str, values: np.ndarray) -> np.ndarray:
    exponent = SMIS_SCALE_EXPONENTS.get(dataset, 0)
    return values / (10.0**exponent)


def dataset_label(dataset: str, include_smis_scale: bool = False) -> str:
    label = DISPLAY_NAMES.get(dataset, dataset)
    if include_smis_scale:
        exponent = SMIS_SCALE_EXPONENTS.get(dataset, 0)
        if exponent:
            label += "\n" + rf"$\times 10^{{{exponent}}}$"
    return label


def draw_heatmap_panel(
    ax: plt.Axes,
    values: np.ndarray,
    half_widths: np.ndarray,
    highlights: np.ndarray,
    title: str,
    column_labels: list[str],
    row_labels: list[str],
    highlight_color: str,
) -> None:
    cmap = ListedColormap(["#FFFFFF", highlight_color])
    ax.imshow(highlights.astype(int), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=18, pad=12, fontweight="bold")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Datasets", fontsize=16, labelpad=10)
    ax.set_ylabel("Methods", fontsize=16, labelpad=10)
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
                fontsize=10.4,
                color="black",
                fontweight="bold" if highlights[row, col] else "normal",
                linespacing=0.95,
            )


def collect_data(datasets: list[str], model: str, n_rep: int, target: float) -> tuple[list[str], pd.DataFrame]:
    rows = []
    valid_datasets = []
    for dataset in datasets:
        data = read_dataset(dataset, model, n_rep, target)
        if data is None:
            continue
        rows.append(data)
        valid_datasets.append(dataset)
    if not rows:
        raise FileNotFoundError("No complete scarcity-Q4 coverage and SMIS summaries found.")
    return valid_datasets, pd.concat(rows, ignore_index=True)


def matrices(datasets: list[str], data: pd.DataFrame) -> dict[str, np.ndarray]:
    method_labels = [label for _, label in METHODS]
    shape = (len(method_labels), len(datasets))
    output = {
        "coverage_values": np.full(shape, np.nan),
        "coverage_hw": np.full(shape, np.nan),
        "coverage_highlight": np.zeros(shape, dtype=bool),
        "smis_values": np.full(shape, np.nan),
        "smis_hw": np.full(shape, np.nan),
        "smis_highlight": np.zeros(shape, dtype=bool),
    }
    method_index = {label: idx for idx, label in enumerate(method_labels)}
    dataset_index = {dataset: idx for idx, dataset in enumerate(datasets)}
    for _, row in data.iterrows():
        i = method_index[row["label"]]
        j = dataset_index[row["dataset"]]
        output["coverage_values"][i, j] = row["q4_coverage_mean"]
        output["coverage_hw"][i, j] = row["q4_coverage_hw"]
        output["coverage_highlight"][i, j] = row["coverage_eligible"]
        output["smis_values"][i, j] = row["smis_mean"]
        output["smis_hw"][i, j] = row["smis_hw"]
        output["smis_highlight"][i, j] = row["smis_competitive"]

    for j, dataset in enumerate(datasets):
        output["smis_values"][:, j] = scaled_smis_values(dataset, output["smis_values"][:, j])
        output["smis_hw"][:, j] = scaled_smis_values(dataset, output["smis_hw"][:, j])
    return output


def plot_heatmap(datasets: list[str], data: pd.DataFrame, extra_formats: bool) -> None:
    method_labels = [label for _, label in METHODS]
    mats = matrices(datasets, data)
    fig, axes = plt.subplots(1, 2, figsize=(22, 7.4), constrained_layout=False)
    draw_heatmap_panel(
        axes[0],
        mats["coverage_values"],
        mats["coverage_hw"],
        mats["coverage_highlight"],
        "Scarcity-Q4 Coverage",
        [dataset_label(dataset) for dataset in datasets],
        method_labels,
        "#D99AA5",
    )
    draw_heatmap_panel(
        axes[1],
        mats["smis_values"],
        mats["smis_hw"],
        mats["smis_highlight"],
        "SMIS",
        [dataset_label(dataset, include_smis_scale=True) for dataset in datasets],
        method_labels,
        "#74A9CF",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.985), w_pad=2.2)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / "scarcity_q4_coverage_smis_heatmap.png"
    pdf_path = FIGURES_DIR / "scarcity_q4_coverage_smis_heatmap.pdf"
    csv_path = FIGURES_DIR / "scarcity_q4_coverage_smis_heatmap.csv"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {png_path}")
    if extra_formats:
        fig.savefig(pdf_path, bbox_inches="tight")
        data.to_csv(csv_path, index=False)
        print(f"Saved {pdf_path}")
        print(f"Saved {csv_path}")
    plt.close(fig)


def plot_summary_barplot(datasets: list[str], data: pd.DataFrame, extra_formats: bool) -> None:
    method_labels = [label for _, label in METHODS]
    coverage_counts = (
        data.loc[data["coverage_eligible"]]
        .groupby("label")["dataset"]
        .nunique()
        .reindex(method_labels, fill_value=0)
    )
    selected_counts = (
        data.loc[data["selected"]]
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
        label="valid scarcity-Q4 coverage",
    )
    ax.bar(
        x,
        selected_counts.to_numpy(),
        color=colors,
        alpha=0.9,
        label="valid scarcity-Q4 coverage + competitive SMIS",
    )

    denominator = len(datasets)
    for idx, value in enumerate(selected_counts.to_numpy()):
        ax.text(idx, value + 0.25, f"{int(value)}/{denominator}", ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=25, ha="right", fontsize=13)
    ax.set_ylabel("Number of datasets", fontsize=14)
    ax.set_ylim(0, max(denominator, int(coverage_counts.max()) + 2))
    ax.set_title("Scarcity-Q4 Coverage-Adjusted SMIS Summary", fontsize=17, fontweight="bold", pad=42)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    summary = pd.DataFrame(
        {
            "method": method_labels,
            "valid_scarcity_q4_coverage_datasets": coverage_counts.to_numpy(),
            "selected_datasets": selected_counts.to_numpy(),
        }
    )
    png_path = FIGURES_DIR / "scarcity_q4_coverage_smis_summary.png"
    pdf_path = FIGURES_DIR / "scarcity_q4_coverage_smis_summary.pdf"
    csv_path = FIGURES_DIR / "scarcity_q4_coverage_smis_summary.csv"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {png_path}")
    if extra_formats:
        fig.savefig(pdf_path, bbox_inches="tight")
        summary.to_csv(csv_path, index=False)
        print(f"Saved {pdf_path}")
        print(f"Saved {csv_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model", default="qnn")
    parser.add_argument("--n-rep", type=int, default=50)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--extra-formats", action="store_true", help="Also save PDF and CSV outputs.")
    return parser.parse_args()


def main() -> None:
    set_plot_style()
    args = parse_args()
    datasets, data = collect_data(args.datasets, args.model, args.n_rep, args.target)
    plot_heatmap(datasets, data, args.extra_formats)
    plot_summary_barplot(datasets, data, args.extra_formats)
    print("\nSelected methods by dataset:")
    for dataset, dataset_data in data.groupby("dataset", sort=False):
        labels = ", ".join(dataset_data.loc[dataset_data["selected"], "label"].tolist())
        rule = dataset_data["coverage_rule"].iloc[0]
        print(f"- {dataset}: {labels} ({rule})")


if __name__ == "__main__":
    main()
