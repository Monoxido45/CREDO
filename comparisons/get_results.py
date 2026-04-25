# Code for generating final result tables.
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
FINAL_TABLES_DIR = RESULTS_DIR / "final_tables"

METHODS = [
    "credo_QNN",
    "credo_QNN_adaptive",
    "cqr",
    "cqrr",
    "uacqrs",
    "uacqrp",
    "EPIC",
]

CATBOOST_DATASETS = [
    "airfoil",
    "concrete",
    "cycle",
    "electric",
    "homes",
    "meps19",
    "protein",
    "star",
    "superconductivity",
    "WEC",
    "winered",
    "winewhite",
]

QNN_DATASET_ORDER = [
    "concrete",
    "airfoil",
    "winewhite",
    "star",
    "winered",
    "cycle",
    "electric",
   # "meps19",
]


def discover_datasets(model="qnn"):
    suffix = f"_{model}_summary"
    datasets = []
    for path in RESULTS_DIR.glob(f"*{suffix}"):
        if path.is_dir():
            datasets.append(path.name.removesuffix(suffix))
    ordered = [dataset for dataset in QNN_DATASET_ORDER if dataset in datasets]
    ordered.extend(sorted(set(datasets) - set(ordered)))
    return ordered


def format_mean_se(mean, sd, n_rep=30, digits=3):
    if pd.isna(mean):
        return None
    if pd.isna(sd):
        return f"{round(float(mean), digits)}"
    se = 2 * float(sd) / (n_rep**0.5)
    return f"{round(float(mean), digits)} ({round(se, digits)})"


def metric_file(dataset, model, metric, outlier=False):
    suffix = f"_{metric}_outlier_summary.csv" if outlier else f"_{metric}_summary.csv"
    return RESULTS_DIR / f"{dataset}_{model}_summary" / f"{dataset}{suffix}"


def datasets_with_metric(datasets, model, metric, outlier=False):
    return [
        dataset
        for dataset in datasets
        if metric_file(dataset, model, metric, outlier=outlier).exists()
    ]


def read_metrics_files(datasets, methods, model="catboost", metric="isl", outlier=False, n_rep=30):
    dataframe = pd.DataFrame({"Dataset": datasets})
    for dataset in datasets:
        file_path = metric_file(dataset, model, metric, outlier=outlier)
        if not file_path.exists():
            for method in methods:
                dataframe.loc[dataframe["Dataset"] == dataset, method] = None
            continue

        data = pd.read_csv(file_path)
        filtered_data = data[data["methods"].isin(methods)].reset_index(drop=True)

        for method in methods:
            method_rows = filtered_data[filtered_data["methods"] == method]
            if method_rows.empty:
                dataframe.loc[dataframe["Dataset"] == dataset, method] = None
                continue

            value = format_mean_se(
                method_rows.iloc[0]["mean"],
                method_rows.iloc[0]["sd"],
                n_rep=n_rep,
            )
            dataframe.loc[dataframe["Dataset"] == dataset, method] = value
    return dataframe


def read_scarcity_coverage_files(datasets, methods, model="qnn", bins=("Q3", "Q4"), n_rep=30):
    tables = {}
    for bin_name in bins:
        dataframe = pd.DataFrame({"Dataset": datasets})
        for dataset in datasets:
            file_path = metric_file(dataset, model, "scarcity_coverage")
            if not file_path.exists():
                for method in methods:
                    dataframe.loc[dataframe["Dataset"] == dataset, method] = None
                continue

            data = pd.read_csv(file_path)
            filtered_data = data[
                data["methods"].isin(methods) & (data["scarcity_bin"] == bin_name)
            ].reset_index(drop=True)

            for method in methods:
                method_rows = filtered_data[filtered_data["methods"] == method]
                if method_rows.empty:
                    dataframe.loc[dataframe["Dataset"] == dataset, method] = None
                    continue

                value = format_mean_se(
                    method_rows.iloc[0]["mean"],
                    method_rows.iloc[0]["sd"],
                    n_rep=n_rep,
                )
                dataframe.loc[dataframe["Dataset"] == dataset, method] = value
        tables[bin_name] = dataframe
    return tables


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_table(dataframe, methods, caption=None, label=None):
    lines = [f"\\begin{{tabular}}{{l{'c' * len(methods)}}}", "\\toprule"]
    lines.append("Dataset & " + " & ".join(latex_escape(method) for method in methods) + r"\\")
    lines.append("\\midrule")
    for _, row in dataframe.iterrows():
        values = [latex_escape(row["Dataset"])]
        for method in methods:
            value = row[method]
            values.append("-" if pd.isna(value) else latex_escape(value))
        lines.append(" & ".join(values) + r"\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    table = "\n".join(lines)
    if caption or label:
        wrapped = ["\\begin{table}[t]", "\\centering", table]
        if caption:
            wrapped.append(f"\\caption{{{caption}}}")
        if label:
            wrapped.append(f"\\label{{{label}}}")
        wrapped.append("\\end{table}")
        return "\n".join(wrapped)
    return table


def save_table(dataframe, name, methods=METHODS, caption=None, label=None):
    FINAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_pickle(FINAL_TABLES_DIR / f"{name}.pkl")
    dataframe.to_csv(FINAL_TABLES_DIR / f"{name}.csv", index=False)
    (FINAL_TABLES_DIR / f"{name}.tex").write_text(
        latex_table(dataframe, methods, caption=caption, label=label) + "\n"
    )


def save_catboost_tables(datasets=CATBOOST_DATASETS, methods=METHODS, n_rep=30):
    tables = {
        "result_aisl": read_metrics_files(datasets, methods, "catboost", metric="isl", n_rep=n_rep),
        "result_cover": read_metrics_files(datasets, methods, "catboost", metric="coverage", n_rep=n_rep),
        "result_ratio_outlier": read_metrics_files(
            datasets, methods, "catboost", metric="ratio", outlier=True, n_rep=n_rep
        ),
        "result_cover_outlier": read_metrics_files(
            datasets, methods, "catboost", metric="coverage", outlier=True, n_rep=n_rep
        ),
    }
    for name, dataframe in tables.items():
        save_table(dataframe, name, methods=methods)
    return tables


def save_qnn_tables(datasets=None, methods=METHODS, n_rep=30):
    if datasets is None:
        datasets = discover_datasets("qnn")

    smis_datasets = datasets_with_metric(datasets, "qnn", "isl")
    length_datasets = datasets_with_metric(datasets, "qnn", "interval_length")
    wsc_datasets = datasets_with_metric(datasets, "qnn", "wsc")
    scarcity_worst_datasets = datasets_with_metric(datasets, "qnn", "scarcity_worst_coverage")
    scarcity_datasets = datasets_with_metric(datasets, "qnn", "scarcity_coverage")
    coverage_outlier_datasets = datasets_with_metric(datasets, "qnn", "coverage", outlier=True)
    ratio_outlier_datasets = datasets_with_metric(datasets, "qnn", "ratio", outlier=True)

    tables = {
        "result_qnn_smis": read_metrics_files(smis_datasets, methods, "qnn", metric="isl", n_rep=n_rep),
        "result_qnn_interval_length": read_metrics_files(
            length_datasets, methods, "qnn", metric="interval_length", n_rep=n_rep
        ),
        "result_qnn_wsc": read_metrics_files(wsc_datasets, methods, "qnn", metric="wsc", n_rep=n_rep),
        "result_qnn_scarcity_worst_coverage": read_metrics_files(
            scarcity_worst_datasets, methods, "qnn", metric="scarcity_worst_coverage", n_rep=n_rep
        ),
        "result_qnn_coverage_outlier": read_metrics_files(
            coverage_outlier_datasets, methods, "qnn", metric="coverage", outlier=True, n_rep=n_rep
        ),
        "result_qnn_ratio_outlier": read_metrics_files(
            ratio_outlier_datasets, methods, "qnn", metric="ratio", outlier=True, n_rep=n_rep
        ),
    }

    scarcity_tables = read_scarcity_coverage_files(
        scarcity_datasets, methods, model="qnn", bins=("Q3", "Q4"), n_rep=n_rep
    )
    tables["result_qnn_scarcity_coverage_Q3"] = scarcity_tables["Q3"]
    tables["result_qnn_scarcity_coverage_Q4"] = scarcity_tables["Q4"]

    captions = {
        "result_qnn_smis": "SMIS results with QNN base models.",
        "result_qnn_interval_length": "Interval length results with QNN base models.",
        "result_qnn_wsc": "Worst-slab coverage results with QNN base models.",
        "result_qnn_scarcity_worst_coverage": "Worst scarcity-bin coverage results with QNN base models.",
        "result_qnn_scarcity_coverage_Q3": "Scarcity coverage in the third scarcity quantile with QNN base models.",
        "result_qnn_scarcity_coverage_Q4": "Scarcity coverage in the fourth scarcity quantile with QNN base models.",
        "result_qnn_coverage_outlier": "Outlier coverage results with QNN base models.",
        "result_qnn_ratio_outlier": "Outlier-to-inlier interval-length ratio results with QNN base models.",
    }
    for name, dataframe in tables.items():
        save_table(
            dataframe,
            name,
            methods=methods,
            caption=captions.get(name),
            label=f"tab:{name}",
        )
    return tables


if __name__ == "__main__":
    catboost_tables = save_catboost_tables()
    qnn_tables = save_qnn_tables()

    print("Saved CatBoost tables:")
    for name in catboost_tables:
        print(f"- {FINAL_TABLES_DIR / name}")

    print("\nSaved QNN tables:")
    for name, dataframe in qnn_tables.items():
        print(f"- {FINAL_TABLES_DIR / name} ({len(dataframe)} datasets)")

    print("\nPreview: QNN SMIS")
    print(latex_table(qnn_tables["result_qnn_smis"], METHODS))
