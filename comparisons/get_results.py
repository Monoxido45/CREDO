# Code for generating final result tables.
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
FINAL_TABLES_DIR = RESULTS_DIR / "legacy_tables"
WRITE_PICKLE = False
DEFAULT_CATBOOST_N_REP = 30
DEFAULT_QNN_N_REP = 50
EXCLUDED_DISCOVERY_DATASETS = {
    "bike",
    "news",
}

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
    "concrete",
    "airfoil",
    "winered",
    "communities",
    "star",
    "winewhite",
    "cycle",
    "electric",
    "meps19",
    "superconductivity",
    "homes",
    "protein",
    "WEC",
]

QNN_DATASET_ORDER = [
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

DATASET_SIZE_ORDER = [
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
    "blog",
    "WEC",
    "kernel",
]


def detector_suffix(outlier_detector):
    return "" if outlier_detector == "lof" else f"_{outlier_detector}"


def dataset_sort_key(dataset):
    try:
        size_rank = DATASET_SIZE_ORDER.index(dataset)
    except ValueError:
        size_rank = len(DATASET_SIZE_ORDER)
    return size_rank, dataset.lower()


def sort_datasets_by_size(datasets):
    return sorted(datasets, key=dataset_sort_key)


def discover_datasets(model="qnn"):
    suffix = f"_{model}_summary"
    datasets = []
    for path in RESULTS_DIR.glob(f"*{suffix}"):
        if path.is_dir():
            dataset = path.name.removesuffix(suffix)
            if dataset not in EXCLUDED_DISCOVERY_DATASETS:
                datasets.append(dataset)
    return sort_datasets_by_size(datasets)


def format_mean_se(mean, sd, n_rep=DEFAULT_CATBOOST_N_REP, digits=3):
    if pd.isna(mean):
        return None
    if pd.isna(sd):
        return f"{round(float(mean), digits)}"
    se = 2 * float(sd) / (n_rep**0.5)
    return f"{round(float(mean), digits)} ({round(se, digits)})"


def metric_file(
    dataset,
    model,
    metric,
    outlier=False,
    outlier_detector="lof",
    scarcity_method="knn",
):
    if outlier:
        suffix = f"_{metric}_outlier{detector_suffix(outlier_detector)}_summary.csv"
    elif metric in {"scarcity_coverage", "scarcity_worst_coverage"} and scarcity_method == "isolation_forest":
        suffix = f"_{metric}_isolation_forest_summary.csv"
    else:
        suffix = f"_{metric}_summary.csv"
    return RESULTS_DIR / f"{dataset}_{model}_summary" / f"{dataset}{suffix}"


def datasets_with_metric(
    datasets,
    model,
    metric,
    outlier=False,
    outlier_detector="lof",
    scarcity_method="knn",
):
    return [
        dataset
        for dataset in datasets
        if metric_file(
            dataset,
            model,
            metric,
            outlier=outlier,
            outlier_detector=outlier_detector,
            scarcity_method=scarcity_method,
        ).exists()
    ]


def read_metrics_files(
    datasets,
    methods,
    model="catboost",
    metric="isl",
    outlier=False,
    n_rep=DEFAULT_CATBOOST_N_REP,
    outlier_detector="lof",
    scarcity_method="knn",
):
    dataframe = pd.DataFrame({"Dataset": datasets})
    for dataset in datasets:
        file_path = metric_file(
            dataset,
            model,
            metric,
            outlier=outlier,
            outlier_detector=outlier_detector,
            scarcity_method=scarcity_method,
        )
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


def read_scarcity_coverage_files(
    datasets,
    methods,
    model="qnn",
    bins=("Q3", "Q4"),
    n_rep=DEFAULT_QNN_N_REP,
    scarcity_method="knn",
):
    tables = {}
    for bin_name in bins:
        dataframe = pd.DataFrame({"Dataset": datasets})
        for dataset in datasets:
            file_path = metric_file(
                dataset,
                model,
                "scarcity_coverage",
                scarcity_method=scarcity_method,
            )
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
    if WRITE_PICKLE:
        dataframe.to_pickle(FINAL_TABLES_DIR / f"{name}.pkl")
    dataframe.to_csv(FINAL_TABLES_DIR / f"{name}.csv", index=False)
    (FINAL_TABLES_DIR / f"{name}.tex").write_text(
        latex_table(dataframe, methods, caption=caption, label=label) + "\n"
    )


def save_catboost_tables(
    datasets=None,
    methods=METHODS,
    n_rep=DEFAULT_CATBOOST_N_REP,
    outlier_detector="lof",
    include_outlier=True,
):
    if datasets is None:
        datasets = discover_datasets("catboost")
    else:
        datasets = sort_datasets_by_size(datasets)

    suffix = detector_suffix(outlier_detector)
    tables = {
        "result_aisl": read_metrics_files(datasets, methods, "catboost", metric="isl", n_rep=n_rep),
        "result_cover": read_metrics_files(datasets, methods, "catboost", metric="coverage", n_rep=n_rep),
    }
    if include_outlier:
        coverage_outlier_datasets = datasets_with_metric(
            datasets, "catboost", "coverage", outlier=True, outlier_detector=outlier_detector
        )
        ratio_outlier_datasets = datasets_with_metric(
            datasets, "catboost", "ratio", outlier=True, outlier_detector=outlier_detector
        )
        tables[f"result_cover_outlier{suffix}"] = read_metrics_files(
            coverage_outlier_datasets,
            methods,
            "catboost",
            metric="coverage",
            outlier=True,
            n_rep=n_rep,
            outlier_detector=outlier_detector,
        )
        tables[f"result_ratio_outlier{suffix}"] = read_metrics_files(
            ratio_outlier_datasets,
            methods,
            "catboost",
            metric="ratio",
            outlier=True,
            n_rep=n_rep,
            outlier_detector=outlier_detector,
        )
    for name, dataframe in tables.items():
        save_table(dataframe, name, methods=methods)
    return tables


def save_qnn_tables(
    datasets=None,
    methods=METHODS,
    n_rep=DEFAULT_QNN_N_REP,
    outlier_detector="lof",
    model="qnn",
    table_prefix=None,
    model_label=None,
    include_outlier=True,
    scarcity_bins=("Q4",),
    scarcity_method="knn",
):
    if table_prefix is None:
        table_prefix = f"result_{model}"
    if model_label is None:
        model_label = model.upper()
    if datasets is None:
        datasets = discover_datasets(model)
    else:
        datasets = sort_datasets_by_size(datasets)

    smis_datasets = datasets_with_metric(datasets, model, "isl")
    length_datasets = datasets_with_metric(datasets, model, "interval_length")
    coverage_datasets = datasets_with_metric(datasets, model, "coverage")
    scarcity_worst_datasets = datasets_with_metric(
        datasets, model, "scarcity_worst_coverage", scarcity_method=scarcity_method
    )
    scarcity_datasets = datasets_with_metric(
        datasets, model, "scarcity_coverage", scarcity_method=scarcity_method
    )
    suffix = detector_suffix(outlier_detector)

    tables = {
        f"{table_prefix}_smis": read_metrics_files(smis_datasets, methods, model, metric="isl", n_rep=n_rep),
        f"{table_prefix}_interval_length": read_metrics_files(
            length_datasets, methods, model, metric="interval_length", n_rep=n_rep
        ),
        f"{table_prefix}_coverage": read_metrics_files(
            coverage_datasets, methods, model, metric="coverage", n_rep=n_rep
        ),
        f"{table_prefix}_scarcity_worst_coverage": read_metrics_files(
            scarcity_worst_datasets,
            methods,
            model,
            metric="scarcity_worst_coverage",
            n_rep=n_rep,
            scarcity_method=scarcity_method,
        ),
    }
    if include_outlier:
        coverage_outlier_datasets = datasets_with_metric(
            datasets, model, "coverage", outlier=True, outlier_detector=outlier_detector
        )
        ratio_outlier_datasets = datasets_with_metric(
            datasets, model, "ratio", outlier=True, outlier_detector=outlier_detector
        )
        tables[f"{table_prefix}_coverage_outlier{suffix}"] = read_metrics_files(
            coverage_outlier_datasets,
            methods,
            model,
            metric="coverage",
            outlier=True,
            n_rep=n_rep,
            outlier_detector=outlier_detector,
        )
        tables[f"{table_prefix}_ratio_outlier{suffix}"] = read_metrics_files(
            ratio_outlier_datasets,
            methods,
            model,
            metric="ratio",
            outlier=True,
            n_rep=n_rep,
            outlier_detector=outlier_detector,
        )

    scarcity_tables = read_scarcity_coverage_files(
        scarcity_datasets,
        methods,
        model=model,
        bins=scarcity_bins,
        n_rep=n_rep,
        scarcity_method=scarcity_method,
    )
    for scarcity_bin, table in scarcity_tables.items():
        tables[f"{table_prefix}_scarcity_coverage_{scarcity_bin}"] = table

    captions = {
        f"{table_prefix}_smis": f"SMIS results with {model_label} base models.",
        f"{table_prefix}_interval_length": f"Interval length results with {model_label} base models.",
        f"{table_prefix}_coverage": f"Marginal coverage results with {model_label} base models.",
        f"{table_prefix}_scarcity_worst_coverage": f"Worst scarcity-bin coverage results with {model_label} base models.",
        f"{table_prefix}_scarcity_coverage_Q4": f"Scarcity coverage in the fourth scarcity quantile with {model_label} base models.",
        f"{table_prefix}_coverage_outlier{suffix}": f"Outlier coverage results with {model_label} base models.",
        f"{table_prefix}_ratio_outlier{suffix}": f"Outlier-to-inlier interval-length ratio results with {model_label} base models.",
    }
    if "Q3" in scarcity_bins:
        captions[f"{table_prefix}_scarcity_coverage_Q3"] = (
            f"Scarcity coverage in the third scarcity quantile with {model_label} base models."
        )
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
    parser = ArgumentParser()
    parser.add_argument("--outlier_detector", choices=["lof", "isolation_forest"], default="lof")
    parser.add_argument("--model", choices=["qnn", "qnn_mc"], default="qnn")
    parser.add_argument(
        "--model-slug",
        default=None,
        help="Optional result-folder slug, e.g. qnn_scarcity_iforest or qnn_no_train_dropout.",
    )
    parser.add_argument(
        "--scarcity-method",
        choices=["knn", "isolation_forest"],
        default="knn",
        help="Scarcity score used by the requested scarcity tables.",
    )
    parser.add_argument("--output-dir", type=Path, default=FINAL_TABLES_DIR)
    parser.add_argument("--write-pickle", action="store_true", help="Also write legacy pickle table files.")
    parser.add_argument("--include-catboost", action="store_true", help="Also generate legacy CatBoost tables.")
    parser.add_argument("--include-q3", action="store_true", help="Also generate scarcity Q3 coverage tables.")
    parser.add_argument(
        "--n_rep",
        type=int,
        default=DEFAULT_QNN_N_REP,
        help="Number of repetitions used for the selected QNN table standard errors.",
    )
    parser.add_argument(
        "--catboost_n_rep",
        type=int,
        default=DEFAULT_CATBOOST_N_REP,
        help="Number of repetitions used for CatBoost table standard errors.",
    )
    parser.add_argument(
        "--main_only",
        action="store_true",
        help="Save only main tables, skipping outlier coverage/ratio tables.",
    )
    args = parser.parse_args()

    FINAL_TABLES_DIR = args.output_dir
    WRITE_PICKLE = args.write_pickle
    model_slug = args.model_slug or args.model
    table_model_label = "QNN_MC" if args.model == "qnn_mc" else "QNN"

    catboost_tables = {}
    if args.include_catboost:
        catboost_tables = save_catboost_tables(
            outlier_detector=args.outlier_detector,
            n_rep=args.catboost_n_rep,
            include_outlier=not args.main_only,
        )
    qnn_tables = save_qnn_tables(
        outlier_detector=args.outlier_detector,
        model=model_slug,
        n_rep=args.n_rep,
        table_prefix=f"result_{model_slug}",
        model_label=table_model_label,
        include_outlier=not args.main_only,
        scarcity_bins=("Q3", "Q4") if args.include_q3 else ("Q4",),
        scarcity_method=args.scarcity_method,
    )

    if catboost_tables:
        print("Saved CatBoost tables:")
        for name in catboost_tables:
            print(f"- {FINAL_TABLES_DIR / name}")

    print(f"\nSaved {model_slug.upper()} tables:")
    for name, dataframe in qnn_tables.items():
        print(f"- {FINAL_TABLES_DIR / name} ({len(dataframe)} datasets)")

    print(f"\nPreview: {model_slug.upper()} SMIS")
    print(latex_table(qnn_tables[f"result_{model_slug}_smis"], METHODS))
