# Code for generating all data results
import pandas as pd
import os

original_path = os.getcwd()
folder_path = "/results/"


def read_metrics_files(datasets, methods, model = "catboost", metric = "isl", outlier=False):
    # Create DataFrame with datasets names
    dataframe = pd.DataFrame({"Dataset": datasets})
    for file_name in datasets:
        if not outlier:
            file_path = (
                original_path
                + folder_path
                + f"{file_name}_{model}_summary/{file_name}_{metric}_summary.csv"
            )
        else:
            file_path = (
                original_path
                + folder_path
                + f"{file_name}_{model}_summary/{file_name}_{metric}_outlier_summary.csv"
            )

        data = pd.read_csv(file_path)

        filtered_data = data[data["methods"].isin(methods)].reset_index(drop=True)

        for method in methods:
            if method in filtered_data["methods"].values:
                value_mean = filtered_data[filtered_data["methods"] == method].iloc[
                    0, 2
                ]
                value_std = filtered_data[filtered_data["methods"] == method].iloc[
                    0, 3
                ]
                if not pd.isnull(value_mean):
                    value_std = 2 * float(value_std) / (30**0.5)
                    value_mean, value_std = round(float(value_mean), 3), round(
                        value_std, 3
                    )
                    value = f"{value_mean} ({value_std})"
                    dataframe.loc[dataframe["Dataset"] == file_name, method] = value
            else:
                dataframe.loc[dataframe["Dataset"] == file_name, method] = None
            
    return dataframe


# Define metrics and datset names
# First for quantile regression
# OBS: In the paper, it is called EPICSCORE, but the CSV data was labeled ECP. They are the same method.
methods = ["credo_GP","credo_QNN",
           "cqr", "cqrr", 
           "uacqrs", "uacqrp", 
           "EPIC"]

file_names = [
    "airfoil",
    "concrete",
    "cycle",
    "electric",
    "homes",
    "meps19",
    "news",
    "protein",
    "superconductivity",
    "winered",
    "winewhite",
]

# Saving each results in pickle files
def save_all_res(file_names, methods, model = "catboost"):
        # ratio
        result_ratio = read_metrics_files(
            file_names,
            methods,
            model,
            metric = "ratio",
            outlier = True,
            )

        # coverage outlier
        result_cover_out = read_metrics_files(
            file_names,
            methods,
            model,
            metric = "coverage",
            outlier=True,
            )

        # AISL outlier
        result_aisl_out = read_metrics_files(
            file_names,
            methods,
            model,
            metric = "isl",
            outlier=True,
        )

        # IL
        results_il = read_metrics_files(
            file_names,
            methods,
            model,
            metric="IL",
        )

        # ISL
        results_isl = read_metrics_files(
            file_names,
            methods,
            model,
            metric="isl"
        )

        # pcorr
        results_pcorr = read_metrics_files(
            file_names,
            methods,
            model,
            metric="pcorr"
        )

        # coverage
        results_cover = read_metrics_files(
            file_names,
            methods,
            model,
            metric="coverage"
        )


        os.makedirs(original_path + "/results/final_tables", exist_ok=True)
        results_isl.to_pickle(
            original_path + 
            "/results/final_tables/result_aisl.pkl"
        )
        results_cover.to_pickle(
            original_path
            + "/results/final_tables/result_cover.pkl"
        )
        results_il.to_pickle(
            original_path
            + "/results/final_tables/result_il.pkl"
        )
        results_pcorr.to_pickle(
            original_path
            + "/results/final_tables/result_pcorr.pkl"
        )

        result_ratio.to_pickle(
            original_path
            + "/results/final_tables/result_ratio_outlier.pkl"
        )
        result_aisl_out.to_pickle(
            original_path
            + "/results/final_tables/result_aisl_outlier.pkl"
        )
        result_cover_out.to_pickle(
            original_path
            + "/results/final_tables/result_cover_outlier.pkl"
        )

        return [results_isl, results_cover, results_il, 
                results_pcorr, result_ratio, result_aisl_out, 
                result_cover_out]


res_list = save_all_res(file_names, methods)
print(res_list)

result = res_list[0]
# tables
print(f"\\begin{{tabular}}{{l{'c' * len(methods)}}}\\toprule")
print("Dataset & " + " & ".join(methods) + "\\\\\\midrule")
for _, row in result.iterrows():
    values = [row["Dataset"]]
    for metric in result.columns[1:]:
        value = row[metric]
        if pd.isna(value):
            values.append("-")
        else:
            values.append(f"{value}")
    print(" & ".join(values) + "\\\\")
print("\\bottomrule\\end{tabular}")