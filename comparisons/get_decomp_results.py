# Code for generating all data results
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

original_path = os.getcwd()
folder_path = "/results/"

def read_metrics_files(datasets):
    # Create DataFrame with datasets names
    data_dict_boxplot = {}
    data_dict_barplot = {}
    for file_name in datasets:
        file_path_obs_inlier = (
                original_path
                + folder_path
                + f"{file_name}_unc_by_observation/{file_name}_epis_unc_inlier_obs_mean.csv"
            )
        
        file_path_obs_outlier = (
                original_path
                + folder_path
                + f"{file_name}_unc_by_observation/{file_name}_epis_unc_outlier_obs_mean.csv"
            )

        # boxplot data — simply append rows (same columns) and keep two variables
        inlier_obs = pd.read_csv(file_path_obs_inlier).iloc[:, 0].values
        outlier_obs = pd.read_csv(file_path_obs_outlier).iloc[:, 0].values

        inlier_lab = np.repeat("inlier", len(inlier_obs))
        outlier_lab = np.repeat("outlier", len(outlier_obs))
        data_boxplot = pd.DataFrame({"epistemic_uncertainty": np.concatenate([inlier_obs, outlier_obs]), 
                                     "type": np.concatenate([inlier_lab, outlier_lab])})
        data_dict_boxplot[file_name] = data_boxplot

        # barplot data
        file_path_obs_summary = (
                original_path
                + folder_path
                + f"{file_name}_unc_summary/{file_name}_general_summary.csv"
            )
        
        data_summary = pd.read_csv(file_path_obs_summary)
        inlier_mean, inlier_std = data_summary.iloc[0, 2], data_summary.iloc[0, 3]
        outlier_mean, outlier_std = data_summary.iloc[1, 2], data_summary.iloc[1, 3]

        data_barplot = pd.DataFrame({
            "type": [inlier_lab, outlier_lab],
            "mean": [inlier_mean, outlier_mean],
            "se": [2*inlier_std/(30**0.5), 2*outlier_std/(30**0.5)]
        })
        data_dict_barplot[file_name] = data_barplot
            
    return data_dict_boxplot, data_dict_barplot

file_names = [
    "concrete",
    "airfoil",
    "winewhite",
    "star",
    "winered",
    "cycle",
    "electric",
    "meps19",
    "superconductivity",
    "homes",
    "protein",
    "WEC",
]

data_boxplot, data_barplot = read_metrics_files(file_names)

# pure matplotlib boxplots (no seaborn), scatter removed; show y-ticks only on subplot 0 and 5
# increase fonts globally and tighten layout to remove extra white space
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

fig_box, axes_box = plt.subplots(2, 6, figsize=(24, 10), sharex=False)
axes_box = axes_box.flatten()
colors = ["C0", "C1"]
show_ytick_idxs = {0, 6}  # show y ticks only for the 1st and 6th subplot (0-based indices)

for i, (name, df) in enumerate(data_boxplot.items()):
    ax = axes_box[i]

    # normalize possible array-like 'type' cells to scalars
    types = df["type"].apply(lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) else x)

    inlier = df.loc[types == "inlier", "epistemic_uncertainty"].astype(float).values
    outlier = df.loc[types == "outlier", "epistemic_uncertainty"].astype(float).values

    # ensure non-empty arrays for boxplot (matplotlib can choke on empty lists)
    data = [inlier if inlier.size > 0 else np.array([np.nan]),
            outlier if outlier.size > 0 else np.array([np.nan])]

    # horizontal boxplot
    bp = ax.boxplot(data, vert=False, labels=["inlier", "outlier"], widths=0.6,
                    patch_artist=True, showfliers=False)

    # style boxes and medians
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_color("k")
        median.set_linewidth(1.6)

    # tighten vertical spacing for category positions and make titles/labels larger
    ax.set_ylim(0.5, 2.5)
    ax.set_title(name, fontsize=18)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # show y-ticks (labels) only for specified subplots, hide for others
    if i in show_ytick_idxs:
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["inlier", "outlier"], fontsize=16)
    else:
        ax.set_yticks([])

    # do not set per-axis x labels; we'll add one centralized xlabel below the last row

# hide any unused subplots
for j in range(len(data_boxplot), 12):
    axes_box[j].axis("off")

# decrease bottom margin to bring the centralized xlabel closer to the last row
fig_box.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.08, hspace=0.25, wspace=0.18)

# add one centralized xlabel below the last row, moved closer and font size 16
fig_box.text(0.5, 0.06, "Epistemic uncertainty", ha="center", va="center", fontsize=16)

# finalize layout and show
fig_box.tight_layout(rect=[0.06, 0.08, 0.99, 0.95])
plt.show()

# --- Barplots with error bars: 2 rows x 6 columns (vertical bars, thinner) ---
fig_bar, axes_bar = plt.subplots(2, 6, figsize=(24, 8), sharex=False)
axes_bar = axes_bar.flatten()

# reduce left margin so there's less extra space, use small labelpad for y-labels
fig_bar.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, hspace=0.35, wspace=0.3)

def _scalar_type(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0]
    return x

# show y-axis label only for the 1st and 6th subplot (0-based indices: 0 and 6)
show_ylabel_idxs = {0, 6}

for i, (name, df) in enumerate(data_barplot.items()):
    ax = axes_bar[i]
    types_raw = df["type"].tolist()
    types = [_scalar_type(t) for t in types_raw]
    means = np.array(df["mean"], dtype=float)
    ses = np.array(df["se"], dtype=float)

    positions = np.arange(len(types))
    colors = ["C0", "C1"][: len(types)]

    # vertical bars, slightly thinner width
    width = 0.4
    bars = ax.bar(positions, means, color=colors, width=width, align="center")
    for pos, mean, se in zip(positions, means, ses):
        ax.errorbar(pos, mean, yerr=se, fmt="none", ecolor="k", capsize=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(types)

    # remove the x-axis (type) tick labels for the first row of subplots
    if i < 6:
        ax.tick_params(axis="x", labelbottom=False)

    # set y-label only for selected subplots, use small labelpad to avoid extra left space
    if i in show_ylabel_idxs:
        ax.set_ylabel("Epistemic uncertainty", rotation=90, labelpad=6)
    else:
        ax.set_ylabel("")

    ax.set_title(name)

# Hide any unused subplots
for j in range(len(data_barplot), 12):
    axes_bar[j].axis("off")

# apply tight layout taking into account the reduced left margin
fig_bar.tight_layout(rect=[0.06, 0.05, 0.98, 0.95])
plt.show()
