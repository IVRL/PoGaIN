"""
PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
Süsstrunk, Fellow, IEEE.
"""

import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import utils
from compute_results import read_csv
from implementations import log_likelihood

color_dict = {
    "REAL": "black",
    "FOI": "C0",
    "CNN": "C1",
    "VAR": "C2",
    "OURS": "C3",
}

marker_dict = {
    "REAL": "",
    "FOI": "o",
    "CNN": "X",
    "VAR": "s",
    "OURS": "D",
}

name_dict = {
    "REAL": "\\textit{REAL}",
    "FOI": "\\textit{FOI}",
    "CNN": "\\textit{CNN}",
    "VAR": "\\textit{VAR}",
    "OURS": "\\textbf{\\textit{OURS}}",
}

sort_key = lambda x: order[x]
order = {
    "REAL": 0,
    "FOI": 1,
    "CNN": 2,
    "VAR": 3,
    "OURS": 4,
}

plt.rcParams.update(
    {
        "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
        "axes.facecolor": (0.95, 0.95, 0.95, 1.0),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    }
)


# Default is 10
SIZE = 12

# Before 16
BIGGER = 18

plt.rc("font", size=SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SIZE)  # legend fontsize
plt.rc("figure", titlesize=SIZE)  # fontsize of the figure title


plt.rcParams.update({"legend.handlelength": 0.6})
plt.rcParams.update({"legend.handletextpad": 0.5})
plt.rcParams.update({"text.usetex": True, "font.family": "cm"})


def main():
    fig_path = os.path.join(os.path.dirname(__file__), "figures")
    tab_path = os.path.join(os.path.dirname(__file__), "tables")

    results_path = os.path.join(os.path.dirname(__file__), "csvs/results.csv")

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(tab_path):
        os.makedirs(tab_path)

    df = read_csv(results_path)

    df.replace(to_replace=-np.inf, value=np.nan, inplace=True)
    df.replace(to_replace=1e-8, value=np.nan, inplace=True)

    df_wo_real = df.drop(df[df.Method == "REAL"].index)

    df_wo_foi = df_wo_real.drop(df[df.Method == "FOI"].index)

    df_wo_foi_real = df_wo_foi.drop(df_wo_foi[df_wo_foi.Method == "REAL"].index)

    df_wo_foi_real_cnn = df_wo_foi_real.drop(
        df_wo_foi_real[df_wo_foi_real.Method == "CNN"].index
    )

    # --------------------------------------------------------------------------

    fig, axs = plt.subplots(figsize=(8, 5.2), nrows=2, dpi=600)
    fig.tight_layout()

    fig.subplots_adjust(hspace=0.49)

    plot_mse(df_wo_real, ax=axs[0], letter="a")
    plot_mse(df_wo_real, ax=axs[1], letter="b")

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].set_title("MSE on ${\hat{a}}^{-1}$")
    axs[1].set_title("MSE on ${\hat{b}}^2$")

    fig.savefig(
        os.path.join(fig_path, "mse.png"),
        bbox_inches="tight",
        pad_inches=0.06,
    )

    # --------------------------------------------------------------------------

    with open(os.path.join(tab_path, "error_a.tex"), "w") as f:
        f.writelines(
            df_wo_real.groupby(["Method"])
            .apply(compute_mse, letter="a")
            .groupby("Method")
            .describe()
            .style.to_latex()
            .replace("%", " percent quartile")
        )

    with open(os.path.join(tab_path, "error_b.tex"), "w") as f:
        f.writelines(
            df_wo_real.groupby(["Method"])
            .apply(compute_mse, letter="b")
            .groupby("Method")
            .describe()
            .style.to_latex()
            .replace("%", " percent quartile")
        )

    # --------------------------------------------------------------------------

    fig, axs = plt.subplots(figsize=(8, 5.2), nrows=2, dpi=600)
    fig.tight_layout()

    fig.subplots_adjust(hspace=0.49)

    plot_bias(df_wo_foi_real_cnn, ax=axs[0], letter="a")
    plot_bias(df_wo_foi_real_cnn, ax=axs[1], letter="b")

    axs[0].axhline(y=0, color="black", linestyle="-")
    axs[1].axhline(y=0, color="black", linestyle="-")

    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].set_title("Bias of ${\hat{a}}^{-1}$")
    axs[1].set_title("Bias of ${\hat{b}}^2$")

    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.savefig(
        os.path.join(fig_path, "bias.png"),
        bbox_inches="tight",
        pad_inches=0.06,
    )

    # --------------------------------------------------------------------------

    fig, axs = plt.subplots(figsize=(8, 5.2), nrows=2, dpi=600)
    fig.tight_layout()

    fig.subplots_adjust(hspace=0.33)

    plot_mse_image_dependence(df_wo_foi_real, ax=axs[0], letter="a")
    plot_mse_image_dependence(df_wo_foi_real, ax=axs[1], letter="b")

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    axs[0].set_title("MSE on ${\hat{a}}^{-1}$")
    axs[1].set_title("MSE on ${\hat{b}}^2$")

    fig.savefig(
        os.path.join(fig_path, "mse_dependence.png"),
        bbox_inches="tight",
        pad_inches=0.06,
    )

    # --------------------------------------------------------------------------

    mse_a = df_wo_foi_real.groupby(["Method", "Name"]).apply(compute_mse, letter="a")

    inliers_a = compute_inliers(mse_a)

    percent_a = (
        mse_a[inliers_a].groupby("Method").count() / mse_a.groupby("Method").count()
    )

    with open(os.path.join(tab_path, "outliers_a.tex"), "w") as f:
        f.writelines(percent_a.reset_index().style.to_latex())

    mse_b = df_wo_foi_real.groupby(["Method", "Name"]).apply(compute_mse, letter="b")

    inliers_b = compute_inliers(mse_b)

    percent_b = (
        mse_b[inliers_b].groupby("Method").count() / mse_b.groupby("Method").count()
    )

    with open(os.path.join(tab_path, "outliers_b.tex"), "w") as f:
        f.writelines(percent_b.reset_index().style.to_latex())

    inliers_combined = pd.Series(
        inliers_a.values & inliers_b.values, index=inliers_a.index
    )

    percent_combined = (
        mse_a[inliers_combined].groupby("Method").count()
        / mse_a.groupby("Method").count()
    )

    with open(os.path.join(tab_path, "outliers_combined.tex"), "w") as f:
        f.writelines(percent_combined.reset_index().style.to_latex())

    # --------------------------------------------------------------------------

    fig, axs = plt.subplots(nrows=3, figsize=(8, 5.2), dpi=600)
    fig.tight_layout()

    fig.subplots_adjust(hspace=0.9)

    pivoted = df.pivot(
        index=["Name", "Seed", "Real_a", "Real_b"], columns="Method", values="LL"
    )

    df2 = pivoted.copy()
    df2["OURS"] = abs(((pivoted.OURS - pivoted.REAL) / pivoted.REAL))
    df2["VAR"] = abs(((pivoted.VAR - pivoted.REAL) / pivoted.REAL))
    df2["CNN"] = abs(((pivoted.CNN - pivoted.REAL) / pivoted.REAL))
    df2 = df2.stack().rename("Relative_LL").reset_index()

    temp_var = (
        df2[df2.Method == "VAR"]
        .groupby(["Real_a", "Real_b"])
        .Relative_LL.apply(np.nanmean)
        .reset_index()
    )
    temp_ours = (
        df2[df2.Method == "OURS"]
        .groupby(["Real_a", "Real_b"])
        .Relative_LL.apply(np.nanmean)
        .reset_index()
    )
    temp_cnn = (
        df2[df2.Method == "CNN"]
        .groupby(["Real_a", "Real_b"])
        .Relative_LL.apply(np.nanmean)
        .reset_index()
    )

    vmin = min([temp.Relative_LL.min() for temp in [temp_var, temp_ours, temp_cnn]])
    vmax = max([temp.Relative_LL.max() for temp in [temp_var, temp_ours, temp_cnn]])

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = "viridis"

    im = cm.ScalarMappable(norm=norm, cmap=cmap)

    plot_ll(temp_cnn, ax=axs[0], cmap=cmap, norm=norm)
    plot_ll(temp_var, ax=axs[1], cmap=cmap, norm=norm)
    plot_ll(temp_ours, ax=axs[2], cmap=cmap, norm=norm)

    axs[0].set_xlabel("a\n")
    axs[0].set_ylabel("b", rotation="vertical")
    axs[0].set_title("Error on $\mathcal{LL}$ of " + name_dict["CNN"])

    axs[1].set_xlabel("a\n")
    axs[1].set_ylabel("b", rotation="vertical")
    axs[1].set_title("Error on $\mathcal{LL}$ of " + name_dict["VAR"])

    axs[2].set_xlabel("a\n")
    axs[2].set_ylabel("b", rotation="vertical")
    axs[2].set_title("Error on $\mathcal{LL}$ of " + name_dict["OURS"])

    fig.colorbar(im, ax=axs, extend="both")
    fig.savefig(os.path.join(fig_path, "ll.png"), bbox_inches="tight", pad_inches=0.06)

    # --------------------------------------------------------------------------

    # This takes a while, so it is commented out by default
    # print("Starting k plots, this takes some time ...")
    # k_max = 200

    # a_values = np.linspace(1, 100, num=25)
    # b_values = np.linspace(0.01, 0.15, num=25)
    # n = 25

    # value_array = np.zeros(k_max + 1)
    # time_array = np.zeros(k_max + 1)

    # for a in a_values:
    #     for b in b_values:
    #         ks = np.empty((n, k_max + 1))
    #         ts = np.empty((n, k_max + 1))

    #         x = utils.create_fake_image(n)
    #         y = utils.add_noise(x, a, b)

    #         for i, k in itertools.product(range(n), range(k_max + 1)):
    #             x_i = np.array(x[i])
    #             y_i = np.array(y[i])

    #             start_time = datetime.now()
    #             ks[i, k] = log_likelihood(x_i, y_i, a, b, k)
    #             end_time = datetime.now()
    #             ts[i, k] = (end_time - start_time).total_seconds()

    #         ks = np.mean(ks, axis=0)
    #         ts = np.mean(ts, axis=0)

    #         value_array += ks
    #         time_array += ts

    # value_array /= len(a_values) * len(b_values)
    # time_array /= len(a_values) * len(b_values)

    # fig, axs = plt.subplots(nrows=2, figsize=(8, 5.2), dpi=600)
    # fig.tight_layout()

    # fig.subplots_adjust(hspace=0.4)

    # axs[0].plot(np.arange(k_max + 1), value_array)
    # axs[1].plot(np.arange(k_max + 1), time_array)

    # axs[0].set_xlabel("k")
    # axs[0].set_ylabel("Log-likelihood")

    # axs[1].set_xlabel("k")
    # axs[1].set_ylabel("Computation time in s")

    # axs[0].grid(True)
    # axs[1].grid(True)

    # axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # fig.savefig(
    #     os.path.join(fig_path, "k_plot.png"),
    #     bbox_inches="tight",
    #     pad_inches=0.06,
    # )


def scatter_plot(df: pd.DataFrame, ax):
    methods = list(df.unstack(1).index)
    methods.sort(key=sort_key)

    df["nanstd"].replace(np.nan, 0, inplace=True)

    for method in methods:
        ax.errorbar(
            df["nanmean"][method].keys(),
            df["nanmean"][method],
            color=color_dict[method],
            marker=marker_dict[method],
            yerr=df["nanstd"][method],
        )
    custom_lines = [
        Line2D(
            [0], [0], color=color_dict[i], ls="", marker=marker_dict[i], markersize=8
        )
        for i in methods
    ]

    names = [name_dict[i] for i in methods]

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(custom_lines, names, loc="center left", bbox_to_anchor=(1, 0.5))


def bar_plot(df: pd.DataFrame, ax, color=None):
    methods = list(df.unstack(1).index)
    methods.sort(key=sort_key)

    ordinate = np.array(df.unstack(0).index)
    n = len(ordinate)
    n_methods = len(methods)
    x = np.arange(n)
    width = 1 / (n_methods + 1)

    df["nanstd"].replace(np.nan, 0, inplace=True)

    for i, method in enumerate(methods):
        ax.bar(
            x + i * width,
            df["nanmean"][method],
            width,
            color=color_dict[method] if color is None else color,
            alpha=1 if color is None else 0.4,
            yerr=df["nanstd"][method],
        )

    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )

    if color:
        ax.get_legend().remove()

    custom_lines = [
        Line2D([0], [0], color=color_dict[i], ls="", marker="s", markersize=8)
        for i in methods
    ]

    names = [name_dict[i] for i in methods]

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(custom_lines, names, loc="center left", bbox_to_anchor=(1, 0.5))


def plot_mse(df: pd.DataFrame, ax, letter):
    mse_df = df.groupby(["Method", f"Real_{letter}"]).apply(compute_mse, letter=letter)
    mse = mse_df.groupby(["Method", f"Real_{letter}"]).agg([np.nanmean, np.nanstd])
    mse["nanstd"] = 0

    scatter_plot(mse, ax)

    ax.set_xlabel(f"${letter}$\n")

    if letter == "a":
        ax.set_ylabel("$\overline{(\hat{a}^{-1} - a^{-1})^2}$", rotation="vertical")
    else:
        ax.set_ylabel("$\overline{(\hat{b}^2 - b^2)^2}$", rotation="vertical")


def compute_inliers(df: pd.DataFrame):
    return (
        df.groupby(["Method", "Name"])
        .transform(
            lambda x: (
                x > x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))
            )
            & (x < (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
        )
        .eq(1)
    )


def plot_mse_image_dependence(df: pd.DataFrame, ax, letter):
    mse_df = df.groupby(["Method", "Name"]).apply(compute_mse, letter=letter)
    complete = mse_df.groupby(["Method", "Name"]).agg([np.nanmean, np.nanstd])
    inliers = (
        mse_df[compute_inliers(mse_df)]
        .groupby(["Method", "Name"])
        .agg([np.nanmean, np.nanstd])
    )

    complete["nanstd"] = 0
    inliers["nanstd"] = 0

    bar_plot(complete, ax)
    bar_plot(inliers, ax, color="black")

    ax.set_xlabel(f"Images\n")

    if letter == "a":
        ax.set_ylabel("$\overline{(\hat{a}^{-1} - a^{-1})^2}$", rotation="vertical")
    else:
        ax.set_ylabel("$\overline{(\hat{b}^2 - b^2)^2}$", rotation="vertical")


def plot_bias(df: pd.DataFrame, ax, letter):
    diff_df = df.groupby(["Method", f"Real_{letter}"]).apply(
        compute_difference, letter=letter
    )
    bias = diff_df.groupby(["Method", f"Real_{letter}"]).agg([np.nanmean, np.nanstd])
    bias["nanstd"] = 0

    scatter_plot(bias, ax)

    ax.set_xlabel(f"${letter}$\n")
    if letter == "a":
        ax.set_ylabel("$\overline{\hat{a}^{-1} - a^{-1}}$", rotation="vertical")
    else:
        ax.set_ylabel("$\overline{\hat{b}^2 - b^2}$", rotation="vertical")


def plot_ll(
    df: pd.DataFrame,
    ax,
    cmap,
    norm,
):
    a_array = np.sort(df.Real_a.unique())
    b_array = np.sort(df.Real_b.unique())
    X, Y = np.meshgrid(a_array, b_array)
    Z = df.pivot(index="Real_b", columns="Real_a").to_numpy()

    ax.pcolor(X, Y, Z, shading="auto", cmap=cmap, norm=norm)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


def compute_mse(df: pd.DataFrame, letter):
    if letter == "a":
        return np.square(1 / df[f"Est_{letter}"] - 1 / df[f"Real_{letter}"])
    elif letter == "b":
        return np.square(
            np.square(df[f"Est_{letter}"]) - np.square(df[f"Real_{letter}"])
        )


def compute_difference(df: pd.DataFrame, letter):
    if letter == "a":
        return 1 / df[f"Est_{letter}"] - 1 / df[f"Real_{letter}"]
    elif letter == "b":
        return np.square(df[f"Est_{letter}"]) - np.square(df[f"Real_{letter}"])


def compute_percentage(df: pd.DataFrame, letter):
    return 100 * df[df[f"Est_{letter}"] > 1e-8][f"Est_{letter}"].count() / len(df)


if __name__ == "__main__":
    main()
