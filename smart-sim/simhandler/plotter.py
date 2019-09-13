"""
Universal function approximators

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from extrema_detector import *


# @brief plot3d_extrema: overlays extremum on dataset
# @param x: array like. Shape (n dim, # samples dim 0, ..., # samples dim n)
# @param f: array like. Shape (# samples dim 0, ..., # samples dim n)
# @param x_extremum: list. Points in x with extremum
# @param save_filepath=None: String. ex: "./3dextremum.pdf"
# @output: .pdf of figure at save_filepath if given
def plot3d_extrema(x, f, x_extremum, save_filepath=None):
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")

    e0 = [x[0][extremum] for extremum in x_extremum]
    e1 = [x[1][extremum] for extremum in x_extremum]
    f_extremum = [f[extremum] for extremum in x_extremum]

    ax.scatter3D(x[0], x[1], f.reshape(-1), cmap="Greens")
    ax.scatter3D(e0, e1, f_extremum, cmap="Reds", s=200, marker="x")
    ax.set_xlabel("$x_o$")
    ax.set_ylabel("$x_1$")
    ax.set_zlabel("f($x_0, x_1$;D)")
    ax.set_title("Detected Extremum in Dataset D")
    if save_filepath is not None and isinstance(save_filepath, str):
        plt.savefig(save_filepath)
    plt.show()


# @brief mgrid_heatplot_extrema: heatmap of extremum
# @param f: array like. Shape (# samples dim 0, ..., # samples dim n)
# @param x_extremum: list. Points in x with extremum
# @param save_filepath=None: String. ex: "./heatplot_extremum.pdf"
# @output: .pdf of figure at save_filepath if given
def mgrid_heatplot_extrema(f, x_extremum, save_filepath=None):
    if not mgrid_shape(x) and x.shape[0] == 2:
        raise UserWarning("Function only built for mgrid-type entries with 2 dimensions")

    f_extremum_overlay = np.zeros(f.shape)
    for i in x_extremum:
        f_extremum_overlay[i] = f[i]

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    plt.imshow(f_extremum_overlay, cmap="hot", interpolation="nearest")
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.title("Detected Extremum")

    ticks = range(0, len(x[1, 0]), int(len(x[1, 0])/10))
    val_ticks = [round(x[1, 0][i], 2) for i in ticks]
    plt.xticks(ticks, labels=val_ticks, rotation=90)
    plt.yticks(ticks, labels=val_ticks)

    if save_filepath is not None and isinstance(save_filepath, str):
        plt.savefig(save_filepath)
    plt.show()


def plot2d_extrema(x, f, x_extremum, save_filepath=None):
    '''
    Overlays extremum on dataset in 2D

    Parameters
    ----------
    x: array-like
        Shape (n dim, # samples dim 0, ..., # samples dim n)
    f: array-like
        Shape (# samples dim 0, ..., # samples dim n)
    x_extremum
        Points in x with extremum
    save_filepath=None: String
        ex: "./2dextremum.pdf"

    Returns
    -------
    "example.extension" of figure at save_filepath if given
    '''
    if len(x) != 1:
        raise UserWarning("Only plots lists with len(x)==2. Len(x) = ", len(x))

    print("x_extremum: ", x_extremum)
    plt.plot(x[0], f, "o", label="f(x;D)")
    plt.plot([x[0][e] for e in x_extremum], [f[e] for e in x_extremum], "x", markersize=15, label="extremum")
    plt.xlabel("$x$")
    plt.ylabel("$f(x;D)$")
    plt.title("Detected Extremum in Dataset D")
    plt.legend()
    if save_filepath is not None and isinstance(save_filepath, str):
        plt.savefig(save_filepath)
    plt.show()


# @brief plot2d_extrema: overlays extremum on dataset
# @param x: array like. Shape (n dim, # samples dim 0, ..., # samples dim n)
# @param f: array like. Shape (# samples dim 0, ..., # samples dim n)
# @param f_hat: array-like. Shape (# samples dim 0, ..., # samples dim n)
# @param save_filepath=None: String. ex: "./2dextremum.pdf"
# @output: .pdf of figure at save_filepath if given
def plot2d_approximation(x, f, f_hat, save_filepath=None):
    '''
    Overlays approximation on dataset in 2D

    Parameters
    ----------
    x: array-like
        Mgrid-like shape or (n dim, N samples)
    f: array-like
        Shape (n dim, N samples) or (N-samples dim 0, ..., N samples dim n)
    f_hat: array-like
        Shape (n dim, N samples) or (N-samples dim 0, ..., N samples dim n)
    save_filepath=None: String
        ex: "./2dextremum.pdf"

    Returns
    -------
    "example.extension" of figure at save_filepath if given
    '''
    plt.plot(x[0], f, "o", label="$f(x;D)$")
    plt.plot(x[0], f_hat, "x", label="$\hat{f}(x;D)$")
    plt.xlabel("$x$")
    plt.ylabel("$f$")
    plt.title("Approximation of Dataset $D$")
    plt.legend()
    if save_filepath is not None and isinstance(save_filepath, str):
        plt.savefig(save_filepath)
    plt.show()


# @brief plot3d_approximation: overlays approximaton on real dataset
# @param x: array like. Shape (n dim, # samples dim 0, ..., # samples dim n)
# @param f: array like. Shape (# samples dim 0, ..., # samples dim n)
# @param f_hat: array-like. Shape (# samples dim 0, ..., # samples dim n)
# @param save_filepath=None: String. ex: "./3dextremum.pdf"
# @output: .pdf of figure at save_filepath if given
def plot3d_approximation(x, f, f_hat, save_filepath=None):
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x[0], x[1], f.reshape(-1), cmap="Greens")
    ax.scatter3D(x[0], x[1], f_hat.reshape(-1),
                 cmap="Reds", marker="x")
    ax.set_xlabel("$x_o$")
    ax.set_ylabel("$x_1$")
    ax.set_zlabel("f($x_0, x_1$;D)")
    ax.set_title("Plotting Datasets of Real and Approximate Function")
    if save_filepath is not None and isinstance(save_filepath, str):
        plt.savefig(save_filepath)
    plt.show()


def pandas_line_error_approx_plot(df, savefig=""):
    """
    Plots the original and approximate PyPSA power flow results,
    along with error bars at the bottom

    Parameters
    ----------
    df: pandas dataframe
        Must contain "y" (original) and "y_hat" (approx) entries

    savefig: str
        ex: "my_plot.pdf"

    Returns
    -------
    None
        Plots the data and optionally saves the image
    """
    df["error"] = [abs(df["y"].values[i] - df["y_hat"].values[i]) for i in range(len(df))]

    ax = df[["y"]].plot(marker="o", color="r")
    df["y_hat"].plot(marker="o", markerfacecolor="none", ax=ax, label="$\hat{y}$", color='b')
    df["error"].plot(kind="bar", ax=ax, color="C7")
    plt.xticks(rotation=0)
    plt.grid(linestyle="--")
    plt.legend()
    plt.xlabel("Snapshot")
    plt.ylabel("Magnitude")
    plt.title("Plot of PyPSA Approximation Error Magnitude Per Snapshot")

    if isinstance(savefig, str) and savefig != "":
        plt.savefig(savefig)
    plt.show()
