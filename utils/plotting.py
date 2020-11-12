#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import violinplot


def despine(ax):
    """
    Remove upper and right spines of matplotlib
    axis object.

    Input
    ---
    ax: Matplotlib axis object


    Returns
    ---
    Despined copy of ax
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax


def cm2inch(*tupl):
    """
    Convert a value or tuple of values from cm
    to inches.

    Source: https://stackoverflow.com/a/22787457

    Input
    ---
    tupl : float, int or tuple of arbitrary size
        Values to convert

    Returns
    ---
    Converted values in inches.
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def p_str(p):
    """
    Format p-value for plotting.
    """

    if (p <= 0.001):
        p = "%.2g" % p
    else:
        p = '{}'.format(np.round(p, 3))
    return 'p = {}'.format(p)


def t_str(T, df, P):
    """
    Generate string for plotting of t-test results:
    t(df) = p.

    Input
    ---
    T (float):
            t-value of t-test

    df (int):
            degrees of freedom of t-test

    P (float):
            p-value of t-test


    Returns
    ---
    formatted string
    """
    T = np.round(T, 2)
    if (P < 0.0001):
        return 't({}) = {},\np < 0.0001'.format(df, T)
    else:
        P = '{}'.format(np.round(P, 4))
        return 't({}) = {},\np = {}'.format(df, T, P)


def add_ttest(ax,
              T,
              df,
              P,
              X,
              Y,
              y_offset_bar,
              y_offset_text,
              fs=7):
    """
    Add t-test results to matplotlib axis object.
    T-test is indicated by horizontal bar,
    with result plotted above the bar.

    Input
    ---
    ax (matpltolib axis object)

    T (array):
            t-values of t-tests to plot

    df (array):
            degrees of freedoms of t-tests

    P (array):
            p-values of t-tests

    X (array):
            x-coordinates of t-test labeling

    Y (array):
            y-coordinates of t-test labeling

    y_offset_bar (float):
            offset of vertical bars indicating t-test

    y_offset_text (float):
            offset of text with respect to bar

    fs (float):
            fontsize of labeling


    Returns
    ---
    Matplotlib axis object with added t-test labels

    """

    for i, (x, y1) in enumerate(zip(X, Y)):
        ax.plot(x, y1, c='k')
        for xi, xx in enumerate(x):
            ax.plot((xx, xx),
                    [y1[xi]+y_offset_bar,
                     y1[xi]], c='k')
        s = t_str(T[i], df, P[i])
        ax.text(x[0], y1[1]+y_offset_text, s=s, fontsize=fs)

    return ax


def violin(
    data, value_name="value", violin_width=0.8, box_width=0.1, palette=None, ax=None
):
    """
    Forked from: github.com/moltaire/myplotlib

    Make a custom violinplot, with nice inner boxplot.
    Args:
        data (pandas.DataFrame): Data to plot. Each column will be made into one violin.
        violin_width (float, optional): Width of the violins. Defaults to 0.8.
        box_width (float, optional): Width of the boxplot. Defaults to 0.1.
        palette (list, optional): list of colors to use for violins. Defaults to default colors.
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to None.
    Returns:
        matplotlib.axis: Axis with the violinplot.
    """
    if ax is None:
        ax = plt.gca()

    # transform data into long format for seaborn violinplot
    if data.columns.name is None:
        data.columns.name = "variable"
    data_long = pd.melt(data, value_name=value_name)

    # Violinplot
    violinplot(
        x=data.columns.name,
        y=value_name,
        data=data_long,
        palette=palette,
        linewidth=0,
        inner=None,
        scale="width",
        width=violin_width,
        saturation=1,
        ax=ax,
    )

    # Boxplot
    # Matplotlib boxplot uses a different data format (list of arrays)
    boxplot_data = [data[var].values for var in data.columns]

    boxplotArtists = ax.boxplot(
        boxplot_data,
        positions=range(len(boxplot_data)),
        widths=box_width,
        showcaps=False,
        boxprops=dict(linewidth=0.5),
        medianprops=dict(linewidth=0.5, color="black"),
        whiskerprops=dict(linewidth=0.5),
        flierprops=dict(
            marker="o",
            markersize=2,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.25,
            alpha=0.9,
        ),
        manage_ticks=False,
        patch_artist=True,
    )
    for patch in boxplotArtists["boxes"]:
        patch.set_facecolor("white")

    # Adjust x-limits
    ax.set_xlim(-0.5, len(data.columns) + -0.5)

    return ax
