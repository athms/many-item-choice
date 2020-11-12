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

# def plot_bar_panel(x,
#                    means,
#                    sems,
#                    x_label,
#                    y_label,
#                    fs=7,
#                    xticks=None,
#                    ticklabels=None,
#                    ticksize=7,
#                    facecolor=None,
#                    ylim=None,
#                    xlim=None,
#                    label=None,
#                    legend_loc='upper right',
#                    ax=None):
#     """
#     Create bar-plot from input data.
#
#     Input
#     ---
#     x (array):
#             x-index of bars
#
#     means (array):
#             means that define bar-heights
#
#     sems (array):
#             standard error of means
#
#     x_label / y_label (string):
#             axis-labeling
#
#     fs (int):
#             font-size of axis-labeling
#
#     x-ticks (array):
#             ticks of x-axis
#
#     ticklabels (array):
#             labeling of of x-ticks
#
#     ticksize (float):
#             font-size of tick-labeling
#
#     width (float):
#             width of bars
#
#     facecolor (string):
#             color of bars
#
#     ylim / xlim (array):
#             (upper, lower) lims for y/x-axis
#
#     label (string):
#             legend-label for plotted bars
#
#     legend-loc (string):
#             position of legend (upper / lower left / right)
#
#     ax (matplotlib axis):
#             matplotlib axis objects to plot bars on;
#             if ax = None, new axis is created
#
#
#     Returns
#     ---
#     Matpltolib axis object with bar-plot
#     """
#
#     tighten_layout = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=cm2inch(9, 9), dpi=330)
#         tighten_layout = True
#
#     # plot bar
#     if facecolor is None:
#         facecolor = 'white'
#         edgecolor = 'black'
#     else:
#         edgecolor = None
#     ax.bar(x,
#            means,
#            width=(x[1]-x[0]) / 2,
#            color=facecolor,
#            edgecolor=edgecolor,
#            linewidth=1.5,
#            label=label)
#     ax.errorbar(x, means,
#                 yerr=sems,
#                 color='black',
#                 ls='none')
#
#     # label panel
#     ax.set_xlabel(x_label, fontsize=fs)
#     ax.set_ylabel(y_label, fontsize=fs)
#
#     if xticks is not None:
#         ax.set_xticks(xticks)
#
#     if ticklabels is not None:
#         ax.set_xticklabels(ticklabels, fontsize=ticksize)
#
#     if ylim is not None:
#         ax.set_ylim(ylim)
#
#     if xlim is not None:
#         ax.set_xlim(ylim)
#     else:
#         ax.set_xlim(x[0]-1, x[-1]+1)
#
#     if label is not None:
#         ax.legend(loc=legend_loc, frameon=False)
#
#     ax.tick_params(axis='both', which='major', labelsize=ticksize)
#     despine(ax)
#     if tighten_layout:
#         fig.tight_layout()
#
#     return ax
#
#
# def plot_line_panel(x,
#                     means,
#                     sems,
#                     x_label,
#                     y_label,
#                     fs=7,
#                     xticks=None,
#                     ticklabels=None,
#                     ticksize=7,
#                     color='k',
#                     ylim=None,
#                     xlim=None,
#                     label=None,
#                     lw=2,
#                     ls='-',
#                     alpha=.1,
#                     legend_loc='upper right',
#                     ax=None):
#     """
#     Create line-plot from input data.
#
#     Input
#     ---
#     x (array):
#             x-index of line-means
#
#     means (array):
#             means that define line
#
#     sems (array):
#             standard error of means
#
#     x_label / y_label (string):
#             axis-labeling
#
#     fs (int):
#             font-size of axis-labeling
#
#     x-ticks (array):
#             ticks of x-axis
#
#     ticklabels (array):
#             labeling of of x-ticks
#
#     ticksize (float):
#             font-size of tick-labeling
#
#     color (string):
#             color of line
#
#     ylim / xlim (array):
#             (upper, lower) lims for y/x-axis
#
#     label (string):
#             legend-label for plotted line
#
#     lw (float):
#             width of line
#
#     alpha (float):
#             transparency of line
#
#     legend-loc (string):
#             position of legend (upper / lower left / right)
#
#     ax (matplotlib axis):
#             matplotlib axis objects to plot bars on;
#             if ax = None, new axis is created
#
#
#     Returns
#     ---
#     Matpltolib axis object with line-plot
#
#     """
#
#     tighten_layout = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(3, 5))
#         tighten_layout = True
#
#     # plot bar
#     ax.plot(x,
#             means,
#             color=color,
#             ls=ls,
#             lw=lw,
#             label=label)
#
#     ax.fill_between(x,
#                     means-sems,
#                     means+sems,
#                     color=color,
#                     alpha=alpha)
#
#     # label panel
#     ax.set_xlabel(x_label, fontsize=fs)
#     ax.set_ylabel(y_label, fontsize=fs)
#
#     if xticks is not None:
#         ax.set_xticks(xticks)
#
#     if ticklabels is not None:
#         ax.set_xticklabels(ticklabels, fontsize=ticksize)
#
#     if ylim is not None:
#         ax.set_ylim(ylim)
#
#     if xlim is not None:
#         ax.set_xlim(xlim)
#
#     if label is not None:
#         ax.legend(loc=legend_loc, frameon=False, fontsize=fs)
#
#     ax.tick_params(axis='both', which='major', labelsize=ticksize)
#     despine(ax)
#     if tighten_layout:
#         fig.tight_layout()
#
#     return ax
#
#
# def add_regression_line(x,y,ax,
#                         annotate=True,
#                         annotation_pos=(0.1, 0.1),
#                         annotation_halign='left',
#                         fontsize_annotation=7,
#                         color='C0',
#                         color_annotation=None):
#
#     if color_annotation == None:
#         color_annotation = color
#     # LM fit
#     X = sm.add_constant(x)
#     lm = sm.OLS(y, X).fit()
#     intercept, slope = lm.params
#     table, data, columns = summary_table(lm, alpha=1. - 0.95)
#     predicted, mean_ci_lower, mean_ci_upper = data[:,np.array([2, 4, 5])].T
#     xs = np.linspace(*ax.get_xlim(), 100)
#     line = ax.plot(xs, intercept + slope * xs, color=color)
#     sort_idx = np.argsort(x)
#     ax.fill_between(x[sort_idx],
#                     mean_ci_lower[sort_idx],
#                     mean_ci_upper[sort_idx],
#                     color=color,
#                     alpha=0.1)
#     # Annotation
#     tval = lm.tvalues[-1]
#     pval = lm.pvalues[-1]
#     if pval < 0.0001:
#         p_string = r'$P < 0.0001$'
#     else:
#         p_string = r'$P = {}$'.format(np.round(pval, 4))
#     r = np.sign(tval) * np.sqrt(lm.rsquared)
#     annotation = (r'$r = {:.2f}$, '.format(r)) + p_string
#     if annotate:
#         ax.text(*annotation_pos,
#                 annotation,
#                 verticalalignment='bottom',
#                 horizontalalignment=annotation_halign,
#                 transform=ax.transAxes,
#                 fontsize=fontsize_annotation,
#                 color=color_annotation)
#     return ax




# def plot_individual_fit(observed,
#                         predictions,
#                         prediction_labels=None,
#                         colors=None,
#                         fontsize=7,
#                         alpha=1.0,
#                         markers=None,
#                         figsize=None,
#                         limits={
#                                'p_choose_best': (0, 1),
#                                'rt': (0, None),
#                                'gaze_influence': (None, None)
#                            },
#                         axs=None,
#                         regression=False):
#     """
#     Plot individual observed vs predicted data
#     on three metrics:
#     A) response time
#     B) p(choose best)
#     C) gaze influence score
#     For details on these measures,
#     see the manuscript
#
#     Parameters
#     ----------
#     observed : pandas.DataFrame
#         observed response data
#
#     predictions : list of pandas.DataFrame
#         predicted response datasets
#
#     prediction_labels : array_like, strings, optional
#         legend labels for predictions
#
#     colors : array_like, strings, optional
#         colors to use for predictions
#
#     fontsize : int, optional
#         plotting fontsize
#
#     alpha : float, optional
#         alpha level for predictions
#         should be between [0,1]
#
#     figsize : tuple, optional
#         matplotlib figure size
#
#     limits : dict, optional
#         dict containing one entry for:
#         ['rt', 'p_choose_best', 'corrected_p_choose_best']
#         each entry is a tuple, defining the y-limits for
#         the respective metrics
#
#     Returns
#     ---
#     Tuple
#         matplotlib figure object, axs
#     """
#
#     # count number of predictions
#     n_predictions = len(predictions)
#     # define prediction labels
#     if prediction_labels is None:
#         prediction_labels = [
#             'Prediction {}'.format(i + 1) for i in range(n_predictions)
#         ]
#
#     # define figre
#     if figsize is None:
#         figsize = cm2inch(18, 6)
#     if axs is None:
#         fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=330)
#         axs_created = True
#     else:
#         axs_created = False
#     if markers is None:
#         markers =['o' for i in range(len(predictions))]
#     if colors is None:
#         colors =['C{}'.format(i) for i in range(len(predictions))]
#
#     # add default limits
#     for key, lim in zip(['p_choose_best', 'rt', 'gaze_influence'],
#                         [(0,1), (0,None), (None,None)]):
#         if key not in limits.keys():
#             limits[key] = lim
#
#     # create subject summary for observed
#     n_items = np.int(len([c for c in observed.columns if 'item_value_' in c]))
#     for i, prediction in enumerate(predictions):
#         n_items_pred = np.int(
#             len([c for c in prediction.columns if 'item_value_' in c]))
#         if n_items != n_items_pred:
#             raise ValueError(
#                 'observed and prediction {} contain unequal number of items'.
#                 format(i))
#     observed_subject_summary = aggregate_subject_level_data(observed,
#                                                             n_items=n_items)
#
#     # extract oberved value ranges
#     rt_range = extract_range(observed_subject_summary['rt']['mean'],
#                              bound=limits['rt'])
#     best_chosen_range = extract_range(
#         observed_subject_summary['best_chosen']['mean'], bound=limits['p_choose_best'])
#     gaze_influence_range = extract_range(
#         observed_subject_summary['gaze_influence'], bound=limits['gaze_influence'])
#
#     # plot observed vs predicted
#     for m, prediction in enumerate(predictions):
#
#         # create subject summary for prediction
#         prediction_subject_summary = aggregate_subject_level_data(
#             prediction, n_items=n_items)
#
#         # a) Mean RT
#         axs[0].scatter(observed_subject_summary['rt']['mean'],
#                        prediction_subject_summary['rt']['mean'],
#                        marker=markers[m],
#                        color=colors[m],
#                        linewidth=1,
#                        alpha=alpha,
#                        label=prediction_labels[m],
#                        s=30)
#
#         if regression:
#             add_regression_line(x=observed_subject_summary['rt']['mean'],
#                                 y=prediction_subject_summary['rt']['mean'],
#                                 ax=axs[0],
#                                 color=colors[m],
#                                 annotation_pos=(0.1,0.9-(0.1*m)))
#
#
#         # b) P(choose best)
#         axs[1].scatter(observed_subject_summary['best_chosen']['mean'],
#                        prediction_subject_summary['best_chosen']['mean'],
#                        marker=markers[m],
#                        color=colors[m],
#                        linewidth=1,
#                        alpha=alpha,
#                        s=30)
#
#         if regression:
#             add_regression_line(x=observed_subject_summary['best_chosen']['mean'],
#                                 y=prediction_subject_summary['best_chosen']['mean'],
#                                 ax=axs[1],
#                                 color=colors[m],
#                                 annotation_pos=(0.1,0.9-(0.1*m)))
#
#         # c) Gaze Influence
#         axs[2].scatter(observed_subject_summary['gaze_influence'],
#                        prediction_subject_summary['gaze_influence'],
#                        marker=markers[m],
#                        color=colors[m],
#                        linewidth=1,
#                        alpha=alpha,
#                        s=30)
#
#         if regression:
#             add_regression_line(x=observed_subject_summary['gaze_influence'],
#                                 y=prediction_subject_summary['gaze_influence'],
#                                 ax=axs[2],
#                                 color=colors[m],
#                                 annotation_pos=(0.1,0.9-(0.1*m)))
#
#         # update parameter ranges
#         rt_range_prediction = extract_range(
#             prediction_subject_summary['rt']['mean'], bound=limits['rt'])
#         if rt_range[0] > rt_range_prediction[0]:
#             rt_range[0] = rt_range_prediction[0]
#         if rt_range[1] < rt_range_prediction[1]:
#             rt_range[1] = rt_range_prediction[1]
#
#         best_chosen_range_prediction = extract_range(
#             prediction_subject_summary['best_chosen']['mean'], bound=limits['p_choose_best'])
#         if best_chosen_range[0] > best_chosen_range_prediction[0]:
#             best_chosen_range[0] = best_chosen_range_prediction[0]
#         if best_chosen_range[1] < best_chosen_range_prediction[1]:
#             best_chosen_range[1] = best_chosen_range_prediction[1]
#
#         gaze_influence_range_prediction = extract_range(
#             prediction_subject_summary['gaze_influence'], bound=limits['gaze_influence'])
#         if gaze_influence_range[0] > gaze_influence_range_prediction[0]:
#             gaze_influence_range[0] = gaze_influence_range_prediction[0]
#         if gaze_influence_range[1] < gaze_influence_range_prediction[1]:
#             gaze_influence_range[1] = gaze_influence_range_prediction[1]
#
#         # label axes
#         axs[0].set_ylabel('Predicted Mean RT (ms)'.format(
#             prediction_labels[m]),
#                              fontsize=fontsize)
#         axs[0].set_xlabel('Observed\nMean RT (ms)', fontsize=fontsize)
#         axs[1].set_ylabel('Predicted\nP(choose best item)', fontsize=fontsize)
#         axs[1].set_xlabel('Observed\nP(choose best item)', fontsize=fontsize)
#         axs[2].set_ylabel('Predicted\nGaze Influence on P(choose item)',
#                              fontsize=fontsize)
#         axs[2].set_xlabel('Observed\nGaze Influence on P(choose item)',
#                              fontsize=fontsize)
#
#     # update axes limits and ticks
#     if (rt_range[1] - rt_range[0]) > 3:
#         rt_tickstep = 1500
#     else:
#         rt_tickstep = 750
#     rt_ticks = np.arange(rt_range[0], rt_range[1] + rt_tickstep,
#                          rt_tickstep)
#     axs[0].set_yticks(rt_ticks[::2])
#     axs[0].set_xticks(rt_ticks[::2])
#     axs[0].set_xlim(rt_range)
#     axs[0].set_ylim(rt_range)
#     lgnd = axs[0].legend(loc='upper left', frameon=False, fontsize=fontsize-1)
#     for lh in lgnd.legendHandles:
#         lh.set_alpha(1)
#
#     best_chosen_ticks = np.arange(0,1.1,0.2)
#     axs[1].set_yticks(best_chosen_ticks)
#     axs[1].set_xticks(best_chosen_ticks)
#     axs[1].set_xlim(best_chosen_range)
#     axs[1].set_ylim(best_chosen_range)
#
#     gaze_influence_ticks = np.arange(-1,1.1,0.2)
#     axs[2].set_yticks(gaze_influence_ticks)
#     axs[2].set_xticks(gaze_influence_ticks)
#     axs[2].set_xlim(gaze_influence_range)
#     axs[2].set_ylim(gaze_influence_range)
#
#     # label panels
#     for label, ax in zip(list('ABCDEF'), axs.ravel()):
#         if axs_created:
#             ax.text(-0.4,
#                     1.1,
#                     label,
#                     transform=ax.transAxes,
#                     fontsize=fontsize,
#                     fontweight='bold',
#                     va='top')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.tick_params(axis='both', labelsize=fontsize)
#         # plot diagonal
#         ax.plot(ax.get_xlim(),
#                 ax.get_xlim(),
#                 linewidth=1,
#                 color='black',
#                 alpha=1.0,
#                 zorder=-1)
#
#     if axs_created:
#         fig.tight_layout()
#
#     return axs
#
#
#
# def add_gaze_advantage(df, bins=7, return_bins=False):
#     """
#     Add gaze advantage (defined as the difference
#     between an item's gaze and the maximum gaze
#     of all other) to response data
#
#     Input
#     ---
#     df : dataframe
#         response data
#
#     bins : int or array_like, optional
#         defining the bins to use when computing
#         the gaze difference,
#         if an int is given, this many bins will be
#         created,
#         defaults to 7
#
#     return_bins : bool, optional
#         whether or not to return the bins
#
#     Returns
#     ---
#     copy of df (and bins if return_bins=True)
#     """
#
#     # infer number of items
#     gaze_cols = ([col for col in df.columns if col.startswith('gaze_')])
#     n_items = len(gaze_cols)
#
#     gaze = df[gaze_cols].values
#     gaze_advantage = np.zeros_like(gaze)
#     for t in np.arange(gaze.shape[0]):
#         for i in range(n_items):
#             gaze_advantage[t, i] = gaze[t, i] - \
#                 np.mean(gaze[t, np.arange(n_items) != i])
#
#     if isinstance(bins, (int, float)):
#         bins = np.round(np.linspace(-1, 1, bins), 2)
#
#     for i in range(n_items):
#         df['gaze_advantage_{}'.format(i)] = gaze_advantage[:, i]
#         gaze_bins = pd.cut(df['gaze_advantage_{}'.format(i)].values, bins, True)
#         df['gaze_advantage_binned_{}'.format(i)] = bins[gaze_bins.codes]
#     if not return_bins:
#         return df.copy()
#     else:
#         return df.copy(), bins


#
# def compute_corrected_choice(df):
#     """
#     Compute and add corrected choice probability
#     to response data; (see manuscript for details)
#
#     Input
#     ---
#     df : dataframe
#         response data
#
#     Returns
#     ---
#     Copy of df, including corrected_choice column
#     """
#
#     # recode choice
#     n_items = len([ c for c in df.columns if c.startswith('item_value_')])
#     is_choice = np.zeros((df.shape[0], n_items))
#     is_choice[np.arange(is_choice.shape[0]), df['choice'].values.astype(np.int)] = 1
#
#     if n_items > 2:
#         values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
#         value_range_others = np.zeros_like(is_choice)
#         for t in range(value_range_others.shape[0]):
#             for i in range(n_items):
#                 value_range_others[t, i] = values[t, np.arange(n_items) != i].max(
#                     ) - values[t, np.arange(n_items) != i].min()
#     # relative value
#     df = add_value_minus_mean_others(df)
#     relative_values = df[[
#         'value_minus_mean_others_{}'.format(i) for i in range(n_items)
#     ]].values
#
#     df_tmp = pd.DataFrame({
#         "subject": np.repeat(df['subject'].values, n_items),
#         "relative_value": relative_values.ravel(),
#         "is_choice": is_choice.ravel()
#     })
#     if n_items > 2:
#         df_tmp['value_range_others'] = value_range_others.ravel()
#
#     # place in dataframe
#     data_out = []
#     for s, subject in enumerate(df['subject'].unique()):
#         subject_data_tmp = df_tmp[df_tmp['subject'] == subject].copy()
#         if n_items > 2:
#             X = subject_data_tmp[['relative_value', 'value_range_others']]
#             X = sm.add_constant(X)
#             y = subject_data_tmp['is_choice']
#         else:
#             X = subject_data_tmp[['relative_value']]
#             # exclude every second entry, bc 2-item case is symmetrical
#             X = sm.add_constant(X)[::2]
#             y = subject_data_tmp['is_choice'].values[::2]
#         try:
#             logit = sm.Logit(y, X)
#             result = logit.fit(disp=0)
#         except:
#             print('/!\ Dropping "value_range_others" variable while computing logit fit.')
#             X = X.drop('value_range_others', axis=1)
#             logit = sm.Logit(y, X)
#             result = logit.fit(disp=0)
#         predicted_pchoice = result.predict(X)
#
#         subject_data_tmp['corrected_choice'] = (subject_data_tmp['is_choice'] -
#                                                 predicted_pchoice)
#         data_out.append(subject_data_tmp)
#
#     data_out = pd.concat(data_out)
#
#     return data_out.copy()



#
# def extract_range(x, extra=0.25, bound=(None, None)):
#     """
#     Extract range of x-data
#
#     Input
#     ---
#     x : array_like
#         x-data
#
#     extra : float, optional
#         should be between [0,1],
#         defining percentage of x-mean to add / subtract
#         to min / max, when copmuting bounds
#         e.g. upper bound = np.max(x) + extra * np.mean(x)
#
#     bound : tuple, optional
#         if given, these bounds are used
#
#     Returns
#     ---
#     tuple of bounds
#     """
#
#     if bound[0] != None:
#         xmin = bound[0]
#     else:
#         xmean = np.mean(x)
#         xmin = np.min(x) - extra * xmean
#
#     if bound[1] != None:
#         xmax = bound[1]
#     else:
#         xmean = np.mean(x)
#         xmax = np.max(x) + extra * xmean
#
#     return [xmin, xmax]

#
# def plot_correlation(x,
#                      y,
#                      xlabel='',
#                      ylabel='',
#                      title='',
#                      ci=0.95,
#                      alpha=0.5,
#                      size=30,
#                      color='red',
#                      markercolor='black',
#                      marker='o',
#                      xticks=None,
#                      yticks=None,
#                      xticklabels=None,
#                      yticklabels=None,
#                      xlim=None,
#                      ylim=None,
#                      annotate=True,
#                      annotation_pos=(0.1, 0.1),
#                      annotation_halign='left',
#                      fontsize_title=7,
#                      fontsize_axeslabel=7,
#                      fontsize_ticklabels=7,
#                      fontsize_annotation=7,
#                      regression=True,
#                      plot_diagonal=False,
#                      return_correlation=False,
#                      label=None,
#                      ax=None):
#     """
#     Plot correlation between x and y;
#     (scatter-plot and regression line)
#     """
#
#     # Defaults
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # Axes, ticks, ...
#     if xticks is not None:
#         ax.set_xticks(xticks)
#     if yticks is not None:
#         ax.set_yticks(yticks)
#
#     if xticklabels is not None:
#         ax.set_xticklabels(xticklabels, fontsize=fontsize_ticklabels)
#     if yticklabels is not None:
#         ax.set_yticklabels(yticklabels, fontsize=fontsize_ticklabels)
#
#     ax.tick_params(axis='both', which='major', labelsize=fontsize_ticklabels)
#
#     if xlim is not None:
#         ax.set_xlim(xlim)
#     if ylim is not None:
#         ax.set_ylim(ylim)
#
#     # Scatter (translucent dots with solid outlines)
#     ax.scatter(x,
#                y,
#                marker='o',
#                color='none',
#                edgecolor=markercolor,
#                linewidth=1,
#                s=size)
#     ax.scatter(x,
#                y,
#                marker='o',
#                color=markercolor,
#                alpha=alpha,
#                linewidth=0,
#                s=size,
#                label=label)
#
#     if regression:
#         # LM fit
#         X = sm.add_constant(x)
#         lm = sm.OLS(y, X).fit()
#         intercept, slope = lm.params
#         table, data, columns = summary_table(lm, alpha=1. - ci)
#         predicted, mean_ci_lower, mean_ci_upper = data[:,
#                                                        np.array([2, 4, 5])].T
#
#         xs = np.linspace(*ax.get_xlim(), 100)
#         line = ax.plot(xs, intercept + slope * xs, color=color)
#         sort_idx = np.argsort(x)
#         ax.fill_between(x[sort_idx],
#                         mean_ci_lower[sort_idx],
#                         mean_ci_upper[sort_idx],
#                         color=color,
#                         alpha=0.1)
#
#         # Annotation
#         tval = lm.tvalues[-1]
#         pval = lm.pvalues[-1]
#         if pval < 0.0001:
#             p_string = r'$P < 0.0001$'
#         else:
#             p_string = r'$P = {}$'.format(np.round(pval, 4))
#         r = np.sign(tval) * np.sqrt(lm.rsquared)
#         annotation = (r'$r = {:.2f}$, '.format(r)) + p_string
#         if annotate:
#             ax.text(*annotation_pos,
#                     annotation,
#                     verticalalignment='bottom',
#                     horizontalalignment=annotation_halign,
#                     transform=ax.transAxes,
#                     fontsize=fontsize_annotation)
#
#     # Diagonal
#     if plot_diagonal:
#         ax.plot([0, 1], [0, 1],
#                 transform=ax.transAxes,
#                 color='black',
#                 alpha=0.5,
#                 zorder=-10,
#                 lw=1)
#
#     # Labels
#     ax.set_xlabel(xlabel, fontsize=fontsize_axeslabel)
#     ax.set_ylabel(ylabel, fontsize=fontsize_axeslabel)
#     ax.set_title(title, fontsize=fontsize_title)
#
#     sns.despine(ax=ax)
#
#     if return_correlation:
#         return ax, line, annotation
#     else:
#         return ax

#
# def plot_posterior(samples,
#                    kind='hist',
#                    ref_val=None,
#                    precision=2,
#                    alpha=0.05,
#                    bins=20,
#                    burn=0,
#                    ax=None,
#                    fontsize=7,
#                    color='skyblue'):
#     """
#     Arviz is broken, so we do it ourselves.
#
#     Input:
#         samples (TYPE): Description
#         kind (str, optional): Description
#         ref_val (None, optional): Description
#         precision (int, optional): Description
#         alpha (float, optional): Description
#         burn (int, optional): Description
#         ax (None, optional): Description
#
#     Returns:
#         TYPE: Description
#
#     Raises:
#         ValueError: Description
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     if kind == 'hist':
#         ax.hist(samples, color=color, bins=bins)
#     elif kind == 'kde':
#         sns.kdeplot(samples, color=color, ax=ax)
#     else:
#         raise ValueError("'kind' should be 'hist' or 'kde'.")
#
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#
#     # Central tendency
#     ax.text(x=np.mean(xlim),
#             y=(ylim[1] - ylim[0]) * 0.95,
#             s='mean = {}'.format(np.round(samples.mean(), precision)),
#             ha='center',
#             va='center',
#             fontsize=fontsize)
#
#     # HPD
#     hpdvals = hpd(samples, alpha=alpha)
#     ax.fill_between(hpdvals, y1=[0, 0], y2=2 * [(ylim[1] - ylim[0]) * 0.1],
#                     color='black', edgecolor='none', lw=0,
#                     alpha=0.5, zorder=2)
#     # ax.text(x=np.mean(hpdvals),
#     #         y=(ylim[1] - ylim[0]) * 0.1,
#     #         s='{:.0f}% HPD'.format(100 * (1 - alpha)),
#     #         ha='center',
#     #         va='center',
#     #         fontweight='bold',
#     #         fontsize=fontsize)
#     for val in hpdvals:
#         ax.text(x=val,
#                 y=(ylim[1] - ylim[0]) * 0.2,
#                 s='{}'.format(np.round(val, precision)),
#                 ha='center',
#                 va='center',
#                 fontsize=fontsize)
#
#     # Reference Value
#     if ref_val is not None:
#         ax.axvline(ref_val, color='crimson', linewidth=1, alpha=0.5)
#         less = 100 * np.mean(samples < ref_val)
#         more = 100 * np.mean(samples > ref_val)
#         ax.text(x=np.mean(xlim),
#                 y=(ylim[1] - ylim[0]) * 0.5,
#                 s='{:.2f}% < {} < {:.2f}%'.format(less, ref_val, more),
#                 ha='center',
#                 va='center',
#                 fontweight='bold',
#                 color='crimson',
#                 fontsize=fontsize)
#
#     ax.set_xlabel('Sample value', fontsize=fontsize)
#     ax.set_yticks([])
#     ax.tick_params(axis='both', which='major', labelsize=fontsize)
#     sns.despine(ax=ax, left=True, top=True, right=True)
#
#     return ax
#
#
# def traceplot(trace, varnames='all', combine_chains=False,
#               ref_val={}):
#     """A traceplot replacement, because arviz is broken.
#     This is tested for traces that come out of individual
#     and hierarchical GLAM fits.
#
#     Args:
#         trace (PyMC.MultiTrace): A trace object.
#         varnames (str, optional): List of variables to include
#         combine_chains (bool, optional): Toggle concatenation of chains.
#         ref_val (dict, optional): Reference values per parameter.
#
#     Returns:
#         figure, axes
#     """
#     if varnames == 'all':
#         varnames = [var for var in trace.varnames
#                     if not var.endswith('__')]
#     nvars = len(varnames)
#     if combine_chains:
#         nchains = 1
#     else:
#         nchains = trace.nchains
#
#     fig, axs = plt.subplots(nvars, 2, figsize=(8, nvars * 2))
#
#     for v, var in enumerate(varnames):
#
#         samples = trace.get_values(var, combine=combine_chains)
#         if not isinstance(samples, list):
#             samples = [samples]
#
#         for chain in range(nchains):
#             # group level parameters are (nsamples)
#             # but subject level parameters are (nsamples x nsubjects x nconditions)
#             chain_samples = samples[chain]
#             if chain_samples.ndim == 1:
#                 chain_samples = chain_samples[:, None, None]
#             elif chain_samples.ndim == 2:
#                 nsamples, nsubjects = chain_samples.shape
#             elif chain_samples.ndim == 3:
#                 nsamples, nsubjects, nconditions = chain_samples.shape
#             else:
#                 print('Too many dimensions of chain_samples.')
#                 assert False
#
#             # Trace
#             axs[v, 0].set_xlabel('')
#             axs[v, 0].set_ylabel('Sample value')
#             axs[v, 0].set_title(var)
#             axs[v, 0].plot(chain_samples.ravel(),
#                            alpha=0.3)
#
#             # KDE
#             sns.kdeplot(chain_samples.ravel(),
#                         ax=axs[v, 1])
#             axs[v, 1].set_title(var)
#             axs[v, 1].set_ylabel('Frequency')
#
#         # Reference Values
#         if ref_val.get(var, False) is not False:
#             axs[v, 0].axhline(ref_val.get(var),
#                               color='black', linewidth=2)
#             axs[v, 1].axvline(ref_val.get(var),
#                               color='black', linewidth=2)
#
#     fig.tight_layout()
#     return fig, axs
