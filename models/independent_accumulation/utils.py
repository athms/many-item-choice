#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import mode


def sort_according_to_choice(x, choices):
    """
    Sort an array x according to choice indices.

    Args
    ---
        x (ndarray): 2-d array to be sorted along
                dimension 1 so that choices
                always at index 0

    Returns
    ---
        Sorted copy of x.
    """
    x_sorted = np.zeros_like(x) * np.nan
    # Get values of chosen entries, and put them into first column
    x_sorted[:, 0] = x[np.arange(x.shape[0]), choices.astype(int)]
    # and everything else into the next columns
    others = np.vstack([x[i, np.where(np.arange(x.shape[1]) != c)]
                        for i, c in enumerate(choices)])
    x_sorted[:, 1:] = others
    return x_sorted


def sort_according_to_seen(x, gaze):
    """
    """
    x_sorted = np.zeros_like(x) * np.nan
    for i in range(x.shape[0]):
        idx = gaze[i]>0
        x_sorted[i,:np.sum(idx)] = x[i,idx]
    return x_sorted


def format_data(df):
    """
    Extracts and formats data
    from dataframe to model friendly entities.

    Args
    ---
        df (dataframe): aggregate response data

    Returns
    ---
        dict with data entities
    """
    assert len(df['subject'].unique())==1, '/!\ too many subjects in data.'
    n_items = len([col for col in df.columns
                   if col.startswith('item_value_')])
    assert len(df['setsize'].unique())==1, '/!\ too many set sizes in data.'
    assert df['setsize'].unique()[0]==n_items, '/!\ missmatch in setsizes in data.'

    # extract data
    gaze_corrected = df[['gaze_corrected_{}'.format(i) for i in range(n_items)]].values
    fixation_onset = df[['fixation_onset_{}'.format(i) for i in range(n_items)]].values
    values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
    choice = df['choice'].values.astype(int)
    rts = df['rt'].values.astype(int)

    # sort so that choices are in the firts column
    gaze_corrected = sort_according_to_choice(gaze_corrected, choice)
    fixation_onset = sort_according_to_choice(fixation_onset, choice)
    values = sort_according_to_choice(values, choice)

    # sort according to seen
    gaze_corrected = sort_according_to_seen(gaze_corrected, gaze_corrected)
    fixation_onset = sort_according_to_seen(fixation_onset, gaze_corrected)
    values = sort_according_to_seen(values, gaze_corrected)
    n_seen = np.sum(np.isfinite(gaze_corrected), axis=1)
    # re-scale values to 1 - 7
    values_scaled = values + 4

    # compute random choice likelihood
    error_ll = 1. / np.float(n_items * (df['rt'].max() - df['rt'].min()))

    # package data
    output = dict(gaze_corrected=gaze_corrected,
                  fixation_onset=fixation_onset,
                  values=values_scaled,
                  rts=rts,
                  error_ll=error_ll,
                  n_seen=n_seen.astype(np.int32))
    return output


#
# def extract_modes(traces, parameters=None, precision=None, f_burn=0.5):
#     """
#     Extract modesl from PyMC3 traces.
#
#     Input
#     ---
#     traces (PyMC3 traces):
#             single trace of list of traces
#             from which modes to extract
#
#     parameters (array):
#             names of parameters for which
#             to extract modes
#
#     precision (array):
#             decimal precision to round
#             trace values to prior to mode
#             extraction
#
#     f_burn (float):
#             fraction of trace to discard at
#             beginning prior to extracting
#             modes
#
#     Returns
#     ---
#     dict(s) indicating parameter modes
#     """
#
#     if not isinstance(traces, list):
#         traces = [traces]
#
#     modes = []
#
#     for trace in traces:
#
#         if parameters is None:
#             parameters = [var for var in trace.varnames
#                           if not var.endswith('__')]
#
#             print('/!\ Automatically setting parameter precision...')
#             precision_defaults = dict(v=6, gamma=2, tau=2, s=6, SNR=2,  t0=-1)
#             precision = [precision_defaults.get(parameter.split('_')[0], 6)
#                          for parameter in parameters]
#
#         n_samples = len(trace)
#         trace_modes = {}
#
#         for parameter, prec in zip(parameters, precision):
#             trace_modes[parameter] = mode(np.round(trace.get_values(
#                 parameter, burn=int(f_burn*n_samples)), prec))[0][0]
#         modes.append(trace_modes)
#
#     if len(modes) == 1:
#         return modes[0]
#     else:
#         return modes
