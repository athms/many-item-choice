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
    subjects = df['subject'].unique()
    n_subjects = len(subjects)
    n_items = len([col for col in df.columns
                   if col.startswith('item_value_')])
    subject_idx = np.array(df.subject.values.astype(int))
    setsizes = np.sort(df['setsize'].unique().astype(np.int))
    n_setsizes = setsizes.size
    setsize_idx = np.zeros(subject_idx.size).astype(int)
    for si, setsize in enumerate(setsizes):
        setsize_idx[np.where(df['setsize'] == setsize)] = si

    gaze = df[['cumulative_gaze_{}'.format(i) for i in range(n_items)]].values
    values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
    values[gaze==0] = 0 # exclude values of items that were not seen
    choice = df['choice'].values.astype(int)
    rts = df['rt'].values
    rts = rts.astype('int')

    # sort so that choices are in the firts column
    gaze = sort_according_to_choice(gaze, choice)
    values = sort_according_to_choice(values, choice)
    # re-scale values to 1 - 7
    values_scaled = values + 4

    # compute random choice likelihood
    error_lls = np.zeros((n_subjects, n_setsizes))
    for si, setsize in enumerate(setsizes):
        setsize_df = df[df['setsize'] == setsize].copy()
        for s, subject in enumerate(setsize_df['subject'].unique()):
            subject_df = setsize_df[setsize_df['subject'] == subject]
            error_lls[s, si] = 1. / \
                np.float(
                    setsize * (subject_df['rt'].max() - subject_df['rt'].min()))

    output = dict(subjects=subjects,
                  n_subjects=n_subjects,
                  n_items=n_items,
                  subject_idx=subject_idx,
                  setsize_idx=setsize_idx,
                  gaze=gaze,
                  values=values_scaled,
                  rts=rts,
                  error_lls=error_lls)
    return output
