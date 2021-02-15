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


def add_gaze_corrected(data, gaze_data):
    """
    Add gaze_corrected column per item,
    indicating the fraction of trial time
    that the item was looked at,
    after it was first seen in the trial:

    Args
    ---
        data (dataframe): aggregate response data
        gaze_data (dataframe): gaze data

    Returns
    ---
        copy of data
    """

    # read set sizes
    setsize = int(np.unique(data['setsize']))
    subject = int(np.unique(data['subject']))
    assert setsize == np.unique(gaze_data['setsize']), 'Set size does not match in data and gaze data'
    assert subject == np.unique(gaze_data['subject']), 'Subject does not match in data and gaze data'

    df = data.copy()
    gaze_df = gaze_data.copy()

    # insert blank gaze_corrected columns
    trials = np.unique(df['trial'])
    for item in range(setsize):
        df['gaze_corrected_{}'.format(item)] = 0

    # iterate trials
    for trial in trials:
        trial_df = df[df['trial']==trial].copy()
        trial_gaze_df = gaze_df[gaze_df['trial']==trial].copy()
        trial_rt = trial_df['rt'].values[0]
        # iterate items
        for item in range(setsize):
            # when item was first seen
            item_fix_onset = trial_df['fixation_onset_{}'.format(item)].values[0]
            if trial_rt > item_fix_onset:
                # how long it was seen
                item_seen_time = trial_rt - item_fix_onset
                # fraction looked at of item seen time
                item_fix_data = trial_gaze_df[trial_gaze_df['item']==item].copy()
                df.loc[df['trial']==trial, 'gaze_corrected_{}'.format(item)] = item_fix_data['dur'].sum() / item_seen_time

    return df


def format_data(data, gaze_data):
    """
    Extracts and formats data
    from dataframe to model friendly entities.

    Args
    ---
        data (dataframe): aggregate response data

    Returns
    ---
        dict with data entities
    """
    assert len(data['subject'].unique())==1, '/!\ too many subjects in data.'
    assert len(gaze_data['subject'].unique())==1, '/!\ too many subjects in gaze_data.'
    n_items = len([col for col in data.columns if col.startswith('item_value_')])
    assert len(data['setsize'].unique())==1, '/!\ too many set sizes in data.'
    assert len(gaze_data['setsize'].unique())==1, '/!\ too many set sizes in gaze_data.'
    assert data['setsize'].unique()[0]==n_items, '/!\ missmatch in setsizes in data.'
    assert gaze_data['setsize'].unique()[0]==n_items, '/!\ missmatch in setsizes in gaze_data.'

    # add gaze-corrected data columns
    data = add_gaze_corrected(data, gaze_data)

    # extract data
    gaze_corrected = data[['gaze_corrected_{}'.format(i) for i in range(n_items)]].values
    fixation_onset = data[['fixation_onset_{}'.format(i) for i in range(n_items)]].values
    values = data[['item_value_{}'.format(i) for i in range(n_items)]].values
    choice = data['choice'].values.astype(int)
    rts = data['rt'].values.astype(int)

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
    error_ll = 1. / np.float(n_items * (data['rt'].max() - data['rt'].min()))

    # package data
    output = dict(gaze_corrected=gaze_corrected,
                  fixation_onset=fixation_onset,
                  values=values_scaled,
                  rts=rts,
                  error_ll=error_ll,
                  n_seen=n_seen.astype(np.int32))
    return output