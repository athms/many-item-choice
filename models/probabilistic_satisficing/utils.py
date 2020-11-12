#!/usr/bin/env python3

import numpy as np
import pandas as pd


def format_data(data, gaze_data):
    """
    Bring response data in model-friendly format.

    Args
    ---
        data (dataframe): aggregate response data
        gaze_data (dataframe): aggregate gaze data

    Returns
    ---
        Dict with formatted data
    """

    # test that only 1 set size
    setsizes = data['setsize'].unique()
    if len(setsizes)>1:
        assert ValueError('More than one set size in data.')
    if setsizes != gaze_data['setsize'].unique():
        assert ValueError('Set size of gaze data and data does not match.')
    setsize = setsizes[0]

    # extract exp. info
    subjects = data['subject'].unique()
    n_subjects = subjects.size
    n_trials = data.shape[0]

    # extract values
    values = data[['item_value_{}'.format(i) for i in range(
        setsize)]].values.astype(np.int)
    # re-scale values to 1 - 7
    values += 4

    # extract gaze
    gaze = data[['gaze_{}'.format(i) for i in range(
        setsize)]].values.astype(np.int)

    # extract choices / RT
    choices = data['choice'].values.astype(np.int)
    rts = data['rt'].values.astype(np.int)

    # determine max RT
    max_RT = np.max(rts)
    for trial in gaze_data['trial'].unique():
        trial_gaze_data = gaze_data[gaze_data['trial']==trial].copy()
        total_gaze_dur = trial_gaze_data['onset'].values[-1] + trial_gaze_data['dur'].values[-1]
        if total_gaze_dur > max_RT:
            max_RT = total_gaze_dur
    max_RT += 1 # add 1ms buffer

    # extract gaze(t) and values(t)
    gaze_t = np.zeros((n_trials, setsize, max_RT))
    values_t = np.zeros_like(gaze_t)
    for i, trial in enumerate(data.trial.unique()):
        trial_data = data[data['trial']==trial].copy()
        trial_gaze_data = gaze_data[gaze_data['trial']==trial].copy()
        trial_gaze_pos = trial_gaze_data['item'].values.astype(np.int)
        trial_gaze_dur = trial_gaze_data['dur'].values.astype(np.int)
        trial_gaze_onsets = trial_gaze_data['onset'].values.astype(np.int)
        # Max-RT is last "onset" and needs one more value than others
        trial_gaze_onsets = np.append(trial_gaze_onsets, max_RT)
        trial_values = values[i]
        t = 0
        T = 0
        seen = np.zeros(setsize).astype(np.bool)
        for f in range(len(trial_gaze_pos)):
            pos = trial_gaze_pos[f]
            onset = trial_gaze_onsets[f]
            dur = trial_gaze_dur[f]
            offset = onset+dur
            next_onset = int(trial_gaze_onsets[f+1])
            seen[pos] = True
            others = np.arange(setsize)!=pos
            for ms in range(T, next_onset):
                if (t >= onset) and (t <= offset):
                    try:
                        gaze_t[i,pos,t] = gaze_t[i,pos,t-1] + 1
                    except:
                        import pdb; pdb.set_trace()
                else:
                    try:
                        gaze_t[i,pos,t] = gaze_t[i,pos,t-1]
                    except:
                        import pdb; pdb.set_trace()
                gaze_t[i,others,t] = gaze_t[i,others,t-1]
                values_t[i,seen,t] = trial_values[seen]
                t += 1
            T = int(next_onset)
        # normalize
        gaze_t[i,:,:t] /= np.arange(1,t+1)
    gaze_t = np.clip(gaze_t, 0, 1) # make sure everything is between 0, 1

    # define error likelihood
    rt_range = data['rt'].max() - data['rt'].min()
    error_ll = 1. / np.float(setsize * rt_range)

    # package data
    data_dict = dict()
    data_dict['rt'] = rts
    data_dict['choice'] = choices
    data_dict['gaze'] = gaze
    data_dict['values'] = values
    data_dict['gaze_t'] = gaze_t
    data_dict['value_t'] = values_t
    data_dict['error_ll'] = error_ll

    return data_dict
