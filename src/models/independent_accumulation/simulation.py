#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import invgauss
from .utils import add_gaze_corrected


def predict(data, gaze_data, estimates, n_repeats=1, boundary=1., error_weight=0.05, **kwards):
    """
    Simulate response data for all
    subjects in dataframe.

    Args
    ---
        data (dataframe): aggregate respone data
        gaze_data (dataframe): aggregate gaze data 
        estimates (dict): dict of parameter estimates
        n_repeats (int): how often to repeat each trial
        boundary (float): decision boundary
        error_weight (float; 0 - 1): probability with which response
            is simulated with error model

    Returns
    ---
        dataframe of simulated response data
    """

    prediction = pd.DataFrame()

    subjects = np.unique(data['subject'])
    n_items = len([col for col in data.columns
                   if col.startswith('item_value_')])

    for s, subject in enumerate(subjects):

        # subset data
        sub_data = data[data['subject']==subject].copy()
        sub_gaze_data = gaze_data[gaze_data['subject']==subject].copy()

        # get estimates
        parameters = np.array([estimates.get(parameter)[s]
                               for parameter in
                               ['v', 'gamma', 'zeta', 's']])

        # set cols to extract
        value_cols = ['item_value_{}'.format(i) for i in range(n_items)]
        gaze_cols = ['cumulative_gaze_{}'.format(i) for i in range(n_items)]
        gaze_onset_cols = ['gaze_onset_{}'.format(i) for i in range(n_items)]
        stimulus_cols = ['stimulus_{}'.format(i) for i in range(n_items)]

        # add gaze_corrected columns
        sub_data = add_gaze_corrected(sub_data, sub_gaze_data)
        gaze_corrected_cols = ['gaze_corrected_{}'.format(i) for i in range(n_items)]

        # extract data
        values = sub_data[value_cols].values
        gaze = sub_data[gaze_cols].values
        gaze_corrected = sub_data[gaze_corrected_cols].values
        gaze_onset = sub_data[gaze_onset_cols].values
        stimuli = sub_data[stimulus_cols].values
        trial_indeces = sub_data['trial'].values

        # define error range
        rt_min = sub_data['rt'].values.min()
        rt_max = sub_data['rt'].values.max()
        error_range = (rt_min, rt_max)

        # simulate subject
        subject_prediction = simulate_subject(parameters,
                                              values,
                                              gaze, gaze_corrected, gaze_onset,
                                              stimuli,
                                              trial_indeces,
                                              n_repeats=n_repeats,
                                              subject=subject,
                                              boundary=boundary,
                                              error_weight=error_weight,
                                              error_range=error_range)
        prediction = pd.concat([prediction, subject_prediction])

    return prediction


def simulate_subject(parameters, values, gaze, gaze_corrected, gaze_onset, stimuli, trials,
                     n_repeats=1, subject=0, boundary=1, error_weight=0.05, error_range=(0, 5000)):
    """
    Simulate subject
    """

    n_trials, n_items = values.shape

    rts = np.zeros(n_trials * n_repeats) * np.nan
    choices = np.zeros(n_trials * n_repeats) * np.nan
    trial_idx = np.zeros(n_trials * n_repeats) * np.nan
    repeat_idx = np.zeros(n_trials * n_repeats) * np.nan

    running_idx = 0

    for trial, trial_id in enumerate(trials):

        for repeat in range(n_repeats):

            choice_made = False
            while not choice_made:
                choice, rt = simulate_trial(parameters,
                                            values[trial],
                                            gaze_corrected[trial], gaze_onset[trial],
                                            boundary=boundary,
                                            error_weight=error_weight,
                                            error_range=error_range)
                if rt >= 0:
                    choice_made = True

            rts[running_idx] = rt
            choices[running_idx] = choice
            trial_idx[running_idx] = trial_id
            repeat_idx[running_idx] = repeat

            running_idx += 1

    df = pd.DataFrame(dict(subject=np.ones(n_trials*n_repeats) * subject,
                           trial=trial_idx,
                           repeat=repeat_idx,
                           choice=choices,
                           rt=rts))

    for i in range(n_items):
        df['item_value_{}'.format(i)] = np.repeat(values[:, i], n_repeats)
        df['cumulative_gaze_{}'.format(i)] = np.repeat(gaze[:, i], n_repeats)
        df['stimulus_{}'.format(i)] = np.repeat(stimuli[:, i], n_repeats)

    return df


def simulate_trial(parameters, values, gaze_corrected, gaze_onset, boundary=1, error_weight=0.05, error_range=(0, 5000)):
    """
    Simulate trial
    """
    v, gamma, zeta, s = parameters
    n_items = len(values)

    if np.random.uniform(0, 1) < error_weight:
        rt = int(np.random.uniform(*error_range))
        choice = np.random.choice(n_items)

    else:

        drifts = trialdrift(v, gamma, zeta, values, gaze_corrected, zerotol=1e-10)

        mu = float(boundary) / drifts
        lam = float(boundary**2) / (s**2)
        FPTs = invgauss.rvs(mu=mu/lam, scale=lam)

        # add gaze onset
        gaze_onset[gaze_corrected==0] = np.inf
        FPTs += gaze_onset

        choice = np.nanargmin(FPTs)
        rt = int(np.round(FPTs[choice]))

    return choice, rt


def trialdrift(v, gamma, zeta, values, gaze_corrected, zerotol=1e-10):
    """
    Compute trial drifts
    """
    A = gaze_corrected * (values + zeta) + (1-gaze_corrected) * gamma * values
    D = v * A
    D[D < zerotol] = zerotol
    return D
