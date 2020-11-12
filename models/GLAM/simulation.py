#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import invgauss


def predict(data, estimates, n_repeats=1, boundary=1., error_weight=0.05, **kwards):
    """
    Simulate response data for all
    subjects in dataframe.

    Args
    ---
        data (dataframe): aggregate respone data
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

        parameters = np.array([estimates.get(parameter)[s]
                               for parameter in
                               ['v', 'gamma', 'zeta', 's', 'tau']])

        value_cols = ['item_value_{}'.format(i) for i in range(n_items)]
        gaze_cols = ['gaze_{}'.format(i) for i in range(n_items)]
        stimulus_cols = ['stimulus_{}'.format(i) for i in range(n_items)]

        values = data[value_cols][data['subject'] == subject].values
        values_scaled = values + 4
        gaze = data[gaze_cols][data['subject'] == subject].values
        stimuli = data[stimulus_cols][data['subject'] == subject].values
        trial_indeces = data[data['subject'] == subject]['trial'].values

        rt_min = data['rt'][data['subject'] == subject].values.min()
        rt_max = data['rt'][data['subject'] == subject].values.max()
        error_range = (rt_min, rt_max)

        subject_prediction = simulate_subject(parameters,
                                              values_scaled,
                                              gaze,
                                              stimuli,
                                              trial_indeces,
                                              n_repeats=n_repeats,
                                              subject=subject,
                                              boundary=boundary,
                                              error_weight=error_weight,
                                              error_range=error_range)
        prediction = pd.concat([prediction, subject_prediction])

    return prediction


def simulate_subject(parameters, values, gaze, stimuli, trials,
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
                                            gaze[trial],
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
        df['gaze_{}'.format(i)] = np.repeat(gaze[:, i], n_repeats)
        df['stimulus_{}'.format(i)] = np.repeat(stimuli[:, i], n_repeats)

    return df


def simulate_trial(parameters, values, gaze, boundary=1, error_weight=0.05, error_range=(0, 5000)):
    """
    Simulate trial
    """
    v, gamma, zeta, s, tau = parameters
    n_items = len(values)

    if np.random.uniform(0, 1) < error_weight:
        rt = int(np.random.uniform(*error_range))
        choice = np.random.choice(n_items)

    else:
        drifts = trialdrift(v, tau, gamma, zeta, values, gaze, zerotol=1e-10)

        mu = float(boundary) / drifts
        lam = float(boundary**2) / (s**2)
        FPTs = invgauss.rvs(mu=mu/lam, scale=lam)

        choice = np.nanargmin(FPTs)
        rt = int(np.round(FPTs[choice]))

    return choice, rt


def trialdrift(v, tau, gamma, zeta, values, gaze, zerotol=1e-10):
    """
    Compute drifts
    """

    A = gaze * (values + zeta) + (1 - gaze) * gamma * values

    n_items = len(A)
    max_others = np.zeros(n_items)
    for item in range(n_items):
        max_others[item] = np.max(A[np.arange(n_items)!=item])
    R_star = A - max_others
    R = v * 1 / (1 + np.exp(-tau * R_star))

    R[R < zerotol] = zerotol
    return R
