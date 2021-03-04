#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import invgauss
from .utils import format_data


def _softmax(x):
    """
    softmax function
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(data, gaze_data, estimates, n_repeats=1, error_weight=0.05, error_range=(0, 10000)):
    """
    Simulate response data for
    each subject in data.

    Args
    --
        data (dataframe): aggregate response data
        gaze_data (dataframe): aggregate gaze data
        estimates (dict(: hybrid choice model parameter estimates
        n_repeats (int): how often to repeat each trial in data?
        error_weight (float; 0 - 1): probability with which response is simulated from error model
        error_range (array): (lower, upper) RT limit of error model

    Returns
    ---
        dataframe with simualted response data
    """

    prediction = pd.DataFrame()

    subjects = np.unique(data['subject'])
    n_items = len([col for col in data.columns
                   if col.startswith('item_value_')])

    for s, subject in enumerate(subjects):

        # format data
        data_subject = data[data['subject'] == subject].copy()
        gaze_data_subject = gaze_data[gaze_data['subject'] == subject].copy()
        data_dict = format_data(data_subject,
                                gaze_data_subject)

        # extract parameters
        parameters = [estimates.get(parameter)[s]
                      for parameter in ['v', 'alpha', 'gamma', 'zeta', 'tau']]

        # extract data
        value_cols = ['item_value_{}'.format(i) for i in range(n_items)]
        values = data_subject[value_cols].values

        gaze_cols = ['cumulative_gaze_{}'.format(i) for i in range(n_items)]
        gaze = data_subject[gaze_cols].values

        stimulus_cols = ['stimulus_{}'.format(i) for i in range(n_items)]
        stimuli = data_subject[stimulus_cols].values

        # gaze & values by time
        value_t = data_dict['value_t']
        gaze_t = data_dict['gaze_t']

        # error range
        rt_min = data_subject['rt'].values.min()
        rt_max = data_subject['rt'].values.max()
        error_range = (rt_min, rt_max)

        # simulate subject
        subject_prediction = simulate_subject(parameters,
                                              value_t, gaze_t,
                                              values, gaze, stimuli,
                                              data_subject.trial.values,
                                              n_repeats=n_repeats,
                                              subject=subject,
                                              error_weight=error_weight,
                                              error_range=error_range)
        prediction = pd.concat([prediction, subject_prediction])

    return prediction


def simulate_subject(parameters, value_t, gaze_t, values, gaze, stimuli, trials, n_repeats=1, subject=0,
                     error_weight=0.05, error_range=(0, 5000)):
    """
    Simulate subject.
    """

    n_trials, n_items = values.shape

    rts = np.zeros(n_trials * n_repeats) * np.nan
    choices = np.zeros(n_trials * n_repeats) * np.nan
    trial_idx = np.zeros(n_trials * n_repeats) * np.nan
    repeat_idx = np.zeros(n_trials * n_repeats) * np.nan

    running_idx = 0

    for trial, trial_id in enumerate(trials):

        for repeat in range(n_repeats):

            choice, rt = simulate_trial(parameters,
                                        value_t[trial],
                                        gaze_t[trial],
                                        error_weight=error_weight,
                                        error_range=error_range)

            if choice is not None:
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

    df = df[np.isfinite(df['choice'].values)].copy()

    return df


def simulate_trial(parameters, value_t, gaze_t, error_weight=0.05, error_range=(0, 5000), maxtime=50000):
    """
    Simulate trial
    """
    v, alpha, gamma, zeta, tau = parameters
    n_items = value_t.shape[0]

    if np.random.uniform(0, 1) < error_weight:
        rt = int(np.random.uniform(*error_range))
        choice = np.random.choice(n_items)

    else:
        t = 0
        choice = None
        rt = None
        while (choice is None) & (t <= maxtime):
            t += 1
            if t-1 < value_t.shape[-1]:
                v_t = value_t[:,t-1]
                g_t = gaze_t[:,t-1]
            else:
                v_t = value_t[:,-1]
                g_t = gaze_t[:,-1]
            seen_items = np.where(g_t>0)[0]
            # only make choice once at least one item seen
            if len(seen_items) > 0:
                C = g_t * (v_t + zeta) + (1 - g_t) * gamma * v_t
                p_stop = v * t + alpha * np.max(C[seen_items])
                is_stop = np.random.choice([True, False], p=[p_stop, (1-p_stop)])
                if is_stop:
                    rt = t
                    choice = np.random.choice(seen_items, p=_softmax(tau*C[seen_items]))
    return choice, rt
