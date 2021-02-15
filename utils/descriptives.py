#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils.stats import sem_p


def compute_gaze_influence_score(data):
    """
    Compute gaze influence score for each
    subject in the data;

    Gaze influence score is defined
    as the average difference between
    the corrected choice probability
    of all positive and negative relative gaze values
    (see manuscript).

    Input
    ---
    data (dataframe):
            aggregate response data

    Returns
    ---
    array of single-subject gaze influence scores
    """

    df = data.copy()
    n_items = np.int(data['setsize'].unique())

    choice = np.zeros((df.shape[0], n_items))
    choice[np.arange(df.shape[0]), df['choice'].values.astype('int32')] = 1

    # compute value measures
    gaze = df[['gaze_{}'.format(i) for i in range(n_items)]].values
    rel_gaze = np.zeros_like(gaze)
    values = df[['item_value_{}'.format(i) for i in range(n_items)]].values
    rel_values = np.zeros_like(values)
    value_range = np.zeros_like(values)
    for t in range(values.shape[0]):
        for i in range(n_items):
            index = np.where(np.arange(n_items) != i)
            rel_gaze[t, i] = gaze[t, i] - np.max(gaze[t, index])
            rel_values[t, i] = values[t, i] - np.mean(values[t, index])
            value_range[t, i] = np.max(
                values[t, index]) - np.min(values[t, index])

    # create new df
    df_tmp = pd.DataFrame(dict(subject=np.repeat(df['subject'].values, n_items),
                               is_choice=choice.ravel(),
                               value=values.ravel(),
                               rel_value=rel_values.ravel(),
                               value_range_others=value_range.ravel(),
                               rel_gaze=rel_gaze.ravel(),
                               gaze_pos=np.array(
                                   rel_gaze.ravel() > 0, dtype=np.bool),
                               gaze=gaze.ravel()))

    # extimate value-based choice prob.
    # for each individual and subtract
    # from empirical choice
    data_out = pd.DataFrame()
    for s, subject in enumerate(data['subject'].unique()):
        subject_data = df_tmp[df_tmp['subject'] == subject].copy()

        X = subject_data[['rel_value', 'value_range_others']]
        X = sm.add_constant(X)
        y = subject_data['is_choice']

        logit = sm.Logit(y, X)
        result = logit.fit(disp=0, method='lbfgs', maxiter=100)
        predicted_pchoose = result.predict(X)

        subject_data['corrected_choice'] = subject_data['is_choice'] - \
            predicted_pchoose
        data_out = pd.concat([data_out, subject_data])

    # compute corrected psychometric, given gaze
    tmp = data_out.groupby(['subject', 'gaze_pos']
                           ).corrected_choice.mean().unstack()
    gaze_influence_score = (tmp[True] - tmp[False]).values

    return gaze_influence_score


def compute_p_last_gaze_choice(data,
                               gaze_data,
                               is_return_subject_means=True,
                               is_return_data=False):
    """
    Compute mean probability that last trial gaze
    is to the chosen item;

    Input
    ---
    data (dataframe):
            aggregate response data

    gaze_data (dataframe):
            aggregate gaze data

    is_return_subject_means (bool):
            should subject means be returned?

    is_return_data (bool):
            should aggregate of subject-fixation data be returned?

    Returns
    ---
    (subject-fixation-data, subject-means), grand means, standard errors of grand means,
    """

    subjects = data.subject.unique()
    setsizes = data.setsize.unique()

    gaze_data_out = []
    for subject in subjects:
        # load subject data
        subject_data = data[data['subject']==subject].copy()
        subject_gaze_data = gaze_data[gaze_data['subject']==subject].copy()
        seen_items_count = []
        last_gaze_to_choice = []
        for trial in subject_gaze_data.trial.unique():
            trial_gaze_data = subject_gaze_data[subject_gaze_data['trial']==trial].copy()
            last_fixated_item = trial_gaze_data.tail(1)['item'].values[0]
            trial_choice = trial_gaze_data['choice'].values[0]
            if last_fixated_item == trial_choice:
                last_gaze_to_choice.append(np.ones(trial_gaze_data.shape[0]))
            else:
                last_gaze_to_choice.append(np.zeros(trial_gaze_data.shape[0]))
            seen_items_count.append([len(trial_gaze_data['item'].unique())] *
                                    trial_gaze_data.shape[0])
        # index last gaze
        subject_gaze_data['last_gaze_to_choice'] = np.concatenate(last_gaze_to_choice)
        subject_gaze_data['seen_items_count'] = np.concatenate(seen_items_count)
        subject_gaze_data['all_items_seen'] = np.array(
            subject_gaze_data['seen_items_count'] == subject_gaze_data['setsize'],
            dtype=np.int)
        gaze_data_out.append(subject_gaze_data)

    # sub-set data to last gaze
    gaze_data_out = pd.concat(gaze_data_out, sort=True)
    last_gaze_data = gaze_data_out[gaze_data_out['is_last']==1].copy()

    if is_return_subject_means:
        sub_means = np.concatenate([last_gaze_data.groupby(
                    ['setsize', 'subject']).last_gaze_to_choice.mean()[s][:,None]
                                    for s in setsizes], axis=1)
    means = last_gaze_data.groupby(['setsize',
                                    'subject']).last_gaze_to_choice.mean(
    ).groupby(level=0).mean()
    sems = sem_p(means, subjects.size)

    if is_return_data:
        if is_return_subject_means:
            return last_gaze_data, sub_means, means, sems
        else:
            return last_gaze_data, means, sems
    else:
        if is_return_subject_means:
            return sub_means, means, sems
        else:
            return means, sems


def compute_p_choose_best(df):
    """
    Computes subject-wise P(choose best)

    Input
    ---
    df : dataframe
        aggregate response data

    Returns
    ---
        array of subject-wise P(choose best)
    """
    if 'best_chosen' not in df.columns:
        df = add_best_chosen(df)
    return df.groupby('subject').best_chosen.mean().values


def add_best_chosen(df):
    """
    Adds 'best_chosen' variable to DataFrame,
    independent of number of items
    (works with nan columns)

    Input
    ---
    df : dataframe
        aggregate response data

    Returns
    ---
        copy of df with 'best chosen' indicator
    """
    df = df.copy()
    values = df[[c for c in df.columns if c.startswith('item_value_')]].values
    choices = df['choice'].values.astype(np.int)
    best_chosen = (values[np.arange(choices.size), choices] == np.nanmax(
        values, axis=1)).astype(int)
    df['best_chosen'] = best_chosen
    return df
    

def compute_mean_rt(df):
    """
    Computes subject-wise mean RT

    Input
    ---
    df : dataframe
        aggregate response data

    Returns
    ---
        array of subject-wise mean RTs
    """
    return df.groupby('subject').rt.mean().values


def q1(series):
    """
    Extract 25% quantile from
    pandas series

    Input
    ---
        series : pandas series

    Returns
    ---
        25% quantile
    """
    q1 = series.quantile(0.25)
    return q1


def q3(series):
    """
    Extract 75% quantile from
    pandas series

    Input
    ---
        series : pandas series

    Returns
    ---
        27% quantile
    """
    q3 = series.quantile(0.75)
    return q3


def iqr(series):
    """
    Extract inter-quantile range
    (25%-75%)

    Input
    ---
        series : pandas series

    Returns
    ---
        inter-quantile range
        (25-75%)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return IQR


def std(series):
    """
    Extract standar deviation (SD)

    Input
    ---
        series : pandas series

    Returns
    ---
        SD
    """
    sd = series.std(ddof=0)
    return sd


def se(series):
    """
    Extract standar error (SE)

    Input
    ---
        series : pandas series

    Returns
    ---
        SE
    """
    n = len(series)
    se = series.std() / np.sqrt(n)
    return se


def aggregate_subject_level_data(data, n_items):
    """
    Compute subject-level response characteristics on:
    RT, P(choose best), gaze influence score

    The gaze influence score is defined
    as the average difference between
    the corrected choice probability
    of all positive and negative relative gaze values
    (see manuscript)

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the experimental data.
        Each row corresponds to one trial.
        Must include the following columns:

        - `subject` (int, consecutive, starting with 0)
        - `trial` (int, starting with 0)
        - `choice` (int, items should be 0, 1, ..., N)
        - `rt` (float, in seconds)
        - additional variables coding groups or conditions (str or int)

        For each item `i` in the choice set:

        - `item_value_i`: The item value (float, best on a scale between 1 and 10)
        - `gaze_i`: The fraction of total trial time the item was looked at in the trial (float, between 0 and 1)

    n_items : int
        number of choice alternatives in the data

    Returns
    -------
    pandas.DataFrame
        DataFrame of subject-level response characteristics.
    """
    # make copy of data
    data = data.copy()

    # add best chosen variable
    data = add_best_chosen(data)

    # Summarize variables
    subject_summary = data.groupby('subject').agg({
        'rt': ['mean', std, 'min', 'max', se, q1, q3, iqr],
        'best_chosen':
        'mean'
    })
    # Influence of gaze on P(choose left)
    subject_summary['gaze_influence'] = compute_gaze_influence_score(data)

    return subject_summary
