#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import scipy as sp


def sem_p(x, n):
    """
    Compute standard error for probabilities.

    Input
    ---
    x (float):
            average probability
    n (float):
            number of samples from which x was computed


    Returns
    ---
    Float; standard error of x
    """
    return np.sqrt((x*(1-x))/n)


def compute_ttests(X):
    """
    Compute consecutive two-sided t-tests
    over values in columns of input array.

    For an array with 4 columns,
    4 t-tests are computed, between the
    1st/2nd, 2nd/3rd, 3rd/4th, 1st/4th columns.

    Input
    ---
    X (ndarray):
            Array of shape n_samples x n_conditions


    Returns
    ---
    T (array):
            t-values of t-tests

    df (int):
            degrees of freedoms of t-tests

    P (array):
            p-values of t-tests

    """

    from scipy.stats import ttest_rel

    n_comparisons = X.shape[1]
    n_samples = X.shape[0]

    df = n_samples - 1

    T = np.zeros(n_comparisons)
    P = np.zeros(n_comparisons)
    for i in range(n_comparisons-1):
        t, p = ttest_rel(X[:, i], X[:, i+1], nan_policy='omit')
        T[i] = t
        P[i] = p
    t, p = ttest_rel(X[:, 0], X[:, -1])
    T[-1] = t
    P[-1] = p

    return T, df, P


def get(df, prefix):
    """
    Extract those data columns from
    dataframe which include prefix.

    Input
    ---
    df (dataframe)

    prefix (string):
            Prefix of columns to be extracted


    Returns
    ---
    ndarray including data of extracted columns
    """

    n = len([c
             for c in df.columns
             if c.startswith(prefix + '_')])
    columns = [prefix + '_{}'.format(i)
               for i in range(n)]
    return df[columns].values


def get_var(summary, var, verbose=True):
    """
    Get mean and 94% HPD interval for "var" from
    Bambi output summary.

    Input
    ---
    summary (dataframe):
            bambi summary dataframe

    var (sting):
            variable name

    verbose (bool):
            should output be printed to console?


    Returns
    ---
    Mean, (upper, lower) 95% HPD
    """
    mean = summary.loc[var, 'mean']
    hpd_lower = summary.loc[var, 'hdi_3%']
    hpd_upper = summary.loc[var, 'hdi_97%']
    if verbose:
        print('{}; Mean: {}, 94% HPD: {}, {}'.format(
            var, mean, hpd_lower, hpd_upper))
    return mean, (hpd_lower, hpd_upper)


def extract_individual_modes(trace, parameters=None, precision=None, burn=0):
    """
    Extract individual subject modes from
    pymc3 trace.

    Input
    ---
    trace (pymc3 trace):
            PyMC3 trace

    parameters (ndarray):
            Array of parameter names to extract

    precision (int):
            Decimal precision to round trace to,
            before extracting mode

    burn (int):
            How many samples from beginnig
            of trace should be burned?


    Returns
    ---
    array of modes
    """

    from scipy.stats import mode

    modes = []
    n_samples = int(len(trace))
    for parameter, parameter_precision in zip(parameters, precision):
        parameter_trace = np.array([trace[sample][parameter]
                                    for sample in range(int(burn), n_samples)])
        parameter_MAP, _ = mode(np.round(parameter_trace, parameter_precision))
        modes.append(parameter_MAP)

    if len(modes) == 1:
        return modes[0]
    else:
        return np.array(modes).ravel()


def center(x):
    """
    Center values of an array with respect to its mean.

    Input
    ---
    x (array):
            array whose values are to be centered


    Returns
    ---
    centered copy of array
    """
    return (x-np.mean(x))


def check_convergence(summary,
                      parameters=['v', 'gamma', 's', 'tau'],
                      n_eff_required=100,
                      gelman_rubin_criterion=0.05):
    enough_eff_samples = np.all(summary.loc[parameters]['ess_mean'] > n_eff_required)
    good_gelman = np.all(np.abs(summary.loc[parameters]['r_hat'] - 1.0) < gelman_rubin_criterion)
    if not enough_eff_samples or not good_gelman:
        return False
    else:
        return True


def sample_corr(x1, x2, alpha=0.05, verbose=True, return_result=False):

    w, normal_1 = sp.stats.shapiro(x1)
    w, normal_2 = sp.stats.shapiro(x2)
    normality_violated = False

    if (normal_1 < alpha) or (normal_2 < alpha):
        r, p = sp.stats.spearmanr(x1, x2)
        if verbose:
            print('Normality assumption violated: True')
            print('spearman r = {}, p = {}'.format(r, p))
        normality_violated = True
    else:
        r, p = sp.stats.pearsonr(x1, x2)
        if verbose:
            print('pearson r = {}, p = {}'.format(r, p))

    if return_result:
        return r, p, normality_violated
