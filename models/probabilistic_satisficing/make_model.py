#!/usr/bin/env python3

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

from .utils import format_data


def make_ind_model(subject_data, subject_gaze_data, gaze_bias=True, zerorol=1e-10):
    """
    Make single subject probabilistic satisficing choice model.

    Args
    ---
        subject_data_df (dataframe): aggregate response
            data of a single subject
        subject_fixation_data (dataframe): aggregate fixation
            data of a single subject
        gaze_bias (bool): whether to activate gaze bias
            or to set gamma=1 and zeta=0
        zerotol (float): numerical stability term

    Returns
    ---
        Single subject hybrid choice model
    """

    assert len(subject_data['subject'].unique()) == 1, 'data_df contains more than 1 subject.'

    # format data
    data_dict = format_data(subject_data, subject_gaze_data)

    # make ind model
    with pm.Model() as ind_model:

        # likelihood mixture
        p_error = pm.Deterministic('p_error', tt.constant(0.05, dtype='float32'))

        # model paramaters
        v = pm.Uniform('v', 0, 0.001, testval=1e-7)
        alpha = pm.Uniform('alpha', 0, 0.001, testval=1e-7)
        tau = pm.Uniform('tau', 0, 10, testval=1)
        if gaze_bias:
            gamma = pm.Uniform('gamma', 0, 1, testval=0.5)
            zeta = pm.Uniform('zeta', 0, 10, testval=0.5)
        else:
            gamma = pm.Deterministic('gamma', tt.constant(1, dtype='float32'))
            zeta = pm.Deterministic('zeta', tt.constant(0, dtype='float32'))

        # stopping probability
        def stopping_probability(gaze_t, value_t, rt):
            time = tt.arange(1,gaze_t.shape[-1]+1)[None,:]
            # exclude items that were not looked at so far
            C = tt.where(tt.eq(gaze_t, tt.zeros_like(gaze_t)),
                         tt.zeros_like(gaze_t),
                         gaze_t * (value_t + zeta) + (1 - gaze_t) * gamma * value_t)
            q = v * time + alpha * tt.max(C, axis=1)
            q = tt.clip(q, 0, 1)
            Q = tt.cumprod(1 - q, axis=1)
            Q = tt.clip(Q, 0, 1)
            q_corrected = (Q[tt.cast(tt.arange(Q.shape[0]), dtype='int32'), tt.cast(rt-2, dtype='int32')] *
                           q[tt.cast(tt.arange(q.shape[0]), dtype='int32'), tt.cast(rt-1, dtype='int32')])
            return q_corrected, C

        # logp
        def hybrid_logp(rt,
                        choice,
                        gaze_t,
                        value_t,
                        error_ll,
                        zerotol):
            # compute stopping probability
            n_trials = value_t.shape[0]
            n_items = value_t.shape[1]
            q, C = stopping_probability(gaze_t, value_t, rt)
            # compute softmax choice probabilities
            sigma, _ = theano.scan(lambda i, tau, C, rt, n_items:
                                   tt.nnet.nnet.softmax(tau*C[tt.cast(tt.repeat(i, n_items), dtype='int32'),
                                                              tt.arange(n_items, dtype='int32'),
                                                              tt.cast(tt.repeat(rt[i]-1, n_items), dtype='int32')]).flatten(),
                                   sequences=[tt.cast(tt.arange(n_trials), dtype='int32')],
                                   non_sequences=[tau, C, rt, n_items])
            # combine with choice probabilities
            p = q * sigma[tt.arange(n_trials, dtype='int32'), tt.cast(choice, dtype='int32')]
            # mix likelihoods
            l = ((1-p_error) * p) + (p_error * error_ll)
            # safety
            l = tt.where(tt.isnan(l), 0., l)
            l = tt.where(tt.isinf(l), 0., l)
            return tt.log(l + zerotol)

        # data
        obs = pm.DensityDist('obs',
                logp=hybrid_logp,
                observed=dict(rt=data_dict['rt'],
                              choice=data_dict['choice'],
                              gaze_t=data_dict['gaze_t'],
                              value_t=data_dict['value_t'],
                              error_ll=data_dict['error_ll'],
                              zerotol=zerorol))

    return ind_model
