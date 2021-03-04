#!/usr/bin/env python3

import theano.tensor as tt
from theano import scan
import numpy as np


def tt_normal_cdf(x, mu=0, sd=1):
    """
    Normal cumulative distribution function
    Theano tensor implementation
    """
    return (0.5 + 0.5 * tt.erf((x - mu) / (sd * tt.sqrt(2.))))


def tt_wienerpos_fpt_pdf(t, drift, noise, boundary):
    """
    Probability density function of first passage times of
    Wiener process with positive drift towards constant boundary.
    Theano tensor implementation

    Cf https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Relationship_with_Brownian_motion
    """
    mu = boundary / drift
    lam = (boundary**2 / noise**2)
    return ((lam / (2*np.pi*t**3))**0.5 * tt.exp((-lam * (t - mu)**2)/(2*mu**2*t)))


def tt_wienerpos_fpt_cdf(t, drift, noise, boundary):
    """
    Cumulative distribution function of first passage times of
    Wiener process with positive drift towards constant boundary.
    Theano tensor implementation

    Cf https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Relationship_with_Brownian_motion
    """
    mu = boundary / drift
    lam = (boundary**2) / (noise**2)
    return (tt_normal_cdf(tt.sqrt(lam / t) * (t / mu - 1)) +
            tt.exp(2*(lam/mu)) * tt_normal_cdf(-(tt.sqrt(lam / t) * (t / mu + 1))))


def tt_wienerrace_pdf(t, drift, noise, boundary, zerotol=1e-14):
    """
    Probability density function of first passage times of
    a race between multiple Wiener processes with positive drift
    towards a constant boundary. Assumes that chosen item
    is at column 0.
    Theano tensor implementation
    """
    # first passage time densities, single Wiener accumulators
    f = tt_wienerpos_fpt_pdf(t, drift, noise, boundary)
    # first passage time distributions, single Wiener accumulators
    F = tt_wienerpos_fpt_cdf(t, drift, noise, boundary)
    # survival functions
    S = 1. - F
    # race densities
    # Note: drifts should be sorted so that chosen item drift is in first column
    l = f[:, 0] * tt.prod(S[:, 1:], axis=1)
    return l


def tt_trialdrift(v, tau, gamma, zeta, values, gaze, zerotol):
    """
    Computes drifts
    """

    # absolute decision signals
    A = gaze * (values + zeta) + (1 - gaze) * gamma * values

    # relative decision signals
    n_items = tt.cast(A.shape[1], dtype='int32')
    stacked_A = tt.repeat(A, repeats=n_items, axis=1).T
    stacked_A_reshaped = tt.reshape(stacked_A,
                                    newshape=(A.shape[1],
                                              A.shape[1],
                                              A.shape[0])).T
    identity = 1 - tt.eye(n_items)
    max_others = tt.max(stacked_A_reshaped * identity[None, :, :], axis=2)
    R_star = A - max_others
    R = v * 1 / (1 + tt.exp(-tau * R_star))

    # set R = zerotol for items that were not seen
    R = tt.where(tt.eq(gaze, tt.zeros_like(gaze)), zerotol, R)
    R = tt.where(R < zerotol, zerotol, R)
    return R
