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


def tt_wienerrace_pdf(t, drift, noise, boundary, n, zerotol=1e-14):
    """
    Probability density function of first passage times of
    a race between multiple Wiener processes with positive drift
    towards a constant boundary. Assumes that chosen item
    is at column 0.
    Theano tensor implementation
    """
    # first passage time densities, single Wiener accumulators
    f = tt_wienerpos_fpt_pdf(t[:tt.cast(n, dtype='int32')], drift[:tt.cast(n, dtype='int32')], noise, boundary)
    # first passage time distributions, single Wiener accumulators
    F = tt_wienerpos_fpt_cdf(t[:tt.cast(n, dtype='int32')], drift[:tt.cast(n, dtype='int32')], noise, boundary)
    # survival functions
    S = 1. - F
    # race densities
    # Note: drifts should be sorted so that chosen item drift is in first column
    l = tt.switch(n > 1., f[0]*tt.prod(S[1:]), f[0])
    return l


def tt_trialdrift(v, gamma, zeta, values, gaze_corrected, zerotol):
    """
    Computes drifts
    """
    A = gaze_corrected * (values + zeta) + (1 - gaze_corrected) * gamma * values
    D = v * A
    D = tt.where(D < zerotol, zerotol, D)
    return D
