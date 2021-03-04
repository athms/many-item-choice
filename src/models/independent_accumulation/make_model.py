#!/usr/bin/env python3

import pymc3 as pm
from .components import *
from .utils import format_data

import theano.tensor as tt
import theano


def make_ind_model(subject_data, subject_gaze_data, gaze_bias=True, **kwargs):
	"""
	Create a individual model from subject_data.
	subject_data can only include data of a single subject.

	Args
	---
		subject_data (dataframe): aggregate response
            data of a single subject
        subject_gaze_data (dataframe): aggregate gaze
            data of a single subject
		gaze_bias (bool): whether to activate gaze bias
            or to set gamma=1 and zeta=0

	Returns
	---
		PyMC3 model object
	"""

	assert len(subject_data['subject'].unique()
			   ) == 1, 'subject_data contains more than 1 subject.'

	# format data
	data_dict = format_data(subject_data, subject_gaze_data)

	with pm.Model() as ind_model:

		# Mechanics
		b = pm.Deterministic('b', tt.constant(1, dtype='float32'))
		p_error = pm.Deterministic('p_error', tt.constant(0.05, dtype='float32'))

		# Parameter priors
		v = pm.Uniform('v', 1e-7, 0.005, testval=1e-4)
		s = pm.Uniform('s', 1e-7, 0.05, testval=1e-3)
		if gaze_bias:
			gamma = pm.Uniform('gamma', 0, 1, testval=0.5)
			zeta = pm.Uniform('zeta', 0, 10, testval=0.5)
		else:
			gamma = pm.Deterministic('gamma', tt.constant(1, dtype='float32'))
			zeta = pm.Deterministic('zeta', tt.constant(0, dtype='float32'))

		# logp
		def logp_ind(rt,
					 gaze_corrected,
					 values,
					 gaze_onset,
					 n_seen,
					 error_ll,
					 zerotol):

			# compute drifts
			drift = tt_trialdrift(v,
								  gamma,
								  zeta,
								  values,
								  gaze_corrected,
								  zerotol)

			# correct rt for gaze onset
			time_seen = rt[:,None] - gaze_onset
			time_seen = tt.where(time_seen < 1., 1., time_seen)

			# compute corrected FTP
			model_ll, _ = theano.scan(lambda i, time_seen, drift, s, b, n_seen:
	                                   tt_wienerrace_pdf(time_seen[i], drift[i], s, b, n_seen[i]),
	                                   sequences=[tt.cast(tt.arange(time_seen.shape[0]), dtype='int32')],
	                                   non_sequences=[time_seen, drift, s, b, n_seen])

			# mix likelihoods
			mixed_ll = (1-p_error)*model_ll + p_error*error_ll

			# safety
			mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
			mixed_ll = tt.where(tt.isinf(mixed_ll), 0., mixed_ll)

			return tt.log(mixed_ll + zerotol)

		# obs
		obs = pm.DensityDist('obs', logp=logp_ind,
							 observed=dict(rt=data_dict['rts'],
										   gaze_corrected=data_dict['gaze_corrected'],
										   values=data_dict['values'],
										   gaze_onset=data_dict['gaze_onset'],
										   n_seen=data_dict['n_seen'],
										   error_ll=data_dict['error_ll'],
										   zerotol=1e-10))

	return ind_model
