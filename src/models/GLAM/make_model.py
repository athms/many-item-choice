#!/usr/bin/env python3

import pymc3 as pm
from .components import *
from .utils import format_data

import theano.tensor as tt
import theano


def make_ind_model(data_df, gaze_bias=True, gaze_bias_type='full', **kwargs):
	"""
	Create a individual model from data_df.
	data_df can only include data of a single subject.

	Args
	---
		data_df (dataframe): Aggregate response data
		gaze_bias (bool): whether to activate gaze bias
            or to set gamma=1 and zeta=0
		gaze_bias_type (string): Determine gaze bias type;
			one of ['full',
					'additive' (gamme = 1),
					'multiplicative' (zeta = 0)]

	Returns
	---
		PyMC3 model object
	"""

	assert len(data_df['subject'].unique()
			   ) == 1, 'data_df contains more than 1 subject.'

	# format data
	data_dict = format_data(data_df)

	with pm.Model() as ind_model:

		# Mechanics
		b = pm.Deterministic('b', tt.constant(1, dtype='float32'))
		p_error = pm.Deterministic('p_error', tt.constant(0.05, dtype='float32'))

		# Parameter priors
		v = pm.Uniform('v', 1e-7, 0.005, testval=1e-4)
		s = pm.Uniform('s', 1e-7, 0.05, testval=1e-3)
		tau = pm.Uniform('tau', 0, 10, testval=1)
		if gaze_bias:
			print('\t/!\ Using {} gaze bias'.format(gaze_bias_type))
			if gaze_bias_type == 'multiplicative':
				gamma = pm.Uniform('gamma', 0, 1, testval=0.5)
				zeta = pm.Deterministic('zeta', tt.constant(0, dtype='float32'))
			elif gaze_bias_type == 'additive':
				gamma = pm.Deterministic('gamma', tt.constant(1, dtype='float32'))
				zeta = pm.Uniform('zeta', 0, 10, testval=0.5)
			elif gaze_bias_type == 'full':
				gamma = pm.Uniform('gamma', 0, 1, testval=0.5)
				zeta = pm.Uniform('zeta', 0, 10, testval=0.5)
			else:
				print('/!\ Invalid gaze bias type: {}'.format(gaze_bias_type))
		else:
			gamma = pm.Deterministic('gamma', tt.constant(1, dtype='float32'))
			zeta = pm.Deterministic('zeta', tt.constant(0, dtype='float32'))

		# logp
		def logp_ind(rt,
					 gaze,
					 values,
					 error_ll,
					 zerotol):

			# # compute drifts
			drift = tt_trialdrift(v,
								  tau,
								  gamma,
								  zeta,
								  values,
								  gaze,
								  zerotol)

			# compute pdf
			model_ll = tt_wienerrace_pdf(rt[:, None],
										 drift,
										 s,
										 b,
										 zerotol)

			# mix likelihoods
			mixed_ll = (1-p_error)*model_ll + p_error*error_ll

			# safety
			mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
			mixed_ll = tt.where(tt.isinf(mixed_ll), 0., mixed_ll)

			return tt.log(mixed_ll + zerotol)

		# obs
		obs = pm.DensityDist('obs', logp=logp_ind,
							 observed=dict(rt=data_dict['rts'],
										   gaze=data_dict['gaze'],
										   values=data_dict['values'],
										   error_ll=data_dict['error_lls'],
										   zerotol=1e-10))

	return ind_model
