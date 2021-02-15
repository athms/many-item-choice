#!/usr/bin/env python3

import os, sys
import pandas as pd
import numpy as np
from scipy.stats import mode
import bambi
import pymc3 as pm
from pymc3.plots import traceplot
import arviz as az
import matplotlib.pyplot as plt

import models
from utils.general import make_sure_path_exists
from utils.stats import check_convergence


if __name__ == '__main__':

    # define subject / setsize
    subject = int(sys.argv[1])
    setsize = int(sys.argv[2])

    print('\nProcessing subject: {} ({})'.format(subject, setsize))

    # make sure all output directories exist
    mfx_dir = 'results/posterior_traces/mixed_effects_models/'
    # make sure output dir exists
    for model_name in ['probabilistic_satisficing', 'independent_accumulation', 'GLAM']:
        for gaze_bias in ['with_active_gaze', 'with_passive_gaze']:
            make_sure_path_exists('results/predictions/{}_{}/'.format(model_name, gaze_bias))
            make_sure_path_exists('results/posterior_traces/{}_{}/'.format(model_name, gaze_bias))
            make_sure_path_exists('results/waic/{}_{}/'.format(model_name, gaze_bias))

    # set start seed
    seed = int(subject + setsize)
    np.random.seed(seed)

    # load data
    data = pd.read_csv('data/summary_files/{}_data.csv'.format(setsize))
    # subset to subject
    subject_data = data[data['subject']==subject].copy()

    # load gaze data
    subject_gaze_data = pd.read_csv('data/subject_files/{}_{}_fixations.csv'.format(subject, setsize))

    # iterate over models
    for model_name in ['probabilistic_satisficing', 'independent_accumulation', 'GLAM']:

        # model settings model
        if model_name == 'independent_accumulation':
            ind_model = models.independent_accumulation
            model_parameters = ['v', 'gamma', 'zeta', 's']
            to_round = [6, 2, 2, 4]
        elif model_name == 'GLAM':
            ind_model = models.GLAM
            model_parameters = ['v', 'gamma', 'zeta', 's', 'tau']
            to_round = [6, 2, 2, 4, 2]
        elif model_name == 'probabilistic_satisficing':
            ind_model = models.probabilistic_satisficing
            model_parameters = ['v', 'alpha', 'gamma', 'zeta', 'tau']
            to_round = [8, 8, 2, 2, 2]
        else:
            raise NameError('Invalid model name')

        # iterate over gaze bias on / off
        for gaze_bias_on, gaze_bias in zip([True, False], ['with_active_gaze', 'with_passive_gaze']):

            print('\tModel: {}_{}'.format(model_name, gaze_bias))

            # estimate
            if not os.path.isfile(
                'results/waic/{}_{}/{}_{}_waic.csv'.format(
                    model_name, gaze_bias, subject, setsize)):
                print('\t\tEstimating parameters')

                # make pymc model
                pymc_model = ind_model.make_ind_model(subject_data, subject_gaze_data=subject_gaze_data, gaze_bias=gaze_bias_on)
                pymc_model.name = model_name

                # sampling
                tries = 0
                model_converged = False
                while (not model_converged) and (tries < 50):
                    with pymc_model:
                        mtrace = pm.sample(draws=5000, tune=5000+tries*5000,
                                           cores=2, chains=2,
                                           step=pm.Metropolis(),
                                           random_seed=np.int(seed+tries))
                        if gaze_bias_on:
                            model_converged = check_convergence(pm.summary(mtrace, round_to="none"),
                                parameters=model_parameters)
                        else:
                            model_converged = check_convergence(pm.summary(mtrace, round_to="none"),
                                parameters=[p for p in model_parameters if p!='gamma' and p!='zeta'])
                        tries += 1

                # store results
                print('\t\tSaving results')
                mtrace_df = pm.trace_to_dataframe(mtrace)
                mtrace_df.to_csv(
                    'results/posterior_traces/{}_{}/{}_{}_mtrace.csv'.format(
                        model_name, gaze_bias, subject, setsize), index=False)
                mtrace_summary = az.summary(mtrace, round_to="none")
                mtrace_summary.to_csv(
                    'results/posterior_traces/{}_{}/{}_{}_mtrace_summary.csv'.format(
                        model_name, gaze_bias, subject, setsize))
                traceplot(mtrace)
                plt.savefig(
                    'results/posterior_traces/{}_{}/{}_{}_mtrace.png'.format(
                        model_name, gaze_bias, subject, setsize), dpi=110)
                plt.close()
                waic_res = pm.waic(mtrace, pymc_model, scale='log')
                waic_df = pd.DataFrame({'WAIC': waic_res.waic,
                                        'WAIC_se': waic_res.waic_se,
                                        'p_WAIC': waic_res.p_waic,
                                        'WAIC_i': np.asarray(waic_res.waic_i).ravel(),
                                        'WAIC_scale': waic_res.waic_scale})
                waic_df.to_csv(
                    'results/waic/{}_{}/{}_{}_waic.csv'.format(
                        model_name, gaze_bias, subject, setsize))

            # simulate
            if not os.path.isfile(
                'results/predictions/{}_{}/{}_{}_prediction.csv'.format(
                    model_name, gaze_bias, subject, setsize)):
                print('\t\tSimulating data..')

                # extract parameter estimates
                mtrace_df = pd.read_csv(
                    'results/posterior_traces/{}_{}/{}_{}_mtrace.csv'.format(
                        model_name, gaze_bias, subject, setsize))
                estimates = dict()
                for parameter, precision in zip(model_parameters, to_round):
                    estimates[parameter] = mode(np.round(mtrace_df[parameter].values, precision))[0]

                # predict
                prediction = ind_model.predict(data=subject_data,
                                               gaze_data=subject_gaze_data,
                                               estimates=estimates,
                                               n_repeats=50)

                # store results
                prediction['setsize'] = setsize
                prediction.to_csv(
                   'results/predictions/{}_{}/{}_{}_prediction.csv'.format(
                     model_name, gaze_bias, subject, setsize))
                print('\t\t..done.')
