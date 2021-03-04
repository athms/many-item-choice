#!/usr/bin/env python3

import sys, os
import numpy as np
import pandas as pd
from scipy.stats import mode
import pymc3 as pm
import matplotlib.pyplot as plt

import models
from utils.general import make_sure_path_exists
from utils.stats import check_convergence


def return_model(model_name):
    # model settings
    if model_name == 'independent_accumulation':
        model = models.independent_accumulation
        model_parameters = ['v', 'gamma', 'zeta', 's']
    elif model_name == 'GLAM':
        model = models.GLAM
        model_parameters = ['v', 'gamma', 'zeta', 's', 'tau']
    elif model_name == 'probabilistic_satisficing':
        model = models.probabilistic_satisficing
        model_parameters = ['v', 'alpha', 'gamma', 'zeta', 'tau']
    else:
        raise NameError('Invalid model name')
    return model, model_parameters


if __name__ == '__main__':

    # determine simulation hyper-parameters
    setsize = 9
    subjects = np.arange(49)
    model_names = ['probabilistic_satisficing', 'independent_accumulation', 'GLAM']
    n_sim_subjects = 10
    np.random.seed(1)
    sim_subjects = np.random.choice(subjects, n_sim_subjects, replace=False)
    n_draws = 5000

    # directories
    data_dir = '../data/'
    results_dir = '../results/'

    # make sure output dir exists
    make_sure_path_exists(results_dir+'model_recovery/simulated_data/')
    make_sure_path_exists(results_dir+'model_recovery/waic/')
    make_sure_path_exists(results_dir+'model_recovery/posterior_traces/')

    # set rounding for model parameters
    to_round = {'probabilistic_satisficing': [8, 8, 2, 2, 2],
                'independent_accumulation': [6, 2, 2, 4],
                'GLAM': [6, 2, 2, 4, 2]}

    # iterate over models
    for mi, gen_model_name in enumerate(model_names):

        # set gen model parameters
        gen_gaze_bias = 'with_active_gaze'
        gen_model, gen_model_parameters = return_model(gen_model_name)

        print('\n\n-----')
        print('Generating model: {}_{}'.format(gen_model_name, gen_gaze_bias))

        # iterate over reps
        for sim_sub in sim_subjects:
            print('\n\tGenerating subject: {}'.format(sim_sub))

            # load data
            print('\tLoading empirical data..')
            full_data = pd.read_csv(data_dir+'summary_files/setsize-{}_desc-data.csv'.format(setsize))
            sub_data = full_data[full_data['subject']==sim_sub].copy()
            sub_gaze_data = pd.read_csv(data_dir+'subject_files/sub-{}_setsize-{}_desc-gazes.csv'.format(sim_sub, setsize))

            # get par estimates
            print('\tLoading parameter estimates..')
            trace_file = results_dir+'posterior_traces/{}_{}/sub-{}_setsize-{}_desc-mtrace.csv'.format(
                    gen_model_name, gen_gaze_bias, sim_sub, setsize)
            if os.path.isfile(trace_file):
                sub_trace_df = pd.read_csv(trace_file)
                sub_est = dict()
                for parameter, precision in zip(gen_model_parameters, to_round[gen_model_name]):
                    sub_est[parameter] = mode(np.round(sub_trace_df[parameter].values, precision))[0]

            # define simulation results filepath
            sim_res_filepath = results_dir+'model_recovery/simulated_data/sub-{}_gen-{}_desc-simulated_data.csv'.format(
                sim_sub, '-'.join((gen_model_name+'_'+gen_gaze_bias).split('_')))
            if not os.path.isfile(sim_res_filepath):

                # simulate
                np.random.seed(mi+sim_sub)
                print('\tSimulating data..')
                sim_data = gen_model.predict(data=sub_data,
                                             gaze_data=sub_gaze_data,
                                             estimates=sub_est,
                                             n_repeats=1)
                # add missing info
                sim_data['setsize'] = setsize
                for col in sub_data.columns:
                    if col not in sim_data.columns:
                        sim_data[col] = sub_data[col].values

                # add gen. parameters to simulated data & save
                for parameter in sub_est:
                    sim_data[parameter] = sub_est[parameter][0]
                sim_data.to_csv(sim_res_filepath, index=False)

            else:
                print('\tRestoring simulated data from: '+sim_res_filepath)
                sim_data = pd.read_csv(sim_res_filepath)

            # fit each model
            # define WAIC results filepath
            waic_res_file = results_dir+'model_recovery/waic/sub-{}_gen-{}_desc-rec_waic.csv'.format(
                sim_sub, '-'.join((gen_model_name+'_'+gen_gaze_bias).split('_')))
            if not os.path.isfile(waic_res_file):
                WAIC = pd.DataFrame()
            else:
                print('\tRestoring WAIC results from: '+waic_res_file)
                WAIC = pd.read_csv(waic_res_file)

            # iterate over recovery models
            for rmi, rec_model_name in enumerate(model_names):

                # get rec model
                rec_gaze_bias = 'with_active_gaze'
                rec_model, rec_model_parameters = return_model(rec_model_name)
                print('\tRecovery model: {}_{}'.format(rec_model_name, rec_gaze_bias))

                # set switch
                fit_model = True

                # define output trace filepath
                trace_res_file = results_dir+'model_recovery/posterior_traces/sub-{}_gen-{}_rec-{}_desc-mtrace.csv'.format(
                    sim_sub,
                    '-'.join((gen_model_name+'_'+gen_gaze_bias).split('_')),
                    '-'.join((rec_model_name+'_'+rec_gaze_bias).split('_')))
                # check if model WAIC exists already
                if 'rec_model' in WAIC.columns:
                    if '{}_{}'.format(rec_model_name, rec_gaze_bias) in WAIC['rec_model'].values:
                        print('\tWAIC: {}'.format(
                            WAIC[WAIC['rec_model']=='{}_{}'.format(
                                rec_model_name, rec_gaze_bias)]['WAIC'].values[0]))
                        fit_model = False

                if fit_model:
                    # make model
                    print('\tMaking pymc3 model..')
                    rec_pymc_model = rec_model.make_ind_model(sim_data, subject_gaze_data=sub_gaze_data)

                    # fit model
                    print('\tSampling posterior..')
                    tries = 0
                    model_converged = False
                    while (not model_converged) and (tries < 50):
                        with rec_pymc_model:
                            rec_mtrace = pm.sample(draws=n_draws, tune=n_draws+tries*n_draws,
                                                   cores=2, chains=2,
                                                   step=pm.Metropolis(),
                                                   random_seed=np.int(mi+sim_sub+rmi+tries))
                            model_converged = check_convergence(pm.summary(rec_mtrace),
                                parameters=rec_model_parameters)
                            tries += 1

                    # compute WAIC
                    print('\tComputing WAIC..')
                    rec_waic_res = pm.waic(rec_mtrace, rec_pymc_model, scale='log')
                    rec_waic_df = pd.DataFrame({'rec_model': '{}_{}'.format(rec_model_name, rec_gaze_bias),
                                                'WAIC': rec_waic_res.waic,
                                                'WAIC_se': rec_waic_res.waic_se,
                                                'WAIC_scale': rec_waic_res.waic_scale},
                                                index=[rmi])
                    WAIC = WAIC.append(rec_waic_df)
                    print('\tWAIC: {}'.format(rec_waic_df['WAIC']))

                    # save trace
                    print('\tSaving trace to {}'.format(trace_res_file))
                    _ = pm.trace_to_dataframe(rec_mtrace).to_csv(trace_res_file, index=False)

                # save WAIC
                WAIC.to_csv(waic_res_file, index=False)

            # readout best fitting model
            best_fitting_model = WAIC[WAIC['WAIC'] == WAIC['WAIC'].max()]['rec_model'].values[0]
            print('\tBest fitting model: {}'.format(best_fitting_model))
