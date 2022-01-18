#!/usr/bin/env python
# coding: utf-8

# # Purpose of this simulation
# 

# # ideas
# import libraries
# make sure these directories are in the python path., 
from bmimultitasks import SimBMIControlMulti, SimBMICosEncKFDec, BMIControlMultiNoWindow, SimpleTargetCapture, SimpleTargetCaptureWithHold
from features import SaveHDF
from features.simulation_features import get_enc_setup, SimKFDecoderRandom,SimIntentionLQRController, SimClockTick
from features.simulation_features import SimHDF, SimTime

from riglib import experiment

from riglib.stereo_opengl.window import FakeWindow
from riglib.bmi import train

import weights

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import itertools #for identical sequences


# In[1]:


from numpy import random
import numpy as np

from simulation_setup_functions import *


# same across all experiments
N_TARGETS = 8
N_TRIALS = 2000 # set a maximum number of trials, this does not control the exp length, rather, the count down controls the experiment length


def run_iter_feat_addition(total_exp_time = 60, n_neurons = 32, fraction_snr = 0.25,
                           bimodal_weight = [0.5, 0.5],
                           norm_val = [50,  10], # sufficient statistics for the first gaussian peak,
                           norm_var_2 = [100, 10], # similarly, sufficient stats for the 2nd gaussian peak. 
                           fixed_noise_level = 5, 
                           noise_mode = 'fixed_gaussian',
                           percent_high_SNR_noises = np.arange(1, 0.9, -0.2),
                           DECODER_MODE = 'random',  # decoder mode selection
                           LEARNER_TYPE = 'feedback' , # to dumb or not dumb it is a question 'feedback'
                           UPDATER_TYPE = 'smooth_batch' , #none or "smooth_batch"
                           data_dump_folder = '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/trained_decoder/',
                           random_seed = 0,
                           ):
    

    ##################################################################################
    # set up file names for comparision
    exp_conds_add = [f'iter_{s}_{random_seed}_{n_neurons}' for s in percent_high_SNR_noises]
    exp_conds_keep = [f'same_{s}_{random_seed}_{n_neurons}' for s in percent_high_SNR_noises]
    exp_conds = [f'wo_FS_{s}_{random_seed}_{n_neurons}' for s in percent_high_SNR_noises]

    exp_conds.extend(exp_conds_add)
    exp_conds.extend(exp_conds_keep)

    NUM_NOISES = len(percent_high_SNR_noises)

    NUM_EXP = len(exp_conds) # how many experiments we are running. 

    print(f'we have experimental conditions {exp_conds}')

    ###############################################################################################
    # set up basic experiments

    from target_capture_task import ConcreteTargetCapture
    seq = ConcreteTargetCapture.out_2D()

    #create a second version of the tasks
    seqs = itertools.tee(seq, NUM_EXP + 1)
    target_seq = list(seqs[NUM_EXP])
    seqs = seqs[:NUM_EXP]


    SAVE_HDF = True
    SAVE_SIM_HDF = True #this makes the task data available as exp.task_data_hist
    DEBUG_FEATURE = False


    #base_class = SimBMIControlMulti
    base_class = SimpleTargetCaptureWithHold

    #for adding experimental features such as encoder, decoder
    feats = []
    feats_2 = []
    feats_set = [] # this is a going to be a list of lists 



    from simulation_features import TimeCountDown
    from features.sync_features import HDFSync
    feats.append(HDFSync)
    feats_2.append(HDFSync)
    feats.append(TimeCountDown)
    feats_2.append(TimeCountDown)


    ##########################################################################################################################
    # ## encoder and feature setup 
    from simulation_setup_functions  import get_enc_setup
    from simulation_setup_functions import generate_binary_feature_distribution, generate_binary_features_by_thresholding
    ENCODER_TYPE = 'cosine_tuned_encoder_with_poisson_noise'

    #neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
    N_NEURONS, N_STATES, sim_C, feature_weights = get_enc_setup(sim_mode = 'two_gaussian_peaks', 
                                               n_neurons= n_neurons,
                                               bimodal_weight= bimodal_weight,
                                               norm_var=norm_val,
                                               norm_var_2=norm_var_2)

    # this basically specifies the noise model 
    (percent_of_count_in_a_list, no_noise_neuron_ind, noise_neuron_ind, no_noise_neuron_list, noise_neuron_list)= \
        generate_binary_features_by_thresholding(percent_high_SNR_noises, norm_val, norm_var_2, feature_weights)

    #set up the encoder
    from features.simulation_features import SimCosineTunedEncWithNoise
    #set up intention feedbackcontroller
    #this ideally set before the encoder
    feats.append(SimIntentionLQRController)
    feats.append(SimCosineTunedEncWithNoise)
    feats_2.append(SimIntentionLQRController)
    feats_2.append(SimCosineTunedEncWithNoise)

    ##########################################################################################################################
    # ## decoder setup

    #take care the decoder setup
    if DECODER_MODE == 'random':
        feats.append(SimKFDecoderRandom)
        feats_2.append(SimKFDecoderRandom)
        print(f'{__name__}: set base class ')
        print(f'{__name__}: selected SimKFDecoderRandom \n')
    else: #defaul to a cosEnc and a pre-traind KF DEC
        from features.simulation_features import SimKFDecoderSup
        feats.append(SimKFDecoderSup)
        feats_2.append(SimKFDecoderSup)
        print(f'{__name__}: set decoder to SimKFDecoderSup\n')

    ##########################################################################################################################
    # ##  clda: learner and updater
    #setting clda parameters 
    ##learner: collects paired data at batch_sizes
    RHO = 0.5
    batch_size = 100

    #learner and updater: actualy set up rho
    UPDATER_BATCH_TIME = 1
    UPDATER_HALF_LIFE = np.log(RHO)  * UPDATER_BATCH_TIME / np.log(0.5)


    #you know what? 
    #learner only collects firing rates labeled with estimated estimates
    #we would also need to use the labeled data
    #now, we can set up a dumb/or not-dumb learner
    if LEARNER_TYPE == 'feedback':
        from features.simulation_features import SimFeedbackLearner
        feats.append(SimFeedbackLearner)
        feats_2.append(SimFeedbackLearner)
    else:
        from features.simulation_features import SimDumbLearner
        feats.append(SimDumbLearner)
        feats_2.append(SimDumbLearner)

    #to update the decoder.
    if UPDATER_TYPE == 'smooth_batch':
        from features.simulation_features import SimSmoothBatch
        feats.append(SimSmoothBatch)
        feats_2.append(SimSmoothBatch)
    else: #defaut to none 
        print(f'{__name__}: need to specify an updater')

    ############################################################################################################################
    # ## feature selector setup
    from feature_selection_feature import FeatureTransformer, TransformerBatchToFit
    from feature_selection_feature import FeatureSelector, LassoFeatureSelector, SNRFeatureSelector, IterativeFeatureSelector
    from feature_selection_feature import ReliabilityFeatureSelector


    #pass the real time limit on clock
    feats.append(FeatureSelector)
    feats_2.append(IterativeFeatureSelector)

    feature_x_meth_arg = [
        ('transpose', None ),
    ]

    kwargs_feature = dict()
    kwargs_feature = {
        'transform_x_flag':False,
        'transform_y_flag':False,
        'n_starting_feats': n_neurons,
        'n_states':  7,
        "train_high_SNR_time": 60
    }

    print('kwargs will be updated in a later time')
    print(f'the feature adaptation project is tracking {kwargs_feature.keys()} ')

    #######################################################################################################################
    #assistor set up assist level
    assist_level = (0.0, 0.0)


    ########################################################################################################################
    # combine experimental features
    exp_feats = [feats] * NUM_EXP

    e_f_2 = [feats_2] * NUM_EXP

    e_f_3 = [feats] * NUM_EXP

    exp_feats.extend(e_f_2)
    exp_feats.extend(e_f_3)

    if DEBUG_FEATURE: 
        from features.simulation_features import DebugFeature
        feats.append(DebugFeature)
        
    if SAVE_HDF: 
        feats.append(SaveHDF)
        feats_2.append(SaveHDF)
    if SAVE_SIM_HDF: 
        feats.append(SimHDF)
        feats_2.append(SimHDF)
        
        
    #pass the real time limit on clock
    feats.append(SimClockTick)
    feats.append(SimTime)

    feats_2.append(SimClockTick)
    feats_2.append(SimTime)

    kwargs_exps = list()

    for i in range(NUM_NOISES):
        d = dict()
        
        d['total_exp_time'] = total_exp_time
        
        d['assist_level'] = assist_level
        
        # feature set up
        d['feature_weights'] = feature_weights
        d['sim_C'] = sim_C

        
        d['noise_mode'] = noise_mode
        d['percent_noise'] = percent_of_count_in_a_list[i]
        d['fixed_noise_level'] = fixed_noise_level
        
        d['batch_size'] = batch_size
        
        d['batch_time'] = UPDATER_BATCH_TIME
        d['half_life'] = UPDATER_HALF_LIFE
        d['no_noise_neuron_ind'] = no_noise_neuron_ind
        d['noise_neuron_ind'] = noise_neuron_ind
        
        d.update(kwargs_feature)
        
        kwargs_exps.append(d)

    kwargs_exps_add = copy.deepcopy(kwargs_exps)
    kwargs_exps_start = copy.deepcopy(kwargs_exps)

    for k in kwargs_exps_add:
        
        k['init_feat_set'] = np.full(N_NEURONS, False, dtype = bool)
        k['init_feat_set'][no_noise_neuron_list] = True

    for k in kwargs_exps_start:
        
        k['init_feat_set'] = np.full(N_NEURONS, False, dtype = bool)
        k['init_feat_set'][no_noise_neuron_list] = True


    kwargs_exps.extend(kwargs_exps_add)
    kwargs_exps.extend(kwargs_exps_start)

    print(f'we have got {len(kwargs_exps)} exps')
    kwargs_exps[1]['init_feat_set']

    #########################################################################################################################
    # ## make and initalize experiment instances

    #seed the experiment
    np.random.seed(0)

    exps = list()#create a list of experiment

    for i,s in enumerate(seqs):
        #spawn the task
        f = exp_feats[i]
        Exp = experiment.make(base_class, feats=f)
        
        e = Exp(s, **kwargs_exps[i])
        exps.append(e)


    exps_np  = np.array(exps, dtype = 'object')



    ######################################################################################################################
    # initialize the experiments
    from simulation_setup_functions import get_KF_C_Q_from_decoder

    from feature_selection_feature import run_exp_loop

    WAIT_FOR_HDF_FILE_TO_STOP = 10

    for i,e in enumerate(exps):
        np.random.seed(random_seed)
        
        e.init()

        # save the decoder if it is the first one. 
        if i == 0:
            (target_C, target_Q) = get_KF_C_Q_from_decoder(e.decoder)
            
            weights.change_target_kalman_filter_with_a_C_mat(e.decoder.filt, target_C, 
                                                    Q= target_Q, debug=False)
        
        else:  # otherwise, just replace it.  
                weights.change_target_kalman_filter_with_a_C_mat(e.decoder.filt, target_C, 
                                                        Q= target_Q, debug=False)
                
        e.select_decoder_features(e.decoder)
        e.record_feature_active_set(e.decoder)
        
        #################################################################
        # actual experiment begins
        run_exp_loop(e, **kwargs_exps[i])
        
        e.hdf.stop()
        print(f'wait for {WAIT_FOR_HDF_FILE_TO_STOP}s for hdf file to save')
        time.sleep(WAIT_FOR_HDF_FILE_TO_STOP)
        
        
        e.save_feature_params()
        
        time.sleep(WAIT_FOR_HDF_FILE_TO_STOP)
        
        e.cleanup_hdf()
    

        e.sinks.reset()
        
        print(f'Finished running  {exp_conds[i]}')
        
        print()
    

    #####################################################################################################################
    # post experiment set up
    import shutil

    import os
    import subprocess

    for i,e in enumerate(exps): 


        import subprocess
        old = e.h5file.name
        new = data_dump_folder + exp_conds[i] +'.h5'
        process = "cp {} {}".format(old,new)
        
        print(process)
        subprocess.run(process, shell=True) # do not remember, assign shell value to True.


    import os
    import aopy
    import tables

    exp_data_all = list()
    exp_data_metadata_all = list()

    for i,e in enumerate(exp_conds):
        files = {

        'hdf':e+'.h5'

        }
        
        file_name = os.path.join(data_dump_folder, files['hdf'])

            
        # write in the exp processing files
        
        aopy.data.save_hdf(data_dump_folder, file_name, kwargs_exps[i], data_group="/feature_selection", append = True)
        
        with tables.open_file(file_name, mode = 'r') as f: print(f)
        
        try:
            d,m = aopy.preproc.parse_bmi3d(data_dump_folder, files)
        except:
            print(f'cannot parse {e}')



if __name__ == "__main__":
    run_iter_feat_addition(percent_high_SNR_noises=[0])
