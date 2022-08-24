#!/usr/bin/env python
# coding: utf-8

# # Purpose of this simulation
# 

# # ideas

# In[1]:



#for comparision
exp_conds = ['wo_feature selection']

for e in exp_conds: print(e)


# # Experimental setup related to the questions
# 
# this part should be configured to directly test the hypothesis put forward in the previous section
# 

# In[2]:


import numpy as np
np.set_printoptions(precision=2, suppress=True)

mean_firing_rate_low = 50
mean_firing_rate_high = 200
noise_mode = 'fixed_gaussian'
fixed_noise_level = 5 #Hz


# In[3]:




neuron_types = ['noisy', 'non_noisy']

n_neurons = 32
n_neurons_noisy_group = 24
n_neurons_no_noise_group = 8


noise_neuron_ind = np.arange(n_neurons_noisy_group)
no_noise_neuron_ind = np.arange(n_neurons_noisy_group, n_neurons_noisy_group + n_neurons_no_noise_group)

neuron_type_indices_in_a_list = [
    noise_neuron_ind, 
    no_noise_neuron_ind
]


noise_neuron_list = np.full(n_neurons, False, dtype = bool)
no_noise_neuron_list = np.full(n_neurons, False, dtype = bool)


noise_neuron_list[noise_neuron_ind] = True
no_noise_neuron_list[no_noise_neuron_ind] = True



neuron_type_bool_list = [
    noise_neuron_list,
    no_noise_neuron_list,
]

N_TYPES_OF_NEURONS = 2

print('We have two types of indices: ')
for t,l in enumerate(neuron_type_indices_in_a_list): print(f'{neuron_types[t]}:{l}')


# In[4]:


percent_of_count = np.ones(n_neurons)[:, np.newaxis]
print(f'set up the variances in a list:')

percent_of_count[noise_neuron_ind] =  1
percent_of_count[no_noise_neuron_ind] = 1

print(f'we therefore know the number of neurons to be {n_neurons}')


# In[5]:


# CHANGE: game mechanics: generate task params
N_TARGETS = 8
N_TRIALS = 200

NUM_EXP = len(exp_conds) # how many experiments we are running. 


# # Config the experiments
# 
# this section largely copyied and pasted from   
# bmi3d-sijia(branch)-bulti_in_experiemnts
# https://github.com/sijia66/brain-python-interface/blob/master/built_in_tasks/sim_task_KF.py

# ## load dependant libraries

# In[6]:


GLOBAL_FIGURE_VERTICAL_SIZE = 4


# In[7]:


# import libraries
# make sure these directories are in the python path., 
from bmimultitasks import SimBMIControlMulti, SimBMICosEncKFDec, BMIControlMultiNoWindow
from features import SaveHDF
from features.simulation_features import get_enc_setup, SimKFDecoderRandom,SimIntentionLQRController, SimClockTick
from features.simulation_features import SimHDF, SimTime

from riglib import experiment

from riglib.stereo_opengl.window import FakeWindow
from riglib.bmi import train


from behaviour_metrics import  filter_state, sort_trials

from weights import calc_p_values_for_spike_batches_use_intended_kin
from weights import calc_single_batch_p_values_by_fitting_kinematics_to_spike_counts
import weights

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import itertools #for identical sequences

np.set_printoptions(precision=2, suppress=True)


# ##  behaviour and task setup

# In[8]:


#seq = SimBMIControlMulti.sim_target_seq_generator_multi(
#N_TARGETS, N_TRIALS)

from target_capture_task import ConcreteTargetCapture
seq = ConcreteTargetCapture.centerout_2D()

#create a second version of the tasks
seqs = itertools.tee(seq, NUM_EXP + 1)
target_seq = list(seqs[NUM_EXP])

seqs = seqs[:NUM_EXP]


SAVE_HDF = True
SAVE_SIM_HDF = True #this makes the task data available as exp.task_data_hist
DEBUG_FEATURE = False


#base_class = SimBMIControlMulti
base_class = BMIControlMultiNoWindow

#for adding experimental features such as encoder, decoder
feats = []
feats_2 = []
feats_set = [] # this is a going to be a list of lists 


# ## Additional task setup

# In[9]:


from simulation_features import TimeCountDown
from features.sync_features import HDFSync


feats.append(HDFSync)
feats_2.append(HDFSync)

feats.append(TimeCountDown)
feats_2.append(TimeCountDown)

total_exp_time = 600# in seconds


# ## encoder
# 
# the cosine tuned encoder uses a poisson process, right
# https://en.wikipedia.org/wiki/Poisson_distribution
# so if the lambda is 1, then it's very likely 

# In[10]:


from features.simulation_features import get_enc_setup

ENCODER_TYPE = 'cosine_tuned_encoder_with_poisson_noise'

#neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'rand', n_neurons= n_neurons)


print(no_noise_neuron_list)
#multiply our the neurons
sim_C[noise_neuron_list] =  sim_C[noise_neuron_list]  * mean_firing_rate_low
sim_C[no_noise_neuron_list]  = sim_C[no_noise_neuron_list] * mean_firing_rate_high


print(sim_C)

#set up intention feedbackcontroller
#this ideally set before the encoder
feats.append(SimIntentionLQRController)

#set up the encoder
from features.simulation_features import SimCosineTunedEncWithNoise
feats.append(SimCosineTunedEncWithNoise)


feats_2.append(SimIntentionLQRController)
feats_2.append(SimCosineTunedEncWithNoise)


# ## decoder setup

# In[11]:


#clda on random 
DECODER_MODE = 'random' # random 

   #take care the decoder setup
if DECODER_MODE == 'random':
    feats.append(SimKFDecoderRandom)
    feats_2.append(SimKFDecoderRandom)
    print(f'{__name__}: set base class ')
    print(f'{__name__}: selected SimKFDecoderRandom \n')
else: #defaul to a cosEnc and a pre-traind KF DEC
    from features.simulation_features import SimKFDecoderSup
    feats.append(SimKFDecoderSup)
    feats_2.append(SimKFDecoderRandom)
    print(f'{__name__}: set decoder to SimKFDecoderSup\n')


# ##  clda: learner and updater

# In[12]:



#setting clda parameters 
##learner: collects paired data at batch_sizes
RHO = 0.5
batch_size = 100

#learner and updater: actualy set up rho
UPDATER_BATCH_TIME = 1
UPDATER_HALF_LIFE = np.log(RHO)  * UPDATER_BATCH_TIME / np.log(0.5)



LEARNER_TYPE = 'feedback' # to dumb or not dumb it is a question 'feedback'
UPDATER_TYPE = 'smooth_batch' #none or "smooth_batch"


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


# ## feature selector setup

# In[13]:


from feature_selection_feature import FeatureTransformer, TransformerBatchToFit
from feature_selection_feature import FeatureSelector, LassoFeatureSelector, SNRFeatureSelector, IterativeFeatureSelector
from feature_selection_feature import ReliabilityFeatureSelector


#pass the real time limit on clock
feats.append(FeatureSelector)
feats_2.append(ReliabilityFeatureSelector)


feature_x_meth_arg = [
    ('transpose', None ),
]

kwargs_feature = dict()
kwargs_feature = {
    'transform_x_flag':True,
    'transform_y_flag':True,
    'feature_x_transformer':FeatureTransformer(feature_x_meth_arg),
    'feature_y_transformer':TransformerBatchToFit(),
    'n_starting_feats': n_neurons,
    'n_states':  7
}

print('kwargs will be updated in a later time')
print(f'the feature adaptation project is tracking {kwargs_feature.keys()} ')


# ## assistor setup

# In[14]:


#assistor set up assist level
assist_level = (0.0, 0.0)


# ## (Check) config the experiment

# In[15]:


#exp_feats = [feats, feats_2, feats]
exp_feats = [feats]


# In[16]:


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


# In[17]:


kwargs_exps = list()

for i in range(NUM_EXP):
    d = dict()
    
    d['total_exp_time'] = total_exp_time
    d['assist_level'] = assist_level
    d['sim_C'] = sim_C
    
    d['noise_mode'] = noise_mode
    d['percent_noise'] = percent_of_count
    d['fixed_noise_level'] = fixed_noise_level
    
    d['batch_size'] = batch_size
    
    d['batch_time'] = UPDATER_BATCH_TIME
    d['half_life'] = UPDATER_HALF_LIFE
    
    
    d.update(kwargs_feature)
    
    kwargs_exps.append(d)

#kwargs_exps[1]['init_feat_set'] = np.full(N_NEURONS, True, dtype = bool)




# ## make and initalize experiment instances

# In[18]:


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
    
#run the ini
for e in exps_np: 
    e.init()
    print('next')
    print()


# # Pre-experiment check: check the Kalman filter before training

# In[19]:


# we plot the encoder directions
from afs_plotting import plot_prefered_directions

figure_encoder_direction, axes_encoder = plt.subplots()

encoder_mat_C = exps_np[0].encoder.C


plot_prefered_directions(encoder_mat_C, ax = axes_encoder)
axes_encoder.set_title('Distributions of encoder preferred directions')
axes_encoder.set_xlabel('C matrix weights \n a lot of vectors are clustered around ')


# In[20]:




print('we replace the encoder using the weights')
print('assume, they are all randomly initialized get the first decoder')

first_decoder = exps_np[0].decoder
target_C = first_decoder.filt.C
target_Q = np.copy(first_decoder.filt.Q)

print()
diag_val = 10000
np.fill_diagonal(target_Q, diag_val)

    
#replace the decoder
for i,e in enumerate(exps):
    weights.change_target_kalman_filter_with_a_C_mat(e.decoder.filt, target_C, 
                                                     Q= target_Q, debug=False)
    e.select_decoder_features(e.decoder)
    e.record_feature_active_set(e.decoder)
    


# In[21]:


print('we check the new decoder C matrix:')

figure_decoder_C, axs_decoder_C = plt.subplots(nrows=2, 
                               ncols=NUM_EXP, figsize = [GLOBAL_FIGURE_VERTICAL_SIZE * NUM_EXP, GLOBAL_FIGURE_VERTICAL_SIZE * 2],squeeze = False)
figure_decoder_C.suptitle('KF C Matrix Before Training ')

for i,e in enumerate(exps):
    C = e.decoder.filt.C
    plot_prefered_directions(C, ax = axs_decoder_C[0,i])
    axs_decoder_C[0,i].set_title(exp_conds[i])


# # Experiment run: assemble into a complete loop

# ## actually running the experiments

# In[22]:


from feature_selection_feature import run_exp_loop
for i,e in enumerate(exps):
    np.random.seed(0)
    run_exp_loop(e, **kwargs_exps[i])
    e.hdf.stop()

    time.sleep(10)
    
    e.cleanup_hdf()
    print(f'Finished running  {exp_conds[i]}')
    print()
    print()



import shutil
import os
for e in exps: 
    shutil.move(e.h5file.name, os.getcwd() +'\\'+ exp_conds[i]+'.h5')
    print(os.getcwd() + exp_conds[i])




