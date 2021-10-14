#!/usr/bin/env python
# coding: utf-8

# 

# # Purposes of this document
# 
# the reason I am doing this is that I want to make this really easy kind of thing.  

# # ideas

# In[3]:



#random_seeds = [0, 100, 1000]
#num_rep = len(random_seeds)


# apply leo's idea, we are gonna be suspicious everything, right. 

# In[4]:


import numpy as np

percent_high_SNR_noises = np.array([0.6])

num_noises = len(percent_high_SNR_noises)


percent_high_SNR_noises_labels = [f'{s:.2f}' for s in percent_high_SNR_noises]

print(percent_high_SNR_noises_labels)
print(num_noises)


# In[5]:


#the second idea is actually to prune the feature set
batch_to_prune = 50


# # Experimental setup related to the questions
# 
# this part should be configured to directly test the hypothesis put forward in the previous section
# 

# In[6]:


mean_firing_rate_low = 50
mean_firing_rate_high = 200


# In[7]:



np.set_printoptions(precision=2, suppress=True)

noise_mode = 'fixed_gaussian'
fixed_noise_level = 5#Hz

n_neurons = 32

n_neurons_per_group = 8



no_noise_neuron_ind = np.arange(n_neurons_per_group)
medium_noise_neuron_ind = np.arange(n_neurons_per_group, 2 * n_neurons_per_group)
low_noise_neuron_ind = np.arange(2*n_neurons_per_group, 3 * n_neurons_per_group)
high_noise_neuron_ind = np.arange(3 * n_neurons_per_group, 4* n_neurons_per_group)

neuron_type_indices_in_a_list = [
    high_noise_neuron_ind,
    medium_noise_neuron_ind,
    low_noise_neuron_ind,
    no_noise_neuron_ind
]

N_TYPES_OF_NEURONS = len(neuron_type_indices_in_a_list)


no_noise_neuron_list = np.full(n_neurons, False, dtype = bool)
low_noise_neuron_list = np.full(n_neurons, False, dtype = bool)
medium_noise_neuron_list = np.full(n_neurons, False, dtype = bool)
high_noise_neuron_list = np.full(n_neurons, False, dtype = bool)

with_noise_neuron_list = np.full(n_neurons, False, dtype = bool)



no_noise_neuron_list[no_noise_neuron_ind] = True
low_noise_neuron_list[low_noise_neuron_ind] = True
medium_noise_neuron_list[medium_noise_neuron_ind] = True
high_noise_neuron_list[high_noise_neuron_ind] = True

with_noise_neuron_list[low_noise_neuron_ind] = True
with_noise_neuron_list[medium_noise_neuron_ind] = True
with_noise_neuron_list[high_noise_neuron_ind] = True

neuron_type_bool_list = [
    high_noise_neuron_list,
    medium_noise_neuron_list,
    low_noise_neuron_list,
    no_noise_neuron_list,
]

print(neuron_type_bool_list)


percent_of_count = np.ones(n_neurons)[:, np.newaxis]

print(f'set up the variances in a list:')

percent_of_count[high_noise_neuron_ind] = 1 
percent_of_count[medium_noise_neuron_ind] = 1
percent_of_count[low_noise_neuron_ind] = 1
percent_of_count[no_noise_neuron_ind] = 0.8

print(f'we therefore know the number of neurons to be {n_neurons}')
print(percent_of_count)


# In[8]:


# make percent of count into a list 
percent_of_count_in_a_list = list()

for i in range(num_noises):
    percent_of_count = np.ones(n_neurons)[:, np.newaxis]


    percent_of_count[high_noise_neuron_ind] =  1
    percent_of_count[medium_noise_neuron_ind] = 1
    percent_of_count[low_noise_neuron_ind] = 1
    percent_of_count[no_noise_neuron_ind] = percent_high_SNR_noises[i]
    
    percent_of_count_in_a_list.append(percent_of_count)


# In[9]:



#for comparision
#for comparision

exp_conds = [ 'Full set', 'Iterative feature selection',]

print(f'we have experimental conditions {exp_conds}')

#setting clda parameters 
##learner: collects paird data at batch_sizes
RHO = 0.5
batch_size = 100


#assistor set up assist level
assist_level = (0.0, 0.0)

#learner and updater: actualy set up rho
UPDATER_BATCH_TIME = 1
UPDATER_HALF_LIFE = np.log(RHO)  * UPDATER_BATCH_TIME / np.log(0.5)


# In[10]:


# CHANGE: game mechanics: generate task params
N_TARGETS = 8
N_TRIALS = 800

NUM_EXP_COND = len(exp_conds) # how many experiments we are running. 
NUM_EXP = NUM_EXP_COND


# # Config the experiments
# 
# this section largely copyied and pasted from   
# bmi3d-sijia(branch)-bulti_in_experiemnts
# https://github.com/sijia66/brain-python-interface/blob/master/built_in_tasks/sim_task_KF.py

# ## load dependant libraries

# In[11]:


from afs_plotting import *


# In[12]:


# import libraries
# make sure these directories are in the python path., 
from bmimultitasks import SimBMIControlMulti, SimBMICosEncKFDec, BMIControlMultiNoWindow
from features import SaveHDF
from features.simulation_features import get_enc_setup, SimKFDecoderRandom,SimIntentionLQRController, SimClockTick
from features.simulation_features import SimHDF, SimTime

from riglib import experiment

from riglib.stereo_opengl.window import FakeWindow
from riglib.bmi import train


from behaviour_metrics import filter_state, sort_trials

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

# In[13]:


seq = SimBMIControlMulti.sim_target_seq_generator_multi(
N_TARGETS, N_TRIALS)

#create a second version of the tasks
seqs = itertools.tee(seq, NUM_EXP + 1)
target_seq = list(seqs[NUM_EXP])

seqs = seqs[:NUM_EXP]


SAVE_HDF = False
SAVE_SIM_HDF = True #this makes the task data available as exp.task_data_hist
DEBUG_FEATURE = False


#base_class = SimBMIControlMulti
base_class = BMIControlMultiNoWindow

#for adding experimental features such as encoder, decoder
feats = []
feats_2 = []


# goin

# ## Time control

# In[14]:


from simulation_features import TimeCountDown

feats.append(TimeCountDown)
feats_2.append(TimeCountDown)

total_exp_time = 1200# in seconds


# ## encoder
# 
# the cosine tuned encoder uses a poisson process, right
# https://en.wikipedia.org/wiki/Poisson_distribution
# so if the lambda is 1, then it's very likely 

# In[15]:


from features.simulation_features import get_enc_setup

ENCODER_TYPE = 'cosine_tuned_encoder_with_poisson_noise'

#neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'rot_90', n_neurons= n_neurons)


print(no_noise_neuron_list)
#multiply our the neurons
sim_C[with_noise_neuron_list] =  sim_C[with_noise_neuron_list]  * mean_firing_rate_low
sim_C[no_noise_neuron_list]  = sim_C[no_noise_neuron_list] * mean_firing_rate_high


print(sim_C.shape)

#set up intention feedbackcontroller
#this ideally set before the encoder
feats.append(SimIntentionLQRController)

#set up the encoder
from features.simulation_features import SimCosineTunedEncWithNoise
feats.append(SimCosineTunedEncWithNoise)


feats_2.append(SimIntentionLQRController)
feats_2.append(SimCosineTunedEncWithNoise)


# ## decoder setup

# In[16]:


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

# In[17]:




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

# In[18]:


from feature_selection_feature import FeatureTransformer, TransformerBatchToFit
from feature_selection_feature import FeatureSelector, LassoFeatureSelector, SNRFeatureSelector, IterativeFeatureSelector
from feature_selection_feature import IterativeFeatureRemoval, ReliabilityFeatureSelector

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

# ## (Check) config the experiment

# In[19]:


exp_feats = [feats, feats_2] 


# In[20]:


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


# In[21]:


percent_of_count_in_a_list


# In[22]:




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



print(f'we have got {len(kwargs_exps)} exps')


# ## make and initalize experiment instances

# In[23]:


seqs


# In[24]:



exps_reps = list()


for i,s in enumerate(seqs):

    f = exp_feats[i]
    Exp = experiment.make(base_class, feats=f)

    e = Exp(s, **kwargs_exps[i])

    exps_reps.append(e)

exps_np  = np.array(exps_reps, dtype = 'object')



#run the ini
for e in exps_np: 
    e.init()
    print('next')
    print()
    


# In[25]:




print(NUM_EXP)


# # Pre-experiment check: check the Kalman filter before training

# In[26]:


from afs_plotting import plot_prefered_directions

print('we replace the encoder using the weights')
print('assume, they are all randomly initialized get the first decoder')
print('get a handle to the first decoder')
first_decoder = exps_np[0].decoder
target_C = first_decoder.filt.C
target_Q = np.copy(first_decoder.filt.Q)

print()
diag_val = 10000
np.fill_diagonal(target_Q, diag_val)

#replace the decoder
for i,e in enumerate(exps_np):
    weights.change_target_kalman_filter_with_a_C_mat(e.decoder.filt, target_C, 
                                                     Q= target_Q, debug=False)
    e.select_decoder_features(e.decoder)
    e.record_feature_active_set(e.decoder)
    
print('we check the new decoder C matrix:')

figure_decoder_C, axs_decoder_C = plt.subplots(nrows=2, 
                               ncols=NUM_EXP, figsize = [GLOBAL_FIGURE_VERTICAL_SIZE * NUM_EXP, GLOBAL_FIGURE_VERTICAL_SIZE * 2],squeeze = False)
figure_decoder_C.suptitle('KF C Matrix Before Training ')

for i,e in enumerate(exps_np):
    C = e.decoder.filt.C
    plot_prefered_directions(C, ax = axs_decoder_C[0,i])
    #axs_decoder_C[0,i].set_title(exp_conds[i])


# 

# In[41]:


from graphviz import Digraph

sim_diagram = Digraph()

#TODO make this into a function 
sim_diagram.node('encoder', f'{exps_np[0].encoder}')
sim_diagram.node('decoder',f'{exps_np[0].decoder}')
sim_diagram.node('learner', f'{exps_np[0].learner} \n                  with a batch size of {batch_size} \n                  and a rho of {RHO}')
sim_diagram.node('updater', f'{exps_np[0].updater}')


sim_diagram.edge('encoder', 'decoder')
sim_diagram.edge('decoder', 'learner')
sim_diagram.edge('learner','updater')
sim_diagram.edge('updater','decoder')

sim_diagram


# # Experiment run: assemble into a complete loop

# ## actually running the experiments

# In[25]:


len(kwargs_exps)


# In[26]:


from feature_selection_feature import run_exp_loop
seed_index = -1

for i,e in enumerate(exps_np):
    
    np.random.seed(1000)
    
    
    run_exp_loop(e, **kwargs_exps[i])
        
    print()


# # Postprocessing the data for loading

# In[27]:


import seaborn as sns
sns.set_context('talk')


# In[28]:


num_rewards = [e.calc_state_occurrences('reward') for e in  exps_np]

num_rewards = np.array(num_rewards)


num_rewards


# ## declare defs and conventions

# In[29]:


FRAME_RATE = 60
INT_WINDOW_TIME = 10 # s for looking at sample raw data

# some conventions as we go down the loop
X_VEL_STATE_IND = 3
Y_VEL_STATE_IND = 5
X_POS_STATE_IND = 0
Y_POS_STATE_IND = 2

state_indices = [X_POS_STATE_IND,
                 Y_POS_STATE_IND,
                 X_VEL_STATE_IND,
                 Y_VEL_STATE_IND]
state_names = ['x pos ', 'y pos', 'x vel', 'y vel']


INT_WIN_SAMPLES = INT_WINDOW_TIME * FRAME_RATE


# ## Refactor out the data

# In[30]:


exps = exps_np

task_data_hist_np_all = [np.array(e.task_data_hist) for e in exps_np]
len(task_data_hist_np_all)
task_data_hist_np_all[0].dtype


# ## Finished time in seconds

# In[31]:


finished_times_in_seconds = [int(len(s)/FRAME_RATE) for s in task_data_hist_np_all]
finished_times_in_seconds


# # Post data analysis

# ## Overall  trial statistics succuss rate)

# In[32]:


total_exp_time


# In[33]:


state_log = exps[1].state_log 
state_log[-1][1] * 60


# In[34]:


import behaviour_metrics
import importlib
import matplotlib.colors as cm
import matplotlib as mpl

importlib.reload(behaviour_metrics)

window_length = 60 #s


reward_events_per_minute = [behaviour_metrics.calc_event_rate_from_state_log(e.state_log,'reward', 
                                                                             total_time = total_exp_time,
                                                                             window_length = window_length) for e in exps]
reward_events_per_minute = np.array(reward_events_per_minute) * 60 / window_length


import afs_plotting
importlib.reload(afs_plotting)
from afs_plotting import get_cmap


time_vec = np.arange(reward_events_per_minute.shape[1]) * window_length / 60

print(reward_events_per_minute.shape)

cmap = get_cmap(num_noises)
cmap_2 = get_cmap(num_noises, color = mpl.cm.Oranges)
cmap_3 = get_cmap(num_noises, color = mpl.cm.Reds)


for i in range(num_noises):
    plt.plot(time_vec, reward_events_per_minute[i].T,color = 'b')
    plt.plot(time_vec, reward_events_per_minute[i +  num_noises].T,color = 'r')
    
    
plt.ylabel('# of rewards per minute')
plt.title('Reward Rate')
plt.xlabel('Time (min)')
plt.legend(exp_conds)


# In[35]:


trial_dicts_all = []
dict_keys = ['cursor', #behaviour
             'ctrl_input', 'spike_counts', #encoder translates intended ctrl into spike counts
             'decoder_state']


for i in range(NUM_EXP):
    
    segmented_trials = behaviour_metrics.segment_trials_in_state_log(exps[i].state_log)

    task_data_hist_np = task_data_hist_np_all[i]
    trial_dict_0 = behaviour_metrics.sort_trials_use_segmented_log(segmented_trials, 
                               target_seq,
                               task_data_hist_np, dict_keys)
    
    trial_dicts_all.append(trial_dict_0)

for t in trial_dicts_all: print(len(t))


# ## Trajectory analysis

# In[36]:


figure_trajectory, axes_trajectory = plt.subplots(3, num_noises, figsize = (num_noises* GLOBAL_FIGURE_VERTICAL_SIZE, 
                                                  3  * GLOBAL_FIGURE_VERTICAL_SIZE),
                                                 squeeze = False) 


n_roi_trials = 60


CIRCLE_RADIUS = exps[0].target_radius

print(CIRCLE_RADIUS)

for i  in range(num_noises): 

    afs_plotting.add_center_out_grid(axes_trajectory[0,i], target_seq, CIRCLE_RADIUS)
    
    
    sample_trial = trial_dicts_all[i][n_roi_trials]
    trial_cursor_trajectory = sample_trial['cursor']
    
    afs_plotting.plot_trial_trajectory(axes_trajectory[0,i], trial_cursor_trajectory)
    axes_trajectory[0,i].set_title(percent_high_SNR_noises_labels[i])
    
    afs_plotting.add_center_out_grid(axes_trajectory[1,i], target_seq, CIRCLE_RADIUS)
    sample_trial = trial_dicts_all[i + num_noises][n_roi_trials]
    trial_cursor_trajectory = sample_trial['cursor']
    
    afs_plotting.plot_trial_trajectory(axes_trajectory[1,i], trial_cursor_trajectory)
    
    
figure_trajectory.tight_layout


# to make this point, 
# we should be lo

# In[37]:


print('finished trials:')

for i,e in  enumerate(exps): 
    reward_num = e.calc_state_occurrences('reward')
    print(f'{exp_conds[i]}: {reward_num} out of {N_TRIALS}')


# ## encoder
# 
# the job of the encoder is to directly encode intention into firing rates
# the direct measure is just pearson correlation coefficients between 
# the intentions and the firing rates

# ## decoder

# In[38]:


from afs_plotting import plot_prefered_directions

TEXT_OFFSET_VERTICAL = -0.2


figure_decoder_C.suptitle('KF C matrix before and after CLDA')

print('steady state tuning curves:')

for  i,e in enumerate(exps): 

    e = exps[i]
    C = e.decoder.filt.C

    plot_prefered_directions(C, ax = axs_decoder_C[1,i])
    axs_decoder_C[1,i].set_title(percent_high_SNR_noises_labels[i % num_noises])

figure_decoder_C


# In[39]:


figure_KF_C, axes_KF_C = plt.subplots(2, num_noises, figsize = (num_noises* GLOBAL_FIGURE_VERTICAL_SIZE, 
                                                  2 * GLOBAL_FIGURE_VERTICAL_SIZE),
                                     squeeze = False) 


for i in range(num_noises):
    e = exps[i]
    C = e.decoder.filt.C

    plot_prefered_directions(C, ax = axes_KF_C[0,i])
    axes_KF_C[0,i].set_title(f'{percent_high_SNR_noises_labels[i]}')
    
    e = exps[i + num_noises]
    C = e.decoder.filt.C

    plot_prefered_directions(C, ax = axes_KF_C[1,i])
    axs_decoder_C[1,i].set_title(f'{percent_high_SNR_noises_labels[i]}')
    

    
    


# ## Compare to the decoder

# In[40]:


import matplotlib.cm as cm
import importlib

import convergence_analysis
importlib.reload(convergence_analysis)
from convergence_analysis import calc_cosine_sim_bet_two_matrices, calc_cosine_to_target_matrix


figure_C, axes_C = plt.subplots(2,num_noises, figsize = (num_noises * GLOBAL_FIGURE_VERTICAL_SIZE, 
                                               2*GLOBAL_FIGURE_VERTICAL_SIZE),
                               sharey = True, squeeze = False)

time_vec = np.arange(0, total_exp_time + 1, batch_size * 0.1, ) 


cmap = get_cmap(num_noises)

for i in range(num_noises):
    
    e = exps[i]
    enc_directions = e.encoder.C
    dec_directions = np.array(e._used_C_mat_list)
    angles_hist = calc_cosine_to_target_matrix( dec_directions,enc_directions)
    active_angles = np.mean(angles_hist[:, with_noise_neuron_list], axis = 1)
    axes_C[0,i].plot(time_vec, active_angles, color = 'b')
    active_angles = np.mean(angles_hist[:, no_noise_neuron_list], axis = 1)
    axes_C[1,i].plot(time_vec, active_angles, color = 'b')
    
    axes_C[0,i].set_title(f'Rel. noise ratio {percent_high_SNR_noises_labels[i]}')
    
    
    e = exps[i + num_noises]
    enc_directions = e.encoder.C
    dec_directions = np.array(e._used_C_mat_list)
    
    angles_hist = calc_cosine_to_target_matrix( dec_directions,enc_directions)
    active_angles = np.mean(angles_hist[:, with_noise_neuron_list], axis = 1)
    axes_C[0,i].plot(time_vec, active_angles, color = 'y')
    active_angles = np.mean(angles_hist[:, no_noise_neuron_list], axis = 1)
    axes_C[1,i].plot(time_vec, active_angles, color = 'y')
    
    '''
    e = exps[i + 2 * num_noises]
    enc_directions = e.encoder.C
    dec_directions = np.array(e._used_C_mat_list)
    
    angles_hist = calc_cosine_to_target_matrix( dec_directions,enc_directions)
    active_angles = np.mean(angles_hist[:, with_noise_neuron_list], axis = 1)
    axes_C[0,i].plot(time_vec, active_angles, color = 'r')
    active_angles = np.mean(angles_hist[:, no_noise_neuron_list], axis = 1)
    axes_C[1,i].plot(time_vec, active_angles, color = 'r')
    
    '''


    
axes_C[0,0].set_ylabel('Angle (deg)')
axes_C[1,0].set_ylabel('Angle (deg)')

axes_C[0,0].set_xlim([0,100])
axes_C[1,0].set_xlim([0,100])


# # looking at K matrix

#  $K_i^T = \begin{bmatrix} P_{ix} & P_{iy} & V_{ix} & V_{iy} \end{bmatrix}$
# 
# 

# $r_i = norm [\frac{C[i,:]}{Q_{ii}}]$

# In[41]:


exps = exps_np


# In[42]:


figure_k_matrix, axes_k_matrix = plt.subplots(1, 2,
                                          figsize = (GLOBAL_FIGURE_VERTICAL_SIZE * 2,
                                                    GLOBAL_FIGURE_VERTICAL_SIZE * num_noises),
                                             sharey = True, sharex = True, squeeze = False)

for i in range(num_noises):
    
    K = exps[i].k_mat_params[-1].T
    plot_prefered_directions(K, ax  = axes_k_matrix[i, 0])
    axes_k_matrix[i, 0].set_title(percent_high_SNR_noises_labels[i])
    
    K = exps[i + num_noises].k_mat_params[-1].T
    plot_prefered_directions(K, ax  = axes_k_matrix[i, 1])
    axes_k_matrix[i, 1].set_title(percent_high_SNR_noises_labels[i])
    

figure_k_matrix.tight_layout()


# In[43]:


figure_encoder, axes_encoder = plt.subplots(1,2, figsize = (GLOBAL_FIGURE_VERTICAL_SIZE * 2,
                                                    GLOBAL_FIGURE_VERTICAL_SIZE))

def update_xlabels(ax):
    xlabels = [format(label, ',.0f') for label in ax.get_xticks()]
    ax.set_xticklabels(xlabels)

K = exps[i].k_mat_params[10].T / np.max(K)
plot_prefered_directions(K, ax = axes_encoder[0])
axes_encoder[0].set_title('Initial rand. directions')


e = exps[0]
C = e.encoder.C / np.max(C)
plot_prefered_directions(C, ax = axes_encoder[1])
axes_encoder[1].set_title('Encoder directions')


# # Feature analysis

# In[44]:


from matplotlib import colors


exps = exps_np
fig_feature_active_map, axes_feat_active_map = plt.subplots(1, NUM_EXP,
                                                            figsize = ( NUM_EXP* GLOBAL_FIGURE_VERTICAL_SIZE,
                                                                      GLOBAL_FIGURE_VERTICAL_SIZE),
                                                           )
axes_feat_active_map[0].set_ylabel('Learner Batch number')

#color true to yellow
cmap = colors.ListedColormap(['yellow'])

for i, exp in enumerate(exps):

    active_feat_heat_map = np.array(exp._active_feat_set_list, dtype = np.int32)
    
    #https://stackoverflow.com/questions/40985961/matplotlib-how-to-change-data-point-color-based-on-its-boolean-value-consisten
    active_feat_heat_map = np.ma.masked_where(active_feat_heat_map == False, active_feat_heat_map)
    
    a = axes_feat_active_map[i].imshow(active_feat_heat_map.T, cmap = cmap)
    axes_feat_active_map[i].grid(False)
    #color false to blue
    cmap.set_bad(color='blue')
    
    
    axes_feat_active_map[i].set_ylabel('Features')
    axes_feat_active_map[i].set_xlabel('Feature update count')


fig_feature_active_map.tight_layout()
#fig_feature_active_map.colorbar(a, ax=axes_feat_active_map.ravel().tolist())


# In[45]:


fig_feature_active_map_example, axes_feat_active_map_example = plt.subplots(1, 1,
                                                            figsize = ( GLOBAL_FIGURE_VERTICAL_SIZE,
                                                                      GLOBAL_FIGURE_VERTICAL_SIZE),
                                                           )
axes_feat_active_map[0].set_ylabel('Learner Batch number')

#color true to yellow
cmap = colors.ListedColormap(['yellow'])

i = 0
exp =  exps[i]

active_feat_heat_map = np.array(exp._active_feat_set_list, dtype = np.int32)

#https://stackoverflow.com/questions/40985961/matplotlib-how-to-change-data-point-color-based-on-its-boolean-value-consisten
active_feat_heat_map = np.ma.masked_where(active_feat_heat_map == False, active_feat_heat_map)

a = axes_feat_active_map_example.imshow(active_feat_heat_map.T, cmap = cmap)
axes_feat_active_map[i].grid(False)
#color false to blue
cmap.set_bad(color='blue')

axes_feat_active_map_example.set_ylabel('Features')
axes_feat_active_map_example.set_xlabel('Feature update count')
axes_feat_active_map_example.set_title('Full feature set')


# In[46]:


fig_feature_active_map_example, axes_feat_active_map_example = plt.subplots(1, 1,
                                                            figsize = ( GLOBAL_FIGURE_VERTICAL_SIZE,
                                                                      GLOBAL_FIGURE_VERTICAL_SIZE),
                                                           )
axes_feat_active_map[0].set_ylabel('Learner Batch number')

#color true to yellow
cmap = colors.ListedColormap(['yellow'])

i = -1
exp =  exps[i]

active_feat_heat_map = np.array(exp._active_feat_set_list, dtype = np.int32)

#https://stackoverflow.com/questions/40985961/matplotlib-how-to-change-data-point-color-based-on-its-boolean-value-consisten
active_feat_heat_map = np.ma.masked_where(active_feat_heat_map == False, active_feat_heat_map)

a = axes_feat_active_map_example.imshow(active_feat_heat_map.T, cmap = cmap)
axes_feat_active_map_example.grid(False)
#color false to blue
cmap.set_bad(color='blue')

axes_feat_active_map_example.set_ylabel('Features')
axes_feat_active_map_example.set_xlabel('Feature update count')
axes_feat_active_map_example.set_title('Consistent high SNR subset')


# ## Examine used K mat

# In[47]:


fig_Q, axes_Q = plt.subplots(1,2,figsize = (GLOBAL_FIGURE_VERTICAL_SIZE*2, GLOBAL_FIGURE_VERTICAL_SIZE) 
                             ,sharey = True)

cmap_q = get_cmap(num_noises)
cmap_q_1 = get_cmap(num_noises, mpl.cm.Oranges)
cmap_q_2 = get_cmap(num_noises, mpl.cm.Reds)

batch_time = 10

time_vec = np.arange(len(exp._used_Q_diag_list)) * batch_time


for i in range(num_noises):
    exp = exps[i]
    Q_list = np.array(exp._used_Q_diag_list)

    Q_diag_no_noise = np.mean(Q_list[:, no_noise_neuron_list], axis = 1)
    Q_diag_noise = np.mean(Q_list[:, with_noise_neuron_list], axis = 1)
    
    axes_Q[0].plot(time_vec, Q_diag_noise, c = 'b')
    axes_Q[1].plot(time_vec, Q_diag_no_noise, c = 'b')
    
    
    exp = exps[i+num_noises]
    Q_list = np.array(exp._used_Q_diag_list)

    Q_diag_no_noise = np.mean(Q_list[:, no_noise_neuron_list], axis = 1)
    Q_diag_noise = np.mean(Q_list[:, with_noise_neuron_list], axis = 1)
    
    axes_Q[0].plot(time_vec, Q_diag_noise, c = 'r')
    axes_Q[1].plot(time_vec, Q_diag_no_noise, c = 'r')
    

    

axes_Q[0].set_xlabel('Time (s)')
axes_Q[1].set_xlabel('Time (s)')

axes_Q[0].set_title('Noisy neurons')
axes_Q[1].set_title('High SNR neurons')
axes_Q[1].legend(exp_conds)


# In[48]:


figure_k_matrix, axes_k_matrix = plt.subplots(1,NUM_EXP,
                                          figsize = (GLOBAL_FIGURE_VERTICAL_SIZE * NUM_EXP,
                                                    GLOBAL_FIGURE_VERTICAL_SIZE))



for i,e in enumerate(exps):
    K = (e._used_K_mat_list[-1]).T
    print(K)

    plot_prefered_directions(K, ax  = axes_k_matrix[i])
    axes_k_matrix[i].set_title(exp_conds[i])
    


# ## examine used C mat

# In[49]:


updated_C_mat = np.array(exp._used_C_mat_list)


# ## examine Q matrix
# 

# In[50]:


q_list = np.array(exp._used_Q_diag_list)
q_list.shape


# In[51]:


i = 1

c_mat = updated_C_mat[i,:,:]


fig_c_mat, axes_c_mat = plt.subplots(1,2)
pos = axes_c_mat[0].imshow(c_mat)
fig_c_mat.colorbar(pos, ax=axes_c_mat[0])


#normalized
q_diag = q_list[i,:].reshape(-1,1)
q_diag_tile = np.tile(q_diag, (1,7))

pos = axes_c_mat[1].imshow(c_mat/q_diag_tile)

c_mat/q_diag_tile *  1000


# how do we show this is a good measure of the feature quality sort of thing?

# In[52]:


# plot out of the mean weight of convergence

C_norm = updated_C_mat[:,:, (X_VEL_STATE_IND, Y_VEL_STATE_IND)]
C_norm  = np.linalg.norm(C_norm, axis = 2)
C_norm_noisy = np.mean(C_norm[:, with_noise_neuron_list], axis = 1)
C_norm_clean = np.mean(C_norm[:, no_noise_neuron_list], axis = 1)

plt.plot(C_norm_noisy)
plt.plot(C_norm_clean)
plt.xlabel(f'Number of batch updates with a batch size of {batch_size}')
plt.title('For a fixed encoder, the decoder weights \n converge to final values within a couple of batches')
plt.legend(['Noisy', 'high SNR'])


# In[59]:


# Q value 
q_diag_noisy = np.mean(q_list[:, with_noise_neuron_list], axis = 1)
q_diag_clean = np.mean(q_list[:, no_noise_neuron_list], axis = 1)


plt.plot(q_diag_noisy)
plt.plot(q_diag_clean)
plt.xlabel(f'Number of batch updates with a batch size of {batch_size}')
#plt.title('For a fixed encoder ,\n diagnoal values of the Q matrix converge as the same rate as the C matrix ')
plt.legend(['Noisy', 'high SNR'])
plt.xlim((0,10))


# ## Q matrix

# In[54]:


q_list.shape


# In[58]:


num_points = 3
x = np.arange(num_points)

A = np.vstack([x, np.ones(len(x))]).T

r_slopes_Q_3pt = np.zeros(N_NEURONS)

for i in range(N_NEURONS):
    y = q_list[:num_points,i]
    m,c = np.linalg.lstsq(A, y, rcond = 0)[0]
    r_slopes_Q_3pt[i] = m

plt.plot(r_slopes_Q_3pt)
plt.xlabel('Neuron count')
plt.ylabel('Slopes of reduction in error (abs)')
plt.axvline(7, color = 'r')


# In[56]:


plt.hist(q_list[-1,:])


# In[57]:


plt.hist(r_slopes)
plt.xlabel('Slope of 3 point reliabilty index')
plt.ylabel('Counts')
plt.title("Clear separation of the two feature groups")


# In[ ]:


# now we can measure out the things

C_mat = updated_C_mat[:,:,(X_VEL_STATE_IND, Y_VEL_STATE_IND)]
q_list_tile = q_list[:,:, np.newaxis]
q_list_tile = np.tile(q_list_tile, (1,1,2))

q_measure = C_mat / q_list_tile

print(q_measure.shape)

'''

'''
q_measure_norm  = np.linalg.norm(q_measure, axis = 2)
q_measure_noisy= np.mean(q_measure_norm[:, with_noise_neuron_list], axis = 1)
q_measure_clean= np.mean(q_measure_norm[:, no_noise_neuron_list], axis = 1)


plt.plot(q_measure_noisy)
plt.plot(q_measure_clean)
plt.xlabel(f'Number of batch updates with a batch size of {batch_size}')
plt.title('Noisy normalized weights may be used for feature selection')
plt.legend(['Noisy', 'high SNR'])
plt.ylabel('Q normalzied weights')


# In[ ]:


time_snap_shots = [1, 2, 3, 5, 10, 16]

fig, axes = plt.subplots(1, len(time_snap_shots),
                        figsize = (5 * 4, 4))

for i,t in enumerate(time_snap_shots):
    axes[i].hist(q_measure_norm[t,:])
    axes[i].set_xlabel('Reliability index ')
    axes[i].set_ylabel('Counts')
    axes[i].set_title(f'Distribution at batch {t}')

fig.tight_layout()


# In[ ]:


q_list


# In[ ]:


from scipy import stats


p_values = list()

for i in range(2, len(q_measure_norm)):
    (statistic, pvalue) = stats.ks_2samp(q_list[1,:], q_list[i,:])
    p_values.append(pvalue)
    
pvalues = np.array(p_values)

plt.plot(pvalues)


# ## before we tackle with quality measure.
# we can first look at the convergence of the 

# In[ ]:


q_measure_norm_diff = np.diff(q_measure_norm, axis = 0)
plt.plot(q_measure_norm_diff[0,:])
plt.xlabel('Neuron count')
plt.ylabel('Difference in reliability index')
plt.title('Diff. of the Reliability index between the first two batches \n does not discrminate between the noisy features and the clean features')


# In[ ]:


num_points = 3
x = np.arange(num_points)

A = np.vstack([x, np.ones(len(x))]).T

r_slopes_3pt = np.zeros(N_NEURONS)

for i in range(N_NEURONS):
    y = q_measure_norm[:num_points,i]
    m,c = np.linalg.lstsq(A, y, rcond = 0)[0]
    r_slopes_3pt[i] = m

plt.plot(r_slopes_3pt)
plt.xlabel('Neuron count')
plt.ylabel('Slope of 3 point reliabilty index')
plt.title('Using the first 3 batches is more discriminative')


# In[ ]:


num_points = 10
x = np.arange(num_points)

A = np.vstack([x, np.ones(len(x))]).T

r_slopes = np.zeros(N_NEURONS)

for i in range(N_NEURONS):
    y = q_measure_norm[:num_points,i]
    m,c = np.linalg.lstsq(A, y, rcond = 0)[0]
    r_slopes[i] = m

plt.plot(r_slopes)
plt.xlabel('Neuron count')
plt.ylabel('Slope of 10 batch reliabilty index')
plt.title('Using the first 10 batches is even more discriminative')


# In[ ]:


plt.hist(r_slopes)
plt.xlabel('Slope of 3 point reliabilty index')
plt.ylabel('Counts')
plt.title("Clear separation of the two feature groups")


# $r_k =  

# In[ ]:


figure_q_measure, axes_q_measure = plt.subplots(2,3,
                                                figsize = (3 * GLOBAL_FIGURE_VERTICAL_SIZE, 2*GLOBAL_FIGURE_VERTICAL_SIZE))

axes_q_measure[0,0].plot(q_measure_norm_diff[0,:])
axes_q_measure[0,0].set_xlabel('Neuron count')
axes_q_measure[0,0].set_ylabel('Difference in reliability index')
axes_q_measure[0,0].set_title('2 batches')

axes_q_measure[1,0].hist(q_measure_norm_diff[0,:])
axes_q_measure[1,0].set_ylabel('Counts')
axes_q_measure[1,0].set_xlabel('Difference in the reliability index')


axes_q_measure[0,1].plot(r_slopes_3pt)
axes_q_measure[0,1].set_xlabel('Neuron count')
axes_q_measure[0,1].set_ylabel('Slopes of 3 point reliabilty index')
axes_q_measure[0,1].set_title('3 batches')



axes_q_measure[1,1].hist(r_slopes_3pt)
axes_q_measure[1,1].set_xlabel('Slopes of 3 point reliabilty index')
axes_q_measure[1,1].set_ylabel('Counts')


axes_q_measure[0,2].plot(r_slopes)
axes_q_measure[0,2].set_xlabel('Neuron count')
axes_q_measure[0,2].set_ylabel('Slopes of 10 point reliabilty index')
axes_q_measure[0,2].set_title('10 batches')

axes_q_measure[1,2].hist(r_slopes)
axes_q_measure[1,2].set_xlabel('Slopes of 10 batch reliabilty index')
axes_q_measure[1,2].set_ylabel('Counts')



plt.tight_layout()


# we can use a gaussian mixture model to speed up the calculation sort of thing. 

# In[ ]:


from scipy import stats

stats.ks_2samp(r_slopes_3pt, r_slopes)


# In[ ]:


stats.ks_2samp(q_measure_norm_diff[0,:],
               r_slopes_3pt)

