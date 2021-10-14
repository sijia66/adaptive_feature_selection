#!/usr/bin/env python
# coding: utf-8

# # Purposes of this document
# 
# set up tracking linear weights tools
# 
# the observation fit a
# 
# firing_rates = C x states +  Q
# 
# we need to track the weights change in C
# as well as how the goodness of the fit using

# # ideas

# 

# # Experimental setup related to the questions
# 
# this part should be configured to directly test the hypothesis put forward in the previous section
# 

# In[1]:


from features.simulation_features import get_enc_setup

#neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'std')

print('examine the encoder weights distribution')
print('We expect the only the first four neurons carry information:')
print(sim_C)


# In[2]:


import numpy as np

#encoder mean firing rate
neuron_firing_rates  = [100, 100]

#for comparision
exp_conds = [f'enc. mean FR:{b} Hz' for b in neuron_firing_rates]


#setting clda parameters 
##learner: collects paird data at batch_sizes
RHO = 0.5
batch_size = 50


#assistor set up assist level
assist_level = (0.05, 0.0)

#learner and updater: actualy set up rho
UPDATER_BATCH_TIME = 1
UPDATER_HALF_LIFE = np.log(RHO)  * UPDATER_BATCH_TIME / np.log(0.5)


# In[3]:


# CHANGE: game mechanics: generate task params
N_TARGETS = 8
N_TRIALS = 80

NUM_EXP = len(exp_conds) # how many experiments we are running. 


# # Config the experiments
# 
# this section largely copyied and pasted from   
# bmi3d-sijia(branch)-bulti_in_experiemnts
# https://github.com/sijia66/brain-python-interface/blob/master/built_in_tasks/sim_task_KF.py

# ## load dependant libraries

# In[4]:


# import libraries
# make sure these directories are in the python path., 
from bmimultitasks import SimBMIControlMulti, SimBMICosEncKFDec, BMIControlMultiNoWindow
from features import SaveHDF
from features.simulation_features import get_enc_setup, SimKFDecoderRandom, SimCosineTunedEnc,SimIntentionLQRController, SimClockTick
from features.simulation_features import SimHDF, SimTime

from riglib import experiment

from riglib.stereo_opengl.window import FakeWindow
from riglib.bmi import train

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

# In[5]:


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


# ## encoder
# 
# the cosine tuned encoder uses a poisson process, right
# https://en.wikipedia.org/wiki/Poisson_distribution
# so if the lambda is 1, then it's very likely 

# In[6]:


ENCODER_TYPE = 'cosine_tuned_encoder'



#actually multiply out the firing rates. 
sim_C_all = [sim_C * nfr for nfr in neuron_firing_rates]


#set up intention feedbackcontroller
#this ideally set before the encoder
feats.append(SimIntentionLQRController)

#set up the encoder
if ENCODER_TYPE == 'cosine_tuned_encoder' :
    feats.append(SimCosineTunedEnc)
    print(f'{__name__}: selected SimCosineTunedEnc\n')


# ## decoder setup

# In[7]:


#clda on random 
DECODER_MODE = 'random' # random 

   #take care the decoder setup
if DECODER_MODE == 'random':
    feats.append(SimKFDecoderRandom)
    print(f'{__name__}: set base class ')
    print(f'{__name__}: selected SimKFDecoderRandom \n')
else: #defaul to a cosEnc and a pre-traind KF DEC
    from features.simulation_features import SimKFDecoderSup
    feats.append(SimKFDecoderSup)
    print(f'{__name__}: set decoder to SimKFDecoderSup\n')


# ##  clda: learner and updater

# In[8]:




LEARNER_TYPE = 'feedback' # to dumb or not dumb it is a question 'feedback'
UPDATER_TYPE = 'smooth_batch' #none or "smooth_batch"


#you know what? 
#learner only collects firing rates labeled with estimated estimates
#we would also need to use the labeled data
#now, we can set up a dumb/or not-dumb learner
if LEARNER_TYPE == 'feedback':
    from features.simulation_features import SimFeedbackLearner
    feats.append(SimFeedbackLearner)
else:
    from features.simulation_features import SimDumbLearner
    feats.append(SimDumbLearner)

#to update the decoder.
if UPDATER_TYPE == 'smooth_batch':
    from features.simulation_features import SimSmoothBatch
    feats.append(SimSmoothBatch)
else: #defaut to none 
    print(f'{__name__}: need to specify an updater')
    


# ## feature adaptor setup
# 
# 

# In[9]:


from feature_selection_feature import FeatureTransformer, TransformerBatchToFit
from feature_selection_feature import FeatureSelector, LassoFeatureSelector


#pass the real time limit on clock

feats.append(LassoFeatureSelector)

feature_x_meth_arg = [
    ('transpose', None ),
]

kwargs_feature = dict()
kwargs_feature = {
    'transform_x_flag':True,
    'transform_y_flag':True,
    'feature_x_transformer':FeatureTransformer(feature_x_meth_arg),
    'feature_y_transformer':TransformerBatchToFit()
}

print('kwargs will be updated to a later time')

# ## assistor setup

# ## (Check) config the experiment

# In[10]:


if DEBUG_FEATURE: 
    from features.simulation_features import DebugFeature
    feats.append(DebugFeature)
    
if SAVE_HDF: feats.append(SaveHDF)
if SAVE_SIM_HDF: feats.append(SimHDF)
    
    
#pass the real time limit on clock
feats.append(SimClockTick)
feats.append(SimTime)


kwargs_exps = list()

for i in range(NUM_EXP):
    d = dict()
    
    d['assist_level'] = assist_level
    d['sim_C'] = sim_C_all[i]
    d['batch_size'] = batch_size
    
    d['batch_time'] = UPDATER_BATCH_TIME
    d['half_life'] = UPDATER_HALF_LIFE
    
    #d.update(kwargs_feature)
    kwargs_exps.append(d)
    
    


kwargs_exps


# ## make and initalize experiment instances

# In[11]:


#seed the experiment
np.random.seed(0)

#spawn the task
Exp = experiment.make(base_class, feats=feats)

exps = list()#create a list of experiment

for i,s in enumerate(seqs):
    e = Exp(s, **kwargs_exps[i])
    exps.append(e)

#run the ini
for e in exps: e.init()


# # Pre-experiment check: check the Kalman filter before training

# In[ ]:


print('we replace the encoder using the weights')
print('assume, they are all randomly initialized get the first decoder')
print('get a handle to the first decoder')
first_decoder = exps[0].decoder
target_C = first_decoder.filt.C
    
#replace the decoder
for i,e in enumerate(exps):
    weights.change_target_kalman_filter_with_a_C_mat(e.decoder.filt, target_C, debug=False)
    
print('we check the new decoder C matrix:')

decoder_c_figure, axs = plt.subplots(nrows=1, 
                               ncols=NUM_EXP, figsize = [12,4])
decoder_c_figure.suptitle('Decoder C matrix ')

for i,e in enumerate(exps):
    e.decoder.plot_C(ax = axs[i])
    axs[i].set_title(exp_conds[i])


# # Experiment run: assemble into a complete loop

# ##  define the function

# In[ ]:


#make this into a loop

def run_exp_loop(exp,  **kwargs):
        # riglib.experiment: line 597 - 601
    #exp.next_trial = next(exp.gen)
    # -+exp._parse_next_trial()np.arraynp.array


    # we need to set the initial state
    # per fsm.run:  line 138


    # Initialize the FSM before the loop
    exp.set_state(exp.state)
    
    finished_trials = exp.calc_state_occurrences('wait')
    print(f'finished: {finished_trials}')


    while exp.state is not None:

        # exp.fsm_tick()

        ### Execute commands#####
        exp.exec_state_specific_actions(exp.state)

        ###run the bmi loop #####
        # _cycle

        # bmi feature extraction, eh
        #riglib.bmi: 1202
        feature_data = exp.get_features()

        # Determine the target_state and save to file
        current_assist_level = exp.get_current_assist_level()
        target_state = exp.get_target_BMI_state(exp.decoder.states)

        # Determine the assistive control inputs to the Decoder
        #update assistive control level
        exp.update_level()
        if np.any(current_assist_level) > 0:
            current_state = exp.get_current_state()

            if target_state.shape[1] > 1:
                assist_kwargs = exp.assister(current_state, 
                                             target_state[:,0].reshape(-1,1), 
                                             current_assist_level, mode= exp.state)
            else:
                assist_kwargs = exp.assister(current_state, 
                                              target_state, 
                                              current_assist_level, 
                                              mode= exp.state)

            kwargs.update(assist_kwargs)
            
        

        # decode the new features
        # riglib.bmi.bmiloop: line 1245
        neural_features = feature_data[exp.extractor.feature_type]

        # call decoder.
        #tmp = exp.call_decoder(neural_features, target_state, **kwargs)
        neural_obs = neural_features
        learn_flag = exp.learn_flag
        task_state = exp.state

        n_units, n_obs = neural_obs.shape
        # If the target is specified as a 1D position, tile to match
        # the number of dimensions as the neural features
        if np.ndim(target_state) == 1 or (target_state.shape[1] == 1 and n_obs > 1):
            target_state = np.tile(target_state, [1, n_obs])

        decoded_states = np.zeros([exp.bmi_system.decoder.n_states, n_obs])
        update_flag = False

        for k in range(n_obs):
            neural_obs_k = neural_obs[:, k].reshape(-1, 1)
            target_state_k = target_state[:, k]

            # NOTE: the conditional below is *only* for compatibility with older Carmena
            # lab data collected using a different MATLAB-based system. In all python cases,
            # the task_state should never contain NaN values.
            if np.any(np.isnan(target_state_k)):
                task_state = 'no_target'

            #################################
            # Decode the current observation
            #################################
            decodable_obs, decode = exp.bmi_system.feature_accumulator(
                neural_obs_k)
            if decode:  # if a new decodable observation is available from the feature accumulator
                prev_state = exp.bmi_system.decoder.get_state()

                exp.bmi_system.decoder(decodable_obs, **kwargs)
                # Determine whether the current state or previous state should be given to the learner
                if exp.bmi_system.learner.input_state_index == 0:
                    learner_state = exp.bmi_system.decoder.get_state()
                elif exp.bmi_system.learner.input_state_index == -1:
                    learner_state = prev_state
                else:
                    print(("Not implemented yet: %d" %
                           exp.bmi_system.learner.input_state_index))
                    learner_state = prev_state

                if learn_flag:
                    exp.bmi_system.learner(decodable_obs.copy(), learner_state, target_state_k, exp.bmi_system.decoder.get_state(
                    ), task_state, state_order=exp.bmi_system.decoder.ssm.state_order)

            decoded_states[:, k] = exp.bmi_system.decoder.get_state()

            ############################
            # Update decoder parameters
            ############################
            if exp.bmi_system.learner.is_ready():
                batch_data = exp.bmi_system.learner.get_batch()
                batch_data['decoder'] = exp.bmi_system.decoder
                kwargs.update(batch_data)
                exp.bmi_system.updater(**kwargs)
                exp.bmi_system.learner.disable()
                
                #measure features. 
                if isinstance(exp, FeatureSelector):
                    exp.select_features(batch_data['spike_counts'],
                                       batch_data['intended_kin'])
                

            new_params = None  # by default, no new parameters are available
            if exp.bmi_system.has_updater:
                new_params = copy.deepcopy(exp.bmi_system.updater.get_result())

            # Update the decoder if new parameters are available
            if not (new_params is None):
                exp.bmi_system.decoder.update_params(
                    new_params, **exp.bmi_system.updater.update_kwargs)
                new_params['intended_kin'] = batch_data['intended_kin']
                new_params['spike_counts_batch'] = batch_data['spike_counts']

                exp.bmi_system.learner.enable()
                update_flag = True

                # Save new parameters to parameter history
                exp.bmi_system.param_hist.append(new_params)



        # saved as task data
        # return decoded_states, update_flag
        tmp = decoded_states
        exp.task_data['internal_decoder_state'] = tmp

        # reset the plant position
        # @riglib.bmi.BMILoop.move_plant  line:1254
        exp.plant.drive(exp.decoder)

        # check state transitions and run the FSM.
        current_state = exp.state

        # iterate over the possible events which could move the task out of the current state
        for event in exp.status[current_state]:
            # if the event has occurred
            if exp.test_state_transition_event(event):
                # execute commands to end the current state
                exp.end_state(current_state)

                # trigger the transition for the event
                exp.trigger_event(event)

                # stop searching for transition events (transition events must be
                # mutually exclusive for this FSM to function properly)
                break

        # sort out the loop params.
        # inc cycle count
        exp.cycle_count += 1

        # save target data as was done in manualControlTasks._cycle
        exp.task_data['target'] = exp.target_location.copy()
        exp.task_data['target_index'] = exp.target_index

        #done in bmi:_cycle after move_plant
        exp.task_data['loop_time'] = exp.iter_time()


        #fb_controller data
        exp.task_data['target_state'] = target_state

        #encoder data
        #input to this is actually extractor
        exp.task_data['ctrl_input'] = np.reshape(exp.extractor.sim_ctrl, (1,-1))

        #actually output
        exp.task_data['spike_counts'] = feature_data['spike_counts']


        #save the decoder_state
        #from BMILoop.move_plant
        exp.task_data['decoder_state'] = exp.decoder.get_state(shape=(-1,1))
        
        #save bmi_data
        exp.task_data['update_bmi'] = update_flag


        # as well as plant data.
        plant_data = exp.plant.get_data_to_save()
        for key in plant_data:
            exp.task_data[key] = plant_data[key]

        # clda data handled in the above call.

        # save to the list hisory of data.
        exp.task_data_hist.append(exp.task_data.copy())
        
        #print out the trial update whenever wait count changes, alright. 
        if finished_trials != exp.calc_state_occurrences('wait'):
            finished_trials = exp.calc_state_occurrences('wait')
            print(f'finished trials :{finished_trials} with a current assist level of {exp.get_current_assist_level()}')


    if exp.verbose:
        print("end of FSM.run, task state is", exp.state)
    
    


# ## actually running the experiments

# In[ ]:


for i,e in enumerate(exps):
    run_exp_loop(e, **kwargs_exps[i])
    print(f'Finished running  {exp_conds[i]}')


# # Postprocessing the data for loading

# In[ ]:


for e in  exps: print(e.calc_state_occurrences('reward'))


# ## declare defs and conventions

# In[ ]:


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

# In[ ]:


task_data_hist_np_all = [np.array(e.task_data_hist) for e in exps]
len(task_data_hist_np_all)
task_data_hist_np_all[0].dtype


# ## Finished time in seconds

# In[ ]:


finished_times_in_seconds = [int(len(s)/FRAME_RATE) for s in task_data_hist_np_all]
finished_times_in_seconds


# # Post data analysis

# ## Overall  trial statistics succuss rate)

# In[ ]:


def filter_state(state_log:list, state_to_match:str)->list:
    '''#set axis limits of plot (x=0 to 20, y=0 to 20)
plt.axis([0, 20, 0, 20])
plt.axis("equal")

    state_log: a list of tuples (state:string, start_time: float)
    state_to_watch
    
    returns a list of element type
    '''
    
    return list(filter(lambda k: k[0] == state_to_match, state_log) )

def calc_inter_wait_times(wait_log: list)-> list:
    """
    state_log: a list of tuples ("wait", start_time: float)
    return a list of tuples: ("wait", start_time: float, diff_time)
    """
    wait_log_with_diff = list()
    for i, wait_state in enumerate(wait_log):
        if i == len(wait_log)-1: #there is nothing to subtract, just put zero.
            wait_log_with_diff.append((wait_state[1],  0))
            
        else:
            finish_time = wait_log[i+1][1]
            wait_log_with_diff.append((wait_state[1],  finish_time - wait_state[1]))
    
    return np.array(wait_log_with_diff[:-1])


# In[ ]:


state_logs = [e.state_log for e in exps]


# In[ ]:


STATE_CUT_NAME =  'wait'
#get the state logs
wait_logs = [filter_state(s, STATE_CUT_NAME) for s in state_logs]

inter_wait_times = [calc_inter_wait_times(w) for w in wait_logs]
#this has both start times 


# In[ ]:




for i in inter_wait_times:
    plt.scatter(i[:,0], i[:,1])

plt.legend(exp_conds)
plt.xlabel('Training progression(s)')
plt.ylabel('Trial time (s)')


# In[ ]:


wait_time = inter_wait_times[0]
task_data_hist_np = task_data_hist_np_all[0]


# In[ ]:


def sort_trials(wait_time:list, 
                target_seq:list,
                task_data_hist_np:dict, 
                dict_keys, FRAME_RATE = 60):
    trial_dict = list()
    
    for i,row in enumerate(wait_time):
        start_time = row[0]
        inter_wait_time = row[1]

        start_sample = int(start_time * FRAME_RATE)
        inter_wait_sample = int(inter_wait_time * FRAME_RATE)
        stop_sample = start_sample + inter_wait_sample

        single_trial_dict = dict()

        for k in dict_keys:
            
            requested_type_data = np.squeeze(task_data_hist_np[k])
            single_trial_dict[k] =  requested_type_data[start_sample:stop_sample,
                                                       :]
        #add more info
        single_trial_dict['start_time'] = row[0]
        single_trial_dict['inter_wait_time'] = row[1]
        
        #add target info
        single_trial_dict['targets'] = target_seq[i]

        #add the dictionary to the list
        trial_dict.append(single_trial_dict)
        
    return trial_dict


# In[ ]:




trial_dicts_all = []
dict_keys = ['cursor', #behaviour
             'ctrl_input', 'spike_counts', #encoder translates intended ctrl into spike counts
             'decoder_state']

for i in range(NUM_EXP):
    wait_time = inter_wait_times[i]
    task_data_hist_np = task_data_hist_np_all[i]
    
    trial_dict_0 = sort_trials(wait_time, 
                               target_seq,
                               task_data_hist_np, dict_keys)
    
    trial_dicts_all.append(trial_dict_0)

len(trial_dicts_all)


# ## Trajectory analysis

# In[ ]:


n_roi_trials = N_TRIALS - 1
unique_targets =  np.unique(target_seq, axis = 0)


X_CURSOR = 0
Z_CURSOR = 2
CIRCL_ALPHA = 0.2



RANGE_LIM =  15
figure, axes = plt.subplots() 

axes.set_xlim(-RANGE_LIM, RANGE_LIM)
axes.set_ylim(-RANGE_LIM, RANGE_LIM)

CIRCLE_RADIUS = exps[0].target_radius

#plot the targets

#plot the origin

cc = plt.Circle((0,0 ), 
            radius = CIRCLE_RADIUS,
            alpha = CIRCL_ALPHA)

axes.add_artist( cc ) 

for origin_t in unique_targets:
    origin = origin_t[0]
    t = origin_t[1]

    cc = plt.Circle((t[X_CURSOR],t[Z_CURSOR] ), 
                    radius = CIRCLE_RADIUS,
                    alpha = CIRCL_ALPHA)
                     
    axes.set_aspect( 1 ) 
    axes.add_artist( cc ) 
    
    
for trial_dict in trial_dicts_all:
    
    sample_trial = trial_dict[n_roi_trials]
    trial_cursor_trajectory = sample_trial['cursor']
    
    
    axes.plot(trial_cursor_trajectory[:, X_CURSOR], 
             trial_cursor_trajectory[:, Z_CURSOR])


# In[ ]:


print('finished trials:')

for i,e in  enumerate(exps): 
    reward_num = e.calc_state_occurrences('reward')
    print(f'{exp_conds[i]}: {reward_num} out of {N_TRIALS}')


# ## encoder
# 
# the job of the encoder is to directly encode intention into firing rates
# the direct measure is just pearson correlation coefficients between 
# the intentions and the firing rates

# In[ ]:


print('the encoder observation Q matrix')
for i,e in enumerate(exps):
    print(exp_conds[i])
    print(e.encoder.ssm.w)


# In[ ]:


n_exp = 0


spike_count_sample = trial_dicts_all[n_exp][n_roi_trials]['spike_counts']


# ## decoder

# In[ ]:


TEXT_OFFSET_VERTICAL = -0.2

decoder_c_after,axs = plt.subplots(1, NUM_EXP,
                                  figsize = (12,4))

decoder_c_after.suptitle('C matrix after')

print('steady state tuning curves:')
for i,e in enumerate(exps):
    e.decoder.plot_C(ax = axs[i])
    axs[i].set_title(exp_conds[i])
    
    #get the lower left coordinate
    y_lim_range  = axs[i].get_ylim()[1] - axs[i].get_ylim()[0]
    
    axs[i].text(0, TEXT_OFFSET_VERTICAL,
                f'finished {N_TRIALS} trials in {finished_times_in_seconds[i]} s', 
               transform = axs[i].transAxes)

decoder_c_after.text(0, 1.4 * TEXT_OFFSET_VERTICAL, 
                     f'CLDA rho {RHO}',
                    transform = axs[0].transAxes)


# In[ ]:


decoder_c_figure


# ## compare before and after the training

# In[ ]:




N_ROWS = 2 #before and after 
FIGURE_SIZE_2_by_4 = (12,4)
figure_compare_decoder_C_matrix,axs = plt.subplots(2,NUM_EXP, 
                                                   figsize = (12,4))

for i in range(NUM_EXP):
    axs[0, i] = decoder_c_figure.get_axes()[i]
    axs[1, i] = decoder_c_after.get_axes()[i]
    


# # CLDA updates

# ## clda update frequencies

# In[ ]:



clda_params_all = [np.array(e.bmi_system.param_hist) for e in exps]

for c in clda_params_all:
    print(f'did clda for {len(c)} times')


# In[ ]:


update_bmi_all = np.squeeze(task_data_hist_np_all[0]['update_bmi'])


# In[ ]:


plt.plot(update_bmi_all[:240])
plt.xlabel('frame count')


# ## reformat the matrix

# In[ ]:


clda_params = clda_params_all[1]

clda_params_dict_all = list()

for p in clda_params_all:
    clda_params_dict = dict()
    for param_key in p[0].keys():
        clda_params_dict[param_key] = np.array([ record_i[param_key] for record_i in p])
    
    clda_params_dict_all.append(clda_params_dict)


len(clda_params_dict_all)


# ## observation covariance matrix

# In[ ]:


n_sample = 1

for i,c in enumerate(clda_params_dict_all):
    print(exp_conds[i])
    print(c['kf.Q'][n_sample,:,:])
    print()


# ## clda K matrix

# In[ ]:


kf_C = clda_params_dict['kf.C']


# In[ ]:


print('K matrix before:')
print(kf_C[0,:,:])
print('K matrix after:')
print(kf_C[-1,:,:])


# In[ ]:


N_CLDA_ROI_TIME = 800

FIGURE_SIZE = (2,10)

N_NEURONS

f, axs = plt.subplots(1,N_NEURONS,figsize=(16,4))

for i in range(N_NEURONS):
    axs[i].plot(np.squeeze(kf_C[:N_CLDA_ROI_TIME,i, X_VEL_STATE_IND]))
    axs[i].plot(np.squeeze(kf_C[:N_CLDA_ROI_TIME,i, Y_VEL_STATE_IND]))
    axs[i].legend(['x vel', 'y_vel'])
    axs[i].set_title(f'neuron {i} ')
    axs[i].set_xlabel('clda update count ')
    axs[i].set_ylabel('Decoder weight value')


# In[ ]:


exps[0].encoder.C


# ## examine training batches

# In[ ]:


spike_counts_batch = clda_params_dict['spike_counts_batch'] 
intended_kin = clda_params_dict['intended_kin']


# In[ ]:


training_sample_point = 2

print('intended kinematics:')
print(intended_kin[training_sample_point])

print('spike counts:')
print(spike_counts_batch[training_sample_point])

print('trained KF C matrix:')
print(kf_C[training_sample_point])


# # Measure weights change

# In[ ]:


from weights_linear_regression import *
from afs_plotting import plot_prefered_directions


# In[ ]:


kf_C = clda_params_dict['kf.C']
kf_C.shape


# In[ ]:


history_of_L2_norms = calc_a_history_of_matrix_L2norms_along_first_axis(kf_C,
                                                                       (X_VEL_STATE_IND, Y_VEL_STATE_IND))

history_of_L2_norms.shape


# In[ ]:


figure_compare_kf, axes_compare_kf = plt.subplots(1,3, figsize = (12, 4))


# In[ ]:


time_of_interest = 0
kf_slice_start = np.squeeze(kf_C[time_of_interest, :,:])
print(f'kf_slice has kf_slice has shape {kf_slice_start.shape}')

kf_slice_end = np.squeeze(kf_C[-1, :,:])


#plot the trajectjory
axes_compare_kf[0].plot(history_of_L2_norms)

#plot the begining 
plot_prefered_directions(kf_slice_start, 
                         ax = axes_compare_kf[1])
axes_compare_kf[0].set_title('Prefered direction vector L2 norm')
axes_compare_kf[0].set_xlabel('update batch number')



#plot the end
plot_prefered_directions(kf_slice_end, 
                         ax = axes_compare_kf[2])
axes_compare_kf[2].set_title('Preffered directions at conclusion')


figure_compare_kf


# # Measure ssms

# In[ ]:


spike_counts_batch = clda_params_dict['spike_counts_batch'] 
intended_kin = clda_params_dict['intended_kin']

print(f'spike_counts_batch has the shape of {spike_counts_batch.shape}')
print(f'intended_kin has the shape of {intended_kin.shape}')

print(f'kf_C has the shape of {kf_C.shape}')


# In[ ]:


VEL_DECODING_STATES = (X_VEL_STATE_IND, Y_VEL_STATE_IND,)
print(f'we will only be looking the indices of the x vel and y vel {VEL_DECODING_STATES}')

time_slice_of_interest = 0
print(f'time slice of interest {time_slice_of_interest}')

spike_count_slice = np.squeeze(spike_counts_batch[time_slice_of_interest, :, :])
intended_kin_slice = np.squeeze(intended_kin[time_slice_of_interest, VEL_DECODING_STATES, :])
kf_C_slice = np.squeeze(kf_C[time_slice_of_interest, :,VEL_DECODING_STATES])

#rotate the dimensions=
spike_count_slice = spike_count_slice.T
intended_kin_slice = intended_kin_slice.T
kf_C_slice = kf_C_slice.T


print(f'spike_count_slice has the shape of {spike_count_slice.shape}')
print(f'intended_kin_slice has the shape of {intended_kin_slice.shape}')

print(f'kf_C_slice has the shape of {kf_C_slice.shape}')


# In[ ]:


from weights_linear_regression import calc_average_ssm_for_each_X_column

ssm_slice = calc_average_ssm_for_each_X_column(intended_kin_slice, spike_count_slice, kf_C_slice)

ssm_slice.shape


# ## encapsulate into the batch

# In[ ]:


(N_BATCHES, N_NEURONS, N_POINTS_IN_A_BATCH)  =  spike_counts_batch.shape

#initialize an nan array
ssm_all_batches = np.empty((N_BATCHES, N_NEURONS))
ssm_all_batches[:] = np.nan

for n_batch in range(N_BATCHES):
    spike_count_slice = np.squeeze(spike_counts_batch[n_batch, :, :])
    intended_kin_slice = np.squeeze(intended_kin[n_batch, VEL_DECODING_STATES, :])
    kf_C_slice = np.squeeze(kf_C[n_batch,:,VEL_DECODING_STATES])
    
    #rotate the dimensions=
    spike_count_slice = spike_count_slice.T
    intended_kin_slice = intended_kin_slice.T
    kf_C_slice = kf_C_slice.T

    
    ssm_slice = calc_average_ssm_for_each_X_column(intended_kin_slice, 
                                                   spike_count_slice, 
                                                   kf_C_slice)
    ssm_all_batches[n_batch, :] = ssm_slice
    
    


# In[ ]:


neurons_of_interest = range(8)

print(f'look at these two neurons {neurons_of_interest}')

plt.plot(ssm_all_batches[:, neurons_of_interest])


# ## over all ssm

# In[ ]:


(N_BATCHES, N_NEURONS, N_POINTS_IN_A_BATCH)  =  spike_counts_batch.shape

#initialize an nan array
ssm_all_batches = np.empty((N_BATCHES,))


for n_batch in range(N_BATCHES):
    spike_count_slice = np.squeeze(spike_counts_batch[n_batch, :, :])
    intended_kin_slice = np.squeeze(intended_kin[n_batch, VEL_DECODING_STATES, :])
    kf_C_slice = np.squeeze(kf_C[n_batch, :,VEL_DECODING_STATES])
    
    #rotate the dimensions=
    spike_count_slice = spike_count_slice.T
    intended_kin_slice = intended_kin_slice.T
    kf_C_slice = kf_C_slice.T
    
    ssm_slice = calc_ssm_y_minus_x_beta(intended_kin_slice, 
                                                   spike_count_slice, 
                                                   kf_C_slice)
    ssm_all_batches[n_batch] = ssm_slice
    
ssm_all_batches = ssm_all_batches / N_POINTS_IN_A_BATCH
    


# In[ ]:


plt.plot(ssm_all_batches)


# the batch time is small
# a batch should include at least 4 reaches.
# very linear. all of a sudden, it gets way.
# 
# inside target, setting the int to zero. 
# the hold vel is zero.  

# # Use variance to drop constant value features

# ## get the training sample

# In[ ]:


spike_counts_batch = clda_params_dict['spike_counts_batch'] 
intended_kin = clda_params_dict['intended_kin']

print(f'spike_counts_batch has the shape of {spike_counts_batch.shape}')
print(f'intended_kin has the shape of {intended_kin.shape}')

batch_of_interest = 23
print(f'only fit to 1 batch of interest: {batch_of_interest}')

spike_counts_batch_sample_batch = spike_counts_batch[batch_of_interest,:,:]
intended_kin_sample = intended_kin[batch_of_interest,:,:]
print(f'spike counts batch sample batch is {spike_counts_batch_sample_batch.shape}')
print(f'intended kin sample  is {intended_kin_sample.shape}')



# In[ ]:


from sklearn.feature_selection import VarianceThreshold

print('need to transpose to time by features')
feature_vector = spike_counts_batch_sample_batch.T
target_vector = intended_kin_sample.T
print(f'the feature vector has the shape of {feature_vector.shape}')
print(f'the target vector has the shape of {target_vector.shape}\n')


#set up the selector
variance_selector = VarianceThreshold(threshold = 0)

#calculate the empirical variances
feature_variances = variance_selector.fit(feature_vector)

variance_selector_params = variance_selector.get_params()

#fit the transform
transformed_feature_vec = variance_selector.fit_transform(feature_vector)

#this needs to be fitted before returning the selected features, eh. 
selected_feature_indx = variance_selector.get_support()


print(f'we know the feature variances to be {feature_variances}')
print(f'we know the parameters are {variance_selector_params}')
print(f'after transform, we know the dim of the feature vector {transformed_feature_vec.shape}\n')
print(f'we therefore, know the selected indices are {selected_feature_indx}')


# In[ ]:


VERTICAL_FIGURE_SIZE = 4

figure_feature_example, axes_feature_example = plt.subplots(1,2, 
                                                           figsize = (VERTICAL_FIGURE_SIZE * 2, VERTICAL_FIGURE_SIZE))

axes_feature_example[0].plot(feature_vector)
axes_feature_example[0].set_xlabel('sample count')
axes_feature_example[0].set_title('Before variance selection')

axes_feature_example[1].plot(transformed_feature_vec)
axes_feature_example[1].set_xlabel('sample count')
axes_feature_example[1].set_title('After variance selection')


# # Assess lasso fit

# In[ ]:


spike_counts_batch = clda_params_dict['spike_counts_batch'] 
intended_kin = clda_params_dict['intended_kin']

print(f'spike_counts_batch has the shape of {spike_counts_batch.shape}')
print(f'intended_kin has the shape of {intended_kin.shape}')

print(f'kf_C has the shape of {kf_C.shape}')


# In[ ]:


VEL_DECODING_STATES = (X_VEL_STATE_IND, Y_VEL_STATE_IND,)
print(f'we will only be looking the indices of the x vel and y vel {VEL_DECODING_STATES}')

time_slice_of_interest = 0
print(f'time slice of interest {time_slice_of_interest}')

spike_count_slice = np.squeeze(spike_counts_batch[time_slice_of_interest, :, :])
intended_kin_slice = np.squeeze(intended_kin[time_slice_of_interest, VEL_DECODING_STATES, :])
kf_C_slice = np.squeeze(kf_C[time_slice_of_interest, :,VEL_DECODING_STATES])

#rotate the dimensions=
spike_count_slice = spike_count_slice.T
intended_kin_slice = intended_kin_slice.T
kf_C_slice = kf_C_slice.T


print(f'spike_count_slice has the shape of {spike_count_slice.shape}')
print(f'intended_kin_slice has the shape of {intended_kin_slice.shape}')

print(f'kf_C_slice has the shape of {kf_C_slice.shape}')


# In[ ]:


kf_C_slice


# this matrix differs from below
# because this is mixed from the intial model

# In[ ]:


from sklearn import linear_model


linear_reg_model = linear_model.LinearRegression()
linear_reg_model.fit(intended_kin_slice, 
        spike_count_slice)

linear_reg_model.coef_


# ## Fit to Lasso path

# In[ ]:




clf = linear_model.Lasso(alpha = 1,
    max_iter = 100000)
clf.fit(spike_count_slice, intended_kin_slice)

print(clf.coef_)


# apparently, we have this

# In[ ]:


(N_BATCHES, N_NEURONS, N_POINTS_IN_A_BATCH)  =  spike_counts_batch.shape
lasso_alpha = 1

#initialize an nan array
lasso_coefs = np.empty((N_BATCHES, len(VEL_DECODING_STATES), N_NEURONS))
lasso_coefs[:] = np.nan

for n_slice in range(N_BATCHES):
    #get the exact dimensions
    spike_count_slice = np.squeeze(spike_counts_batch[n_slice, :, :])
    intended_kin_slice = np.squeeze(intended_kin[n_slice, VEL_DECODING_STATES, :])
    
    #rotate the dimensions=
    spike_count_slice = spike_count_slice.T
    intended_kin_slice = intended_kin_slice.T

    
    clf = linear_model.Lasso(alpha = lasso_alpha,
        max_iter = 100000)
    clf.fit(spike_count_slice, intended_kin_slice)


    lasso_coefs[n_slice,:,:] = clf.coef_
    


# In[ ]:


print('to use the weight function, we need to swap axis')
OLD_NEURON_AXIS = 2
OLD_STATE_AXIS = 1

lasso_coefs = np.swapaxes(lasso_coefs, OLD_NEURON_AXIS, OLD_STATE_AXIS)
print(f'the new lasso coef has axis of {lasso_coefs.shape}')

lasso_coefs_over_batches = calc_a_history_of_matrix_L2norms_along_first_axis(lasso_coefs,
                                                                       (0,1))


lasso_coefs_over_batches.shape


# In[ ]:


figure_lasso, lasso_axes = plt.subplots(1,2)

encoding_neurons_of_interest = range(4)
nonencoding_neurons_of_interest = range(4, 20)

lasso_axes[0].plot(lasso_coefs_over_batches[:, encoding_neurons_of_interest])

lasso_axes[1].plot(lasso_coefs_over_batches[:,nonencoding_neurons_of_interest])


# In[ ]:


figure_lasso_directions, lasso_pds_axes = plt.subplots(1,2)


time_of_interest = 0
lasso_slice_start = np.squeeze(lasso_coefs[time_of_interest, :,:])
print(f'kf_slice has kf_slice has shape {lasso_slice_start.shape}')

lasso_slice_end = np.squeeze(lasso_coefs[-1, :,:])


#plot the begining 
plot_prefered_directions(lasso_slice_start, 
                         ax = lasso_pds_axes[0])
lasso_pds_axes[0].set_title('Prefered direction vector L2 norm at the begining')
lasso_pds_axes[0].set_xlabel('update batch number')



#plot the end
#plot the begining 
plot_prefered_directions(lasso_slice_end, 
                         ax = lasso_pds_axes[1])
lasso_pds_axes[1].set_title('Prefered direction vector L2 norm at the last update')
lasso_pds_axes[1].set_xlabel('update batch number')




# ## Cross-validate the fit

# In[ ]:


print('get a slice of data')
n_slice = 0

#get the exact dimensions
spike_count_slice = np.squeeze(spike_counts_batch[n_slice, :, :])
intended_kin_slice = np.squeeze(intended_kin[n_slice, VEL_DECODING_STATES, :])

#rotate the dimensions=
spike_count_slice = spike_count_slice.T
intended_kin_slice = intended_kin_slice.T

print(f'just to double check the shapes       spike_count_slice has a shape of {spike_count_slice.shape}      and intended_kin_slice has the shape of {intended_kin_slice.shape}')


# In[ ]:


eps = 1e-4 # this is the alpha range that we are interested in, right. 
CV = 5 
n_alphas = 100
print(f'to work with lasso regression, we are using an alpha range of {eps} as defined by alpha_min / alpha_max')
print(f'and we are doing {CV} fold validation')
print(f'and by default, we are using {n_alphas} alphas ')

lassoCV_result = linear_model.MultiTaskLassoCV(eps = eps,
                                     cv = CV).fit(spike_count_slice, intended_kin_slice)

print(f'lassoCV_result returns the best alpha of {lassoCV_result.alpha_}')
print(f'and the alphas we used are {lassoCV_result.alphas}')


# ## Visualize the lasso path

# In[ ]:


from sklearn.linear_model import lasso_path

#determine the error tolerance
eps = 3e-5

alpha_lasso, coefs_lasso,_  = lasso_path(intended_kin_slice,
                                     spike_count_slice)


print(f'alphas have the shape of {alpha_lasso.shape}')
print(f'coefs have the shape of {coefs_lasso.shape}')

coef_dim_1, coef_2, coef_3 = coefs_lasso.shape


reshaped_coefs_lasso = np.reshape(
    coefs_lasso, (coef_dim_1 * coef_2, coef_3), order = 'F' #Fortan like column first like reshaping
)
print('transpose for plotting')
reshaped_coefs_lasso = reshaped_coefs_lasso.T
print(f'reshaped coefs array has dimension of {reshaped_coefs_lasso.shape}')


# In[ ]:


print('let plot the coefficient trajectory')

neg_log_alpha_lasso = - np.log(alpha_lasso)

plt.plot(neg_log_alpha_lasso, reshaped_coefs_lasso)
plt.xlabel('-log(alpha)')
plt.ylabel('Coefficients')


# In[ ]:


print('reorient the array')


# In[ ]:


a = np.array([
    [[1,2],[0,1]],
    [[1,2],[1,2]],
    [[3,4],[5,6]]
])
print(a.shape)
print(a)


# In[ ]:


np.ravel(a)
        


# In[ ]:


np.reshape(a, (6,2), order = 'F')


# ## Fit lasso to accumulating data

# In[ ]:


(N_BATCHES, N_NEURONS, N_POINTS_IN_A_BATCH)  =  spike_counts_batch.shape
(N_BATCHES, N_STATES, N_POINTS_IN_A_BATCH) = intended_kin.shape

print(f'{intended_kin.shape}')


# In[ ]:


print('we are gonna swap axes for the state as well as ')
OLD_NEURON_AXIS = OLD_STATE_AXIS = 1
OLD_BATCH_TIME_AXIS =  2

spike_counts_batch_reshape = np.swapaxes(spike_counts_batch, OLD_BATCH_TIME_AXIS,OLD_NEURON_AXIS)
intended_kin_reshape = np.swapaxes(intended_kin, OLD_BATCH_TIME_AXIS,OLD_STATE_AXIS)


print(spike_counts_batch_reshape.shape)
print(f'the new intended kin has batch time axis as {intended_kin_reshape.shape}')


# In[ ]:


N_DECODING_STATES = len(VEL_DECODING_STATES)


lasso_alpha_all = [1, 10, 100]


lasso_coefs_all = list()
ols_coefs_all = list()

for lasso_alpha in lasso_alpha_all:
    #initialize an nan array
    lasso_coefs_accumulate = np.empty((N_BATCHES,N_DECODING_STATES, N_NEURONS))
    lasso_coefs_accumulate[:] = np.nan

    #initialize 
    ols_coefs_accumulate = np.empty((N_BATCHES,N_DECODING_STATES, N_NEURONS))
    ols_coefs_accumulate[:] = np.nan
    

    for n_batch in range(1,N_BATCHES):


        spike_count_reshape_slice = np.reshape(spike_counts_batch_reshape[:n_batch, :, :], 
                                               (-1, N_NEURONS)) #keep the neurons and layout the batches

        intended_kin_reshape_slice = np.reshape(intended_kin_reshape[:n_batch, :, VEL_DECODING_STATES],
                                               (-1, N_DECODING_STATES))


        clf = linear_model.Lasso(alpha = lasso_alpha, max_iter = 100000)
        clf.fit(spike_count_reshape_slice, intended_kin_reshape_slice)

        ols = linear_model.LinearRegression()
        ols.fit(spike_count_reshape_slice, intended_kin_reshape_slice)

        lasso_coefs_accumulate[n_batch,:,:] = clf.coef_

        ols_coefs_accumulate[n_batch,:,:] = ols.coef_
    
    lasso_coefs_all.append(lasso_coefs_accumulate)
    ols_coefs_all.append(ols_coefs_accumulate)
    


# In[ ]:


lasso_coefs_all =  np.array(lasso_coefs_all)
ols_coefs_all = np.array(ols_coefs_all)
print(f'the alpha scanning has a shape of {lasso_coefs_all.shape}')
print(f'the alphas has a shape of {ols_coefs_all.shape}')


# In[ ]:


print(lasso_coefs_accumulate.shape)
print('lets look at the coefficients')

figure, axes = plt.subplots(1, len(lasso_alpha_all), figsize = (12, 4))

for i,a in enumerate(lasso_alpha_all):
    
    lasso_coefs_accumulate = np.squeeze(lasso_coefs_all[i, :,:,:])
    ols_coefs_accumulate = np.squeeze(ols_coefs_all[i, :,:,:])

    axes[i].plot(np.reshape(lasso_coefs_accumulate, (N_BATCHES, -1)))

    #plot the ols
    axes[i].plot(np.reshape(ols_coefs_accumulate, (N_BATCHES, -1)),
            linestyle = 'dotted')
    
    axes[i].set_title(f'lasso an alpha {lasso_alpha_all[i]} vs. ols ')

    axes[i].set_xlabel('Cumulative number batches')
    axes[i].set_ylabel('Coefficients')


print()


# # Create a feature selection class
# 
# this is heavily modeled after sklearn.feature selection stuff

# ## feature transformer
# 

# In[ ]:


class FeatureTransformer():
    SUPPORTED_METHODS = [
        'nothing',
        'transpose',
        'select_rows',
        'select_columns'
    ]
    
    def __init__(self, proc_meth_arg, *args, **kwargs):
        '''
        preprocess_methods[a list of tuples]:  
        each tuple of has the format of ('method in string', arguments)
        '''
        #check if the methods are supported
        for meth, arg in proc_meth_arg:
            if not self.check_meth_supported(meth): 
                raise Exception(f'the proc method {meth} is not supported')
        
        #if supported,
        self.proc_meth_arg = proc_meth_arg
        
    def __call__(self, feature_matrix):
        return self.preprocess_features(feature_matrix)
    
    def preprocess_features(self, feature_matrix):
        '''
        iteratively applies the methods to the features
        '''
        feature_matrix_temp = np.copy(feature_matrix)
        
        for meth, arg in self.proc_meth_arg:
            
            if meth == 'nothing':
                pass
            elif meth == 'transpose': 
                feature_matrix_temp = feature_matrix_temp.T
            elif meth =='select_rows':
                feature_matrix_temp = feature_matrix_temp[arg, :]
            elif meth == 'select_columns':
                feature_matrix_temp = feature_matrix_temp[:, arg]
            else:
                raise Exception(f'unsupported method {meth}')
                
        return feature_matrix_temp

    def check_meth_supported(self, proc_meth:str):
        return proc_meth in self.SUPPORTED_METHODS
    
class TransformerBatchToFit(FeatureTransformer):
    X_VEL_STATE_IND = 3
    Y_VEL_STATE_IND = 5
    
    def __init__(self, *args, **kwargs):
        proc_meth_arg = [
            ('select_rows', (self.X_VEL_STATE_IND, self.Y_VEL_STATE_IND)),
            ('transpose', ())
        ]
        
        super().__init__(proc_meth_arg, *args, **kwargs)
        
        


# In[ ]:



A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
     
])

feature_meth_arg = [
    ('transpose', None ),
    ('select_rows',0)
]

feat_trans = FeatureTransformer(feature_meth_arg)

print(feat_trans(A))


# ## Feature selector class

# In[ ]:



class FeatureSelector():
    
    def __init__(self, *args, **kwargs):
        
        self._init_feature_transformer(*args, **kwargs)
        
        self.fit_flag =  False
        self.measure_ready = False
        
        #default threshold to be 0
        self.current_threshold = 0
        self.threshold_ready = False
        
        #set up tracking weights
        self.history_of_weights = list()
        
        #this is a list that tracks the features selected
        self.selected_feature_indices = np.nan
        self.selected_feature_log = []
        
    def __call__(self, feature_matrix, target_matrix):
        tranformed_x, transformed_y = self.transform_features(x_array = feature_matrix, 
                                                              y_array = target_matrix)
        self.measure_features(tranformed_x, transformed_y)
        return self.get_feature_weights()
        
    def transform_features(self, x_array = None, y_array = None):
        
        transformed_results = list()
        
        if self.transform_x_flag: 
            transformed_x_array = self.feature_x_transformer(x_array)
            transformed_results.append(transformed_x_array)
        else:
            transformed_results.append(np.copy(x_array))
            
        if self.transform_y_flag:
            transformed_y_array = self.feature_y_transformer(y_array)
            transformed_results.append(transformed_y_array)
        else:
            transformed_results.append(np.copy(y_array))
            
        return  tuple(transformed_results)
    
        
    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        pass
    
    def get_feature_weights(self):
        pass
        
    def threshold_features(self):
        pass
    
    def get_selected_feature_indices(self):
        return self.selected_feature_indices
    
    
    def _init_feature_transformer(self, *args, **kwargs):
        #initialize preproc features
        self.transform_x_flag = kwargs['transform_x_flag'] if 'transform_x_flag' in kwargs.keys() else False
        self.transform_y_flag = kwargs['transform_y_flag'] if 'transform_y_flag' in kwargs.keys() else False

        if self.transform_x_flag:
            if 'feature_x_transformer' in kwargs.keys():
                self.feature_x_transformer = kwargs['feature_x_transformer']
            else: 
                raise Exception(f'{__name__}: feature_x_transformer specified, but feature_transformer not in kwarg.keys')

        if self.transform_y_flag:
            if 'feature_y_transformer' in kwargs.keys():
                self.feature_y_transformer = kwargs['feature_y_transformer']
            else: 
                raise Exception(f'{__name__}: feature_y_transformer specified, but feature_transformer not in kwarg.keys')


    
    


# In[ ]:



X = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
     
])


feature_x_meth_arg = [
    ('transpose', None ),
    ('select_rows',0)
]

y = np.array([
    [1,2,3],
    [4,5,6],     
])



feature_y_meth_arg = [
    ('transpose', None )
]


kwargs_feature = dict()
kwargs_feature = {
    'transform_x_flag':True,
    'transform_y_flag':True,
    'feature_x_transformer':TransformerBatchToFit(),
    'feature_y_transformer':FeatureTransformer(feature_y_meth_arg)
}


fs_tester = FeatureSelector(**kwargs_feature)
#fs_tester.transform_features(X,y)


# In[ ]:


from sklearn.linear_model import Lasso

class LassoFeatureSelector(FeatureSelector):
    
    DEFAULT_ALPHA = 1
    DEFAULT_MAX_ITERATION = 10000
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        #param related to lasso, default to 1
        self.current_lasso = kwargs['lasso_alpha'] if 'lasso_alpha' in kwargs.keys() else self.DEFAULT_ALPHA
        self.max_iter = kwargs['max_iter'] if 'max_iter' in kwargs.keys() else self.DEFAULT_MAX_ITERATION

        self._init_lasso_regression(self.current_lasso, 
                                   self.max_iter)
        
        print(f'{__name__}: initialized lasso regression with an alpha of {self.current_lasso} and a max number of iteration of {self.max_iter}')

        
    def _init_lasso_regression(self, alpha, max_iter):
        '''
        set up the model
        '''
        self.lasso_model = Lasso(alpha = alpha,  
                                  max_iter= max_iter)
        
    
    def measure_features(self, feature_matrix, target_matrix):
        '''
        in this case, features are measured by their lasso weights
        
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        
        #fitted results to lasso_model._coef
        self.lasso_model.fit(feature_matrix, target_matrix)
        self.measure_ready = True
        
        #save to the history of measures
        self.history_of_weights.append(self.get_feature_weights())
        
    def get_feature_weights(self):
        if self.measure_ready: 
            return self.lasso_model.coef_
        else:
            return np.nan
        
    def get_history_of_weights(self):
        return np.array(self.history_of_weights)
        
    def threshold_features(self):
        pass
    
    def get_selected_feature_indices(self):
        return self.selected_feature_indices
        


# In[ ]:


print('test out the feature selector')

lasso_fs = LassoFeatureSelector()

print(spike_count_reshape_slice.shape)
print(intended_kin_reshape_slice.shape)

lasso_fs.measure_features(spike_count_reshape_slice,
                         intended_kin_reshape_slice)

print(f'this is asking questions about how well the spike count fit to the internal states')
lasso_fs.get_feature_weights()


# ## put it together

# In[ ]:


time_slice_of_interest = 0
print(f'time slice of interest {time_slice_of_interest}')

spike_count_slice = np.squeeze(spike_counts_batch[time_slice_of_interest, :, :])
intended_kin_slice = np.squeeze(intended_kin[time_slice_of_interest, :, :])
kf_C_slice = np.squeeze(kf_C[time_slice_of_interest, :,VEL_DECODING_STATES])

print(spike_count_slice.shape)
print(intended_kin_slice.shape)


# In[ ]:


feature_x_meth_arg = [
    ('transpose', None ),
]


kwargs_feature = dict()
kwargs_feature = {
    'transform_x_flag':True,
    'transform_y_flag':True,
    'feature_x_transformer':FeatureTransformer(feature_x_meth_arg),
    'feature_y_transformer':TransformerBatchToFit()
}

fs_tester = LassoFeatureSelector(**kwargs_feature)
tx_x, tx_y = fs_tester.transform_features(spike_count_slice, intended_kin_slice)

print(tx_x.shape)
print(tx_y.shape)


# In[ ]:


fs_tester.measure_features(tx_x, tx_y)
print(fs_tester.get_feature_weights())


# In[ ]:


fs_tester.get_history_of_weights().shape

