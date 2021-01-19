#!/usr/bin/env python
# coding: utf-8

# # Purposes of this document
# 
# the goal of this jupyter is to run the BMIloop in steps  
# we can  
# 1. understand what is happening. 
# 2. develop new components  
# 
# along these reasons,  
# we will clearly point out where the components are  
# we focus on how information is transmitted between components  
# 
# note all of this is in the branch sijia as of 2021 Jan
# 

# # setting up the simulation components
# 
# this section largely copyied and pasted from   
# bmi3d-sijia(branch)-bulti_in_experiemnts
# https://github.com/sijia66/brain-python-interface/blob/master/built_in_tasks/sim_task_KF.py

# In[ ]:


# import libraries
# make sure these directories are in the python path., 
from bmimultitasks import SimBMIControlMulti, SimBMICosEncKFDec, BMIControlMultiNoWindow
from features import SaveHDF
from features.simulation_features import get_enc_setup, SimKFDecoderRandom, SimCosineTunedEnc,SimIntentionLQRController, SimClockTick
from features.simulation_features import SimHDF, SimTime

from riglib import experiment

from riglib.stereo_opengl.window import FakeWindow

import time
import numpy as np


# ##  set up trial information

# In[ ]:


#generate task params
N_TARGETS = 8
N_TRIALS = 2
seq = SimBMIControlMulti.sim_target_seq_generator_multi(
N_TARGETS, N_TRIALS)


# ##  simulation encoder decoder setup

# In[ ]:


#clda on random 
DECODER_MODE = 'trainedKF' # in this case we load simulation_features.SimKFDecoderRandom
ENCODER_TYPE = 'cosine_tuned_encoder'
LEARNER_TYPE = 'feedback' # to dumb or not dumb it is a question 'feedback'
UPDATER_TYPE = 'smooth_batch' #none or "smooth_batch"


SAVE_HDF = False

SAVE_SIM_HDF = True #this makes the task data available as exp.task_data_hist

DEBUG_FEATURE = False

#neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'toy')


# ## from the setup options, set up experiment

# In[ ]:


# set up assist level
assist_level = (0, 0)

#base_class = SimBMIControlMulti
base_class = BMIControlMultiNoWindow
feats = []

#set up intention feedbackcontroller
#this ideally set before the encoder
feats.append(SimIntentionLQRController)

#set up the encoder
if ENCODER_TYPE == 'cosine_tuned_encoder' :
    feats.append(SimCosineTunedEnc)
    print(f'{__name__}: selected SimCosineTunedEnc\n')
    
    
   #take care the decoder setup
if DECODER_MODE == 'random':
    feats.append(SimKFDecoderRandom)
    print(f'{__name__}: set base class ')
    print(f'{__name__}: selected SimKFDecoderRandom \n')
else: #defaul to a cosEnc and a pre-traind KF DEC
    from features.simulation_features import SimKFDecoderSup
    feats.append(SimKFDecoderSup)
    print(f'{__name__}: set decoder to SimKFDecoderSup\n')
    
    
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
    
if DEBUG_FEATURE: 
    from features.simulation_features import DebugFeature
    feats.append(DebugFeature)
    
if SAVE_HDF: feats.append(SaveHDF)
if SAVE_SIM_HDF: feats.append(SimHDF)
    
    
#pass the real time limit on clock
feats.append(SimClockTick)
feats.append(SimTime)


#save everthing in a kw
kwargs = dict()
kwargs['sim_C'] = sim_C


# ## make our experiment class

# In[ ]:


#spawn the task
Exp = experiment.make(base_class, feats=feats)
#print(Exp)
exp = Exp(seq, **kwargs)
exp.init()


# ## test run a bit

# print(exp.state)
# exp.run() 

# # now comes to the step through BMIloop
# 

# ## decode neural features to move cursor
# this section basically steps through the move_plant stuff
# 
# @riglib.bmi.BMILoop.move_plant

# 
# ### bmi feature extraction, eh
# #riglib.bmi: 1202
# feature_data = exp.get_features()
# feature_data
# 
# ###  Determine the target_state and save to file
# current_assist_level = exp.get_current_assist_level()
# if np.any(current_assist_level > 0) or exp.learn_flag:
#     target_state = exp.get_target_BMI_state(self.decoder.states)
# else:
#     target_state = np.ones([exp.decoder.n_states, exp.decoder.n_subbins]) * np.nan
# 
# 
# #decode the new features
# #riglib.bmi.bmiloop: line 1245
# neural_features = feature_data[exp.extractor.feature_type]
# tmp = exp.call_decoder(neural_features, target_state, **kwargs)
# 
# #saved as task data
# exp.task_data['internal_decoder_state'] = tmp
# tmp
# 
# #### reset the plant position
# ####  @riglib.bmi.BMILoop.move_plant  line:1254
# exp.plant.drive(exp.decoder)
# exp.plant.position

# # assemble into a complete loop

# In[ ]:


# riglib.experiment: line 597 - 601
#exp.next_trial = next(exp.gen)
# -+exp._parse_next_trial()


# we need to set the initial state
# per fsm.run:  line 138


# Initialize the FSM before the loop
exp.set_state(exp.state)


while exp.state is not None:
    try:
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
        if np.any(current_assist_level > 0) or exp.learn_flag:
            target_state = exp.get_target_BMI_state(self.decoder.states)
        else:
            target_state = np.ones(
                [exp.decoder.n_states, exp.decoder.n_subbins]) * np.nan

        # decode the new features
        # riglib.bmi.bmiloop: line 1245
        neural_features = feature_data[exp.extractor.feature_type]
        
        #call decoder. 
        tmp = exp.call_decoder(neural_features, target_state, **kwargs)
        # saved as task data
        exp.task_data['internal_decoder_state'] = tmp

        # reset the plant position
        # @riglib.bmi.BMILoop.move_plant  line:1254
        exp.plant.drive(exp.decoder)

        
        ### check state transitions and run the FSM. 
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

    except:
        print('fsm_tick failed')
    
    ###sort out the loop params. 
    #inc cycle count
    exp.cycle_count += 1
    
    #save target data as was done in manualControlTasks._cycle
    exp.task_data['target'] = exp.target_location.copy()
    exp.task_data['target_index'] = exp.target_index
    #as well as plant data.
    plant_data = exp.plant.get_data_to_save()
    for key in plant_data:
        exp.task_data[key] = plant_data[key]
    
    
    #save to the list hisory of data. 
    exp.task_data_hist.append(exp.task_data.copy())
        

if exp.verbose:
    print("end of FSM.run, task state is", exp.state)


