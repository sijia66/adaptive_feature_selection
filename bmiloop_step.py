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

# In[1]:


# import libraries
# make sure these directories are in the python path., 
from bmimultitasks import SimBMIControlMulti, SimBMICosEncKFDec, BMIControlMultiNoWindow
from features import SaveHDF
from features.simulation_features import get_enc_setup, SimKFDecoderRandom, SimCosineTunedEnc,SimIntentionLQRController, SimClockTick

from riglib import experiment

from riglib.stereo_opengl.window import FakeWindow

import time
import numpy as np


# ##  set up trial information

# In[2]:


#generate task params
N_TARGETS = 8
N_TRIALS = 1
seq = SimBMIControlMulti.sim_target_seq_generator_multi(
N_TARGETS, N_TRIALS)


# ##  simulation encoder decoder setup

# In[3]:


#clda on random 
DECODER_MODE = 'trainedKF' # in this case we load simulation_features.SimKFDecoderRandom
ENCODER_TYPE = 'cosine_tuned_encoder'
LEARNER_TYPE = 'dumb' # to dumb or not dumb it is a question 'feedback'
UPDATER_TYPE = 'none' #none or "smooth_batch"


SAVE_HDF = True
DEBUG_FEATURE = False

#neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'toy')


# ## from the setup options, set up experiment

# In[4]:


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
#to update the decoder.
if UPDATER_TYPE == 'smooth_batch':
    from features.simulation_features import SimSmoothBatch
    feats.append(SimSmoothBatch)
else: #defaut to none 
    print(f'{__name__}: need to specify an updater')
    
if DEBUG_FEATURE: 
    from features.simulation_features import DebugFeature
    feats.append(DebugFeature)
    
    
#pass the real time limit on clock
feats.append(SimClockTick)

#add back the saveHDF feature
if SAVE_HDF:  feats.append(SaveHDF)


#save everthing in a kw
kwargs = dict()
kwargs['sim_C'] = sim_C


# ## make our experiment class

# In[5]:


#spawn the task
Exp = experiment.make(base_class, feats=feats)
#print(Exp)
exp = Exp(seq, **kwargs)
exp.init()



# In[8]:

exp.run() 






# %%
