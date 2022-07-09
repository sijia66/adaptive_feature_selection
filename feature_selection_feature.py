
import functools
import numpy as np
from numpy.lib.function_base import _select_dispatcher
from sklearn.linear_model import Lasso
import cvxpy as cp

from features import SaveHDF

from weights import change_target_kalman_filter_with_a_C_mat
#import 
from icecream import ic

import copy
import time #for sync stuff c

FRAME_RATE = 60

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


class FeatureSelector():
    
    def __init__(self, *args, **kwargs):

        #for multiple inheritance purposes
        super().__init__(*args, **kwargs)
        print('in feature selector mod')
        
        self._init_feature_transformer(*args, **kwargs)
        
        self.fit_flag =  False
        self.measure_ready = False

        self.feature_measure_count = 0
        
        #default threshold to be 0
        self.current_threshold = 0
        self.threshold_ready = False
        
        #set up tracking weights
        self.history_of_weights = list()
        
        #this is a list that tracks the features selected
        self.feature_change_flag = False
        self.decoder_change_flag = False
        self.decoder_changed = False
        self.selected_feature_indices = np.nan
        self.selected_feature_log = []

        #set up initia setting
        self.n_active_feats= kwargs['n_starting_feats']
        self.N_TOTAL_AVAIL_FEATS =  kwargs['n_starting_feats']
        self.n_states = kwargs['n_states']


        if 'init_feat_set' in kwargs.keys():
            self._active_feat_set = kwargs['init_feat_set']
            
            self.decoder_change_flag = True
            self.feature_change_flag = True

            print('use user supplied features')
        else:
            self._active_feat_set = np.full(self.n_active_feats, True, dtype = bool)

        self._prev_feat_set = np.full(self.n_active_feats, True, dtype = bool)


        self._active_feat_set_list = [np.copy(self._active_feat_set)]
        print(f'feature init: {self._active_feat_set_list}')

        #history of featrues
        self.used_C_mat = np.zeros((self.n_active_feats, self.n_states))
        self.used_Q_diag = np.zeros(self.n_active_feats)
        self.used_K_mat = np.zeros((self.n_states, self.n_active_feats)) #this is flipped of course.

        print(f'feature selector: add initial decoder weights')
        self._used_C_mat_list = list()
        self._used_K_mat_list = list()
        self._used_Q_diag_list = list()

        self.used_mfr = np.zeros(self.n_active_feats)
        self.used_sdFR = np.zeros(self.n_active_feats)
        
    def meausure_features_and_target(self, feature_matrix, target_matrix):
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

    def determine_change_features(self):
        pass

    def is_feature_change(self):
        return self.feature_change_flag

    def is_decoder_change(self):
        return self.decoder_change_flag 

    def select_features(self, neural_features):
        try:
            trans_neural_obs =  neural_features[self._active_feat_set]
        except:
            print(self._active_feat_set)
            raise Exception('Did not work')
        return trans_neural_obs

    def measure_neurons_wz_intendedkin_and_spike(self, intended_kin, spike_counts):
        '''
        calculate the y-CX covariance matrix
        and then select the top 25 percentile features
        '''

        C_hat, Q_hat = kfdecoder.KalmanFilter.MLE_obs_model(intended_kin, spike_counts, include_offset=False)


        return (C_hat, Q_hat)

    def select_decoder_features(self, target_decoder, debug = False):
        '''
        
        '''

        self._change_one_flag = False 
 

        #update the used C matrix  with the current values 
        if debug:
            print(self.used_C_mat[self._prev_feat_set, : ].shape )
            print(target_decoder.filt.C.shape)
            
            
        self.used_C_mat[self._prev_feat_set, : ] = np.copy(target_decoder.filt.C)
        transformed_C =  np.matrix(self.used_C_mat[self._active_feat_set, :])

        #only update diagnoal matrix
        self.used_Q_diag[self._prev_feat_set] = np.copy(np.diag(target_decoder.filt.Q))
        transformed_Q_diag = self.used_Q_diag[self._active_feat_set]
        transformed_Q = np.matrix(np.diag(transformed_Q_diag))
        
        #use the updated C,Q matrices to update the decoder
        if debug:
            print(f'active feature set is {self._active_feat_set}')
            print(f'after trans: {transformed_C.shape}')
            print(f'after trans: {transformed_Q.shape}')

        change_target_kalman_filter_with_a_C_mat(target_decoder.filt, transformed_C, 
                                        Q = transformed_Q, 
                                        debug = False
                                        )

        #change the decoder tracking of  mFR, sdFR
        self.used_mfr[self._prev_feat_set] = np.copy(target_decoder.mFR)
        target_decoder.mFR = self.used_mfr[self._active_feat_set]
       
        self.used_sdFR[self._prev_feat_set] = np.copy(target_decoder.sdFR)
        target_decoder.sdFR = self.used_sdFR[self._active_feat_set]
        
        #update the feature count
        target_decoder.n_features = target_decoder.filt.C.shape[0]
        #To-Do: also need to change how many feats we need

        self._prev_feat_set  = np.copy(self._active_feat_set)

        if debug: print(f'decoder change flag to false')
        self.decoder_change_flag = False
    
    def record_feature_active_set(self, target_decoder):
        self._active_feat_set_list.append(np.copy(self._active_feat_set))
        self.used_C_mat[self._active_feat_set,: ] = np.copy(target_decoder.filt.C)
        self._used_C_mat_list.append(np.copy(self.used_C_mat))

        self.used_Q_diag[self._active_feat_set] = np.diag(np.copy(target_decoder.filt.Q))
        self._used_Q_diag_list.append(np.copy(self.used_Q_diag))

        #K matrix, a new k matrix needs to be calculated
        if hasattr(target_decoder.filt.pred_state_P, 'shape'):

            K = target_decoder.filt._calc_kalman_gain(target_decoder.filt.pred_state_P)
        else:
            K = np.nan
        self.used_K_mat[:, self._active_feat_set] = np.copy(K)
        self._used_K_mat_list.append(np.copy(self.used_K_mat))

    def save_feature_params(self):
        import aopy

        #prepare data dict

        data_dict = {
            'feat_set': self._active_feat_set_list,
            'C_mat': self._used_C_mat_list,
            'Q_diag':self._used_Q_diag_list,
            'K_mat':self._used_K_mat_list
        }
        #save to data dict
        #mk temporary directory 
        import tempfile
        print(f'saving feature selection file name to {self.h5file.name}')
        aopy.data.save_hdf(tempfile.gettempdir(), self.h5file.name, data_dict, data_group="/feature_selection", append = True, debug=True)

        import pickle

        with open(self.h5file.name + '.p', "wb") as f:
            pickle.dump(self.bmi_system.param_hist, f)


    def add_new_features(self, target_decoder, num_add_feat):
        '''
        
        '''

        (num_old_feats,num_states) = target_decoder.filt.C.shape
        num_new_feats = num_add_feat + num_old_feats

        #add zero rows to the observation matrix
        prev_C = target_decoder.filt.C
        new_rows = np.zeros((num_add_feat, num_states))
        new_C =  np.vstack((prev_C, new_rows))

        new_C.shape

        #add rows and columns to the observation matrix
        prev_Q = target_decoder.filt.Q
        new_rows = np.zeros((num_add_feat, num_old_feats))
        new_Q = np.vstack((prev_Q, new_rows))

        new_cols = np.zeros((num_new_feats, num_add_feat))
        new_Q = np.hstack((new_Q, new_cols))

        # assign back to the decoder
        change_target_kalman_filter_with_a_C_mat(target_decoder.filt, new_C, Q = new_Q,
                                                debug = False)

    
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

from riglib.bmi import kfdecoder 
class SNRFeatureSelector(FeatureSelector):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.Q_diag_list = list()
        self._change_one_flag = True
    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        self.feature_measure_count += 1

        Q_diag = self.measure_neurons_wz_intendedkin_and_spike(target_matrix, feature_matrix)
        self.determine_change_features(Q_diag)



    def determine_change_features(self, Q_diag):
        #TODO:  this bit is pretty  
        

        if self.feature_measure_count == 1:
            
            self.selected_feature_indices = self.threshold_features(Q_diag, 25)
            print(f'determine_change_features: {self.selected_feature_indices}')


            #this is initial feat sel:
            self._active_feat_set[:] = False
            self._active_feat_set[self.selected_feature_indices] = True

            self.decoder_change_flag = True #turon the flag, the decoder can change its params
            self.feature_change_flag = True
            
            return 
        elif self.feature_measure_count == 6:
            self.selected_feature_indices = self.threshold_features(Q_diag, 50)
        else:
            #if nothing else just returns sort of thing. 
            return

        print(f'determine_change_features: {self.selected_feature_indices}')
        print(f'determine_change_feature: {self.feature_measure_count}')


        # set the active feature set to True
        temp_active_set = np.full(self.N_TOTAL_AVAIL_FEATS, False)
        temp_active_set[self.selected_feature_indices] = True

        print(f'active set before: {self._active_feat_set}')

        #update the feature indices
        self._active_feat_set = np.logical_or(self._active_feat_set, temp_active_set)
        self.decoder_change_flag = True #turon the flag, the decoder can change its params
        self.feature_change_flag = True

        print(f'determine feature change: after set up {self._active_feat_set}')
            



    def threshold_features(self, Q_diag, per):
        Q_per = np.percentile(Q_diag,  per)
        return np.argwhere(Q_diag <= Q_per)

class IterativeFeatureSelector(FeatureSelector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_high_SNR_time = kwargs.pop('train_high_SNR_time', 5)
        print("********************************************************")
        print(f'IterativeFeatureSelector: feature selection at {self.train_high_SNR_time}\n')
        print("********************************************************")
        #TODO: assume the first N number of neurons are high SNR 
        self.N_HIGH_SNR = np.sum(self._active_feat_set)

    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        self.feature_measure_count += 1

        self.determine_change_features()

    def determine_change_features(self):
       
       #assume we selected the first few features.

       train_high_SNR_time = self.train_high_SNR_time

       if self.feature_measure_count <= train_high_SNR_time:
           return

       feature_change_index_time = self.feature_measure_count - train_high_SNR_time
       selected_index = self.N_HIGH_SNR + feature_change_index_time

       if selected_index > self.N_TOTAL_AVAIL_FEATS: return True

       selected_index = np.max(selected_index, 0)

       self._active_feat_set[:selected_index] = True

       self.decoder_change_flag = True
       self.feature_change_flag = True


class ConvexFeatureSelector(FeatureSelector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_high_SNR_time = kwargs.pop('train_high_SNR_time', 5)
        self._objective_offset = kwargs.pop('objective_offset', 1)
        self._selection_threshold = kwargs.pop("threshold selection", 0.5)
        
        print("********************************************************")
        print(f'Convex feature selection: feature selection at {self.train_high_SNR_time}\n')
        print("********************************************************")
        
        #TODO: assume the first N number of neurons are high SNR 
        self.N_HIGH_SNR = np.sum(self._active_feat_set)

    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        self.feature_measure_count += 1

        C_mat, Q_diag = self.measure_neurons_wz_intendedkin_and_spike(target_matrix, feature_matrix)


        self.determine_change_features(C_mat, Q_diag)

    def determine_change_features(self, obs_c_mat, noise_q_mat):

       if self.feature_measure_count <= self.train_high_SNR_time:
           return

        # bad software practice, has to assume access to the kf c decoder

       obs_c_velocity_states_only = obs_c_mat[:, (X_VEL_STATE_IND, Y_VEL_STATE_IND)]
       diag_noise_q_mat = np.diag(np.diag(noise_q_mat))

       if self.feature_measure_count == self.train_high_SNR_time + 1:
            Q_diag_inv =  np.linalg.inv(diag_noise_q_mat)
            self._optimal_val = np.log(np.linalg.det((obs_c_velocity_states_only.T @ Q_diag_inv @ obs_c_velocity_states_only)))


       selected_values = self.convex_feature_selection_by_obj_fraction(obs_c_velocity_states_only, diag_noise_q_mat, 
                                                                      self._optimal_val,
                                                                      self._objective_offset)
       
       # threshold the values and calc the active features.
       selected_indices = np.argwhere(selected_values >= self._selection_threshold)

       # we are gonna take the intersection with exisiting features
       all_selected_features = np.full(self.N_TOTAL_AVAIL_FEATS, False, dtype = bool)
       all_selected_features[selected_indices] = True
       # take the intersection of the features
       self._active_feat_set = np.logical_or(self._active_feat_set, all_selected_features)
       
       # set the flags to make these changes effective. 
       self.decoder_change_flag = True
       self.feature_change_flag = True

    @classmethod
    def convex_feature_selection_by_obj_fraction(self, C, Q_diag, optimal_val, offset):
        """
        trying to solve the convex feature selection problem of the form
        minimize f(z) = log det (C.T Q^(-1) diag(z) C )
        subject to 0.001 <= theta <= 1

        Args:
            C (np.array): num_features by num_states
            Q (np.array): 

        """

        if offset == 0:
            return np.ones((C.shape[0]))

        d = C.shape[0]
        ones_d = np.ones((d,1))
        theta = cp.Variable((d,1))

        Q_diag_inv =  np.linalg.inv(Q_diag)

        constraints = [theta >=0.0,  theta <= 1, 
                    cp.log_det(C.T @ Q_diag_inv @cp.diag(theta) @ C) >= optimal_val - offset]

        feature_selection_objective = cp.Minimize( theta.T @ ones_d)
        feature_selection_problem = cp.Problem(feature_selection_objective, constraints)

        result = feature_selection_problem.solve()

        selected_values = theta.value.copy()

        return np.squeeze(selected_values)


def make_a_vector_of_ones(a_matrix, mode = "same_number_as_row"):

    # deal with a matrix
    assert len(a_matrix.shape) == 2
    num_row, num_col = a_matrix.shape

    if  mode == "same_number_as_row":
        return np.ones((num_row, 1))
    elif mode == "same_number_as_col":
        return np.ones(( num_col,1))
    else:
        raise Exception("only supports modes same_number_as_row or same_number_as_col")

import collections
class JointConvexFeatureSelector(FeatureSelector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_high_SNR_time = kwargs.pop('train_high_SNR_time', 5)
        self.N_TOTAL_AVAIL_FEATS = 128
        
        # the stuff for the achieving some fraction of the feature selection method
        self._objective_offset = kwargs.pop('objective_offset', 1)


        self._setup_sparse_smooth_params(**kwargs)
        self._initialize_deque()

        # after we do optimization, we need these params to get our selection going sort of thing.
        self._selection_threshold = kwargs.pop("threshold_selection", 0.5)
        
        print("********************************************************")
        print(f'Convex feature selection: feature selection at {self.train_high_SNR_time}\n')
        print("********************************************************")
        
        #TODO: assume the first N number of neurons are high SNR 
        self.N_HIGH_SNR = np.sum(self._active_feat_set)

    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        self.feature_measure_count += 1

        C_mat, Q_diag = self.measure_neurons_wz_intendedkin_and_spike(target_matrix, feature_matrix)


        self.determine_change_features(C_mat, Q_diag)
    
    def _initialize_deque(self):

        self._curr_prior_deque = collections.deque()
        # we enque the prior feature score
        for i in range(self._num_lags):
            self._curr_prior_deque.appendleft(np.ones((self.N_TOTAL_AVAIL_FEATS)))

        self._next_disc_memory = np.ones((self.N_TOTAL_AVAIL_FEATS, self._num_lags))

        print("initialized memory deque with length of ", len(self._curr_prior_deque))

    def _setup_sparse_smooth_params(self, **kwargs):

        # the stuff for dual objective feature selection
        self._sparsity_coef = kwargs.pop('sparsity_coef', 1)
        self._smoothness_coef = kwargs.pop("smoothness_coef", 0.5)
        self._num_lags = kwargs.pop("num_of_lags", 5)
        self._alpha = kwargs.pop("past_batch_decay_factor", 0.9)
        

    def determine_change_features(self, obs_c_mat, noise_q_mat):

       if self.feature_measure_count <= self.train_high_SNR_time:
           return

        # bad software practice, has to assume access to the kf c decoder

       obs_c_velocity_states_only = obs_c_mat[:, (X_VEL_STATE_IND, Y_VEL_STATE_IND)]
       diag_noise_q_mat = np.diag(np.diag(noise_q_mat))


        # we use different obj functions for the dual objectives
       print("joint_sparseness_sparseness: ", self._next_disc_memory.shape, "num_lags",  self._num_lags)

       selected_values, result = self.convex_feature_selection_with_joint_smooth_sparse_goals(obs_c_velocity_states_only, 
                                                                diag_noise_q_mat, 
                                                                self._sparsity_coef, 
                                                                self._smoothness_coef,
                                                                self._next_disc_memory)
                                                                
       print("doing joint smooth sparse optimization at batch ", self.feature_measure_count)


       
       # set up the rotation mechanism, sort of thing.
       self._curr_prior_deque.appendleft(selected_values.copy())
       if len(self._curr_prior_deque) > self._num_lags:
           self._curr_prior_deque.pop()

       # set it up so 
       self._next_disc_memory = np.array(self._curr_prior_deque).T
       self._next_disc_memory = self._alpha * self._next_disc_memory


       # threshold the values and calc the active features.
       selected_indices = np.argwhere(selected_values >= self._selection_threshold)

       # we are gonna take the intersection with exisiting features
       all_selected_features = np.full(self.N_TOTAL_AVAIL_FEATS, False, dtype = bool)
       all_selected_features[selected_indices] = True

       self._active_feat_set = all_selected_features
       
       # set the flags to make these changes effective. 
       self.decoder_change_flag = True
       self.feature_change_flag = True

    
    @classmethod
    def convex_feature_selection_by_obj_fraction(self, C, Q_diag, optimal_val, offset):
        """
            trying to solve the convex feature selection problem of the form
            minimize f(z) = log det (C.T Q^(-1) diag(z) C )
            subject to 0.001 <= theta <= 1

            Args:
                C (np.array): num_features by num_states
                Q (np.array): 

        """

        if offset == 0:
            return np.ones((C.shape[0]))

        d = C.shape[0]
        ones_d = np.ones((d,1))
        theta = cp.Variable((d,1))

        Q_diag_inv =  np.linalg.inv(Q_diag)

        constraints = [theta >=0.0,  theta <= 1, 
                    cp.log_det(C.T @ Q_diag_inv @cp.diag(theta) @ C) >= optimal_val - offset]

        feature_selection_objective = cp.Minimize( theta.T @ ones_d)
        feature_selection_problem = cp.Problem(feature_selection_objective, constraints)

        result = feature_selection_problem.solve()

        selected_values = theta.value.copy()

        return np.squeeze(selected_values), result

    
    
    
    @classmethod
    def convex_feature_selection_with_joint_smooth_sparse_goals(self, C, 
                                            Q_diag, 
                                            sparsity_coef,
                                            smoothness_coef,
                                            prior_matrix):

        num_features, num_states = C.shape

        # set up the variables
        ones_d = make_a_vector_of_ones(C, mode = "same_number_as_row")
        ones_prior = make_a_vector_of_ones(prior_matrix, mode = "same_number_as_col")
        
        
        Q_diag_inv =  np.linalg.inv(Q_diag)

        # set up the problem
        theta = cp.Variable((num_features,1))

        # add a condition that if the prior matrix is none, just ignore the smoothness_coef
        if prior_matrix.shape[1] > 0:

            feature_selection_objective = cp.Minimize(-cp.log_det(C.T @ Q_diag_inv @cp.diag(theta) @ C) \
                                                    + sparsity_coef * theta.T @ ones_d  \
                                                    - smoothness_coef * theta.T @ prior_matrix @ ones_prior )
        
        else: 
            feature_selection_objective = cp.Minimize(-cp.log_det(C.T @ Q_diag_inv @cp.diag(theta) @ C) \
                                        + sparsity_coef * 0.1 * theta.T @ ones_d)
            print("only doing sparsity objective")

        constraints = [theta >=0.0,  theta <= 1]
        feature_selection_problem = cp.Problem(feature_selection_objective, constraints)

        # solve the problem 
        result = feature_selection_problem.solve()
        selected_values = theta.value.copy()
        
        return np.squeeze(selected_values), result



class ReliabilityFeatureSelector(FeatureSelector):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.Q_diag_list = list()
        self._change_one_flag = True
        self.reliability_record = list()

    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        self.feature_measure_count += 1

        (C_hat,Q_diag) = self.measure_neurons_wz_intendedkin_and_spike(target_matrix, feature_matrix)

        self.determine_change_features()



    def determine_change_features(self):

        num_pre_est_pt = 60
        if self.feature_measure_count <= num_pre_est_pt:
            return 
        else:
            
            try: 
                q_diag_list = np.array(self._used_Q_diag_list)
                q_diag_list_last = q_diag_list[-1,:]
                C_mat = np.array(self._used_C_mat_list[-1])

                C_mat_norm  = np.linalg.norm(C_mat[:, (X_VEL_STATE_IND, Y_VEL_STATE_IND)], axis = 1)

                r_measure = C_mat_norm / q_diag_list_last

                print(r_measure)

                self.selected_feature_indices = self.threshold_features(r_measure, 75)

                # set the active feature set to True
                temp_active_set = np.full(self.N_TOTAL_AVAIL_FEATS, False)
                temp_active_set[self.selected_feature_indices] = True

                self._active_feat_set = temp_active_set

                
                self.decoder_change_flag = True
                self.feature_change_flag = True
            except:
                print(f'we have a problem with the noisy: {self.selected_feature_indices}')

                self.decoder_change_flag = False
                self.feature_change_flag = False


            return
        
        '''
        elif self.feature_measure_count < train_high_SNR_time:
            #if nothing else just returns sort of thing. 
            return

        
        N_high_SNR = 8
        feature_change_index_time = self.feature_measure_count - train_high_SNR_time
        selected_index = N_high_SNR + feature_change_index_time

        if selected_index > self.N_TOTAL_AVAIL_FEATS: return

        self._active_feat_set[:selected_index] = True

        self.decoder_change_flag = True
        self.feature_change_flag = True

        print(f'determine feature change: after set up {self._active_feat_set}')
        '''
        
    def measure_neurons_wz_intendedkin_and_spike(self, intended_kin, spike_counts):
        '''
        calculate the y-CX covariance matrix
        and then select the top 25 percentile features
        '''

        C_hat, Q_hat = kfdecoder.KalmanFilter.MLE_obs_model(intended_kin, spike_counts, include_offset=False)
        Q_diag = np.diag(Q_hat)
        self.Q_diag_list.append(Q_diag)
        self._used_C_mat_list.append(C_hat)

        return (C_hat, Q_diag)

    def threshold_features(self, Q_diag, per):
        Q_per = np.percentile(Q_diag,  per)
        return np.argwhere(Q_diag >= Q_per)

class SmoothReliabilityFeatureSelector(ReliabilityFeatureSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reliability_record = np.zeros(self.N_TOTAL_AVAIL_FEATS)

    def determine_change_features(self):

        num_pre_est_pt = 3
        if self.feature_measure_count <= num_pre_est_pt:
            return 
        else:
            
            try: 
                q_diag_list = np.array(self._used_Q_diag_list)
                q_diag_list_last = q_diag_list[-1,:]
                C_mat = np.array(self._used_C_mat_list[-1])

                C_mat_norm  = np.linalg.norm(C_mat[:, (X_VEL_STATE_IND, Y_VEL_STATE_IND)], axis = 1)

                r_measure = C_mat_norm / q_diag_list_last

                print(r_measure)

                self.selected_feature_indices = self.threshold_features(r_measure, 75)

                #increment the count of number of times.
                self.reliability_record[self.selected_feature_indices] = self.reliability_record[self.selected_feature_indices] + 1

                #get the N largest elements
                selected_ind = self.reliability_record.argsort()[-8:] #TODO,  magical number 8 is expected 8 most informative features


                # set the active feature set to True
                temp_active_set = np.full(self.N_TOTAL_AVAIL_FEATS, False)
                #TODO change this magical number to whatever number of features we'd like to use.
                
                temp_active_set[selected_ind] = True

                self._active_feat_set = temp_active_set

                
                self.decoder_change_flag = True
                self.feature_change_flag = True
            except:
                print(f'we have a problem with the noisy: {self.selected_feature_indices}')

                self.decoder_change_flag = False
                self.feature_change_flag = False


            return



class IterativeFeatureRemoval(FeatureSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #TODO make this dynamic changeable
        self._change_at_batch = 50
        self._target_active_set = kwargs['target_feature_set']
        self._feature_changed = False

    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        self.feature_measure_count += 1

        self.determine_change_features()

    def determine_change_features(self):
       
       #assume we selected the first few features

       if self.feature_measure_count <= self._change_at_batch:
           return

       if not self._feature_changed: 
           self._active_feat_set = np.copy(self._target_active_set)

           self.decoder_change_flag = True
           self.feature_change_flag = True

           self._feature_changed = True

from sklearn.linear_model import Lasso

class LassoFeatureSelector(FeatureSelector):
    
    DEFAULT_ALPHA = 1
    DEFAULT_MAX_ITERATION = 10000
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        #param related to lasso, default to 1
        self.current_lasso = kwargs['lasso_alpha'] if 'lasso_alpha' in kwargs.keys() else self.DEFAULT_ALPHA
        self.max_iter = kwargs['max_iter'] if 'max_iter' in kwargs.keys() else self.DEFAULT_MAX_ITERATION

        self._adaptive_lasso_flag = kwargs['adaptive_lasso_flag'] if 'adaptive_lasso_flag' in kwargs else False
        self._init_lasso_regression(self.current_lasso, 
                                   self.max_iter)

        if self._adaptive_lasso_flag:
            self._setup_lasso_alpha_curve(**kwargs)
        
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

        self.feature_measure_count += 1

        if len(feature_matrix) != len(target_matrix):
            feature_matrix = feature_matrix.T
            target_matrix = target_matrix.T

        # only select relevant features, in case the hand kinematics.
        
        target_states = [3,5] # 3 for x_vel and 5 for y_vel
        
        #fitted results to lasso_model._coef
        self.lasso_model.fit(feature_matrix, target_matrix[:, target_states])
        self.measure_ready = True
        
        #save to the history of measures
        self.history_of_weights.append(self.get_feature_weights())
    
    def _setup_lasso_alpha_curve(self, **kwargs):
        
        self._start_batch = kwargs['fs_start_batch'] if  'fs_start_batch' in kwargs else 10
        self._finish_batch = kwargs['fs_finish_batch'] if  'fs_finish_batch' in kwargs else 60
        
        self._start_val = kwargs['lasso_start_val'] if  'lasso_start_val' in kwargs else 1
        self._finish_val = kwargs['lasso_finish_val'] if  'fs_finish_val' in kwargs else 10
        
        def calc_linear_alpha_values(start_batch, start_val, 
                             finish_batch, finish_val, 
                             batch_count):
            """
            this is a ReLU function for alpha values.
            """
            slope = (start_val - finish_val)/(start_batch - finish_batch)

            if batch_count < start_batch:
                return start_val
            elif batch_count > finish_batch:
                return finish_val
            else:
                return slope * (batch_count - start_batch) + start_val

        # uses partial function to construct a function to calculate
        # alpha
        self.calc_current_lasso = functools.partial(calc_linear_alpha_values, 
                                    self._start_batch, self._start_val,
                                    self._finish_batch, self._finish_val)
        
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

    def save_feature_params(self):
        import aopy

        #prepare data dict

        data_dict = {
            'lasso_hist': self.history_of_weights,
        }

        if self._adaptive_lasso_flag:
            data_dict['start_batch'] = self._start_batch
            data_dict['start_val'] = self._start_val
            data_dict['finish_batch'] = self._finish_batch
            data_dict['finish_val'] = self._finish_val

        #save to data dict
        #mk temporary directory 
        import tempfile
        print(f'saving feature selection file name to {self.h5file.name}')
        aopy.data.save_hdf(tempfile.gettempdir(), self.h5file.name, data_dict, data_group="/feature_selection", append = True)


def run_exp_loop(exp,  **kwargs):
        # riglib.experiment: line 597 - 601
    #exp.next_trial = next(exp.gen)
    # -+exp._parse_next_trial()np.arraynp.array


    # we need to set the initial state
    # per fsm.run:  line 138


    # Initialize the FSM before the loop
    exp.set_state(exp.state)
    
    finished_trials = exp.calc_state_occurrences('wait')

    # Mark the beginning and end of the experiment
    exp.sync_event('EXP_START')


    while exp.state is not None:

                # inc cycle count
        exp.cycle_count += 1

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
            unselected_decodable_obs, decode = exp.bmi_system.feature_accumulator(
                neural_obs_k)
            
            if exp.is_feature_change():
                #take care of the decoder selection stuff             
                
                decodable_obs = exp.select_features(unselected_decodable_obs)
            else:
                decodable_obs = unselected_decodable_obs.copy()
            
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
                    exp.bmi_system.learner(unselected_decodable_obs.copy(), learner_state, target_state_k, exp.bmi_system.decoder.get_state(
                    ), task_state, state_order=exp.bmi_system.decoder.ssm.state_order)

            decoded_states[:, k] = exp.bmi_system.decoder.get_state()

            ############################
            # Update decoder parameters
            ############################
            if exp.bmi_system.learner.is_ready():
                batch_data = exp.bmi_system.learner.get_batch()
                batch_data['decoder'] = exp.bmi_system.decoder
                
                #for feature selection
                unselected_batch = np.copy(batch_data['spike_counts'])
                selected_batch = np.copy(unselected_batch[exp._active_feat_set,:])
                batch_data['spike_counts'] = selected_batch.copy()
                
                kwargs.update(batch_data)
                exp.bmi_system.updater(**kwargs)
                exp.bmi_system.learner.disable()
                
                #measure features. 
                if isinstance(exp, FeatureSelector):
                    exp.measure_features(unselected_batch,
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
                
                #take care of the decoder selection stuff
                if exp.is_decoder_change():
                    #only select the first four neurons
                    print(f'decoder changes here at {exp.cycle_count}')
                    exp.select_decoder_features(exp.decoder, debug = False)
                
                #record the current feature active set
                exp.record_feature_active_set(exp.decoder)


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
        exp.k_mat_params.append(np.copy(exp.decoder.filt.K))


        #save the decoder_state
        #from BMILoop.move_plant
        exp.task_data['decoder_state'] = exp.decoder.get_state(shape=(-1,1))
        
        #save bmi_data
        exp.task_data['update_bmi'] = update_flag

        #actually saving the state mean and state covariances
        exp.task_data['pred_state_P'] = exp.decoder.filt.pred_state_P
        exp.task_data['post_state_P'] = exp.decoder.filt.post_state_P



        # as well as plant data.
        plant_data = exp.plant.get_data_to_save()
        for key in plant_data:
            exp.task_data[key] = plant_data[key]

        # clda data handled in the above call.

        # save to the list hisory of data.
        exp.task_data_hist.append(exp.task_data.copy())
        

        
        #if exp is a submember of saveHDF, then save to it. 
        if isinstance(exp, SaveHDF):

            # Send task data to any registered sinks
            if exp.task_data is not None: exp.sinks.send("task", exp.task_data)

            #Send sync events
            if exp.sync_every_cycle:
                code = 1 << exp.sync_params['screen_sync_nidaq_pin']
            if exp.has_sync_event:
                exp.sinks.send("sync_events", exp.sync_event_record)
                code |= int(exp.sync_event_record['code'])
                exp.has_sync_event = False
            if exp.sync_every_cycle:
                exp.sync_clock_record['time'] = exp.cycle_count
                exp.sync_clock_record['timestamp'] = time.perf_counter() - exp.t0
                exp.sync_clock_record['prev_tick'] = exp.clock.get_time()
                exp.sinks.send("sync_clock", exp.sync_clock_record)

        #deal with the task count_down features
        if hasattr(exp, 'TOTAL_RUNNNING_TIME'):
            if exp.cycle_count == exp.total_frames:

                print(f'finish  at cycle_count: {exp.cycle_count}')
                exp.sync_event('EXP_END', event_data=0, immediate=True) # Signal the end of the experiment, even if it crashed

                print(exp.calc_trial_num())
                exp.state = None
                print('exit')

                # sort out the loop params.

        
    if exp.verbose:
        print("end of FSM.run, task state is", exp.state)


class FeatureAdapter():
    '''
    this class
    assumes the the extracter is the same
    and tries to preserve the features.

    this meant to be added to the list of features.
    '''
    def __init__(self):
        self.current_feature_indices = self.decoder.filt.n_features
        self.feature_change_hist = list()

    def change_bmi_components():
        pass

    def change_decoder(indices ):
        '''
        
        '''
        pass

    def transform_features(self):
        pass


        