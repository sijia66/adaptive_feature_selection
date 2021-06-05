import numpy as np
from numpy.lib.function_base import _select_dispatcher
from sklearn.linear_model import Lasso

from weights import change_target_kalman_filter_with_a_C_mat
#import 
from icecream import ic


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

    def select_decoder_features(self, target_decoder, debug = True):
        '''
        
        '''

        self._change_one_flag = False 
        print(f'select_decoder_features: _change_one_flag = False')

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
        self._used_K_mat_list.append(np.copy(target_decoder.filt.K))
        self._used_C_mat_list.append(np.copy(self.used_C_mat))
        
        self.used_Q_diag[self._active_feat_set] = np.diag(np.copy(target_decoder.filt.Q))
        self._used_Q_diag_list.append(np.copy(self.used_Q_diag))

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
            

    def measure_neurons_wz_intendedkin_and_spike(self, intended_kin, spike_counts):
        '''
        calculate the y-CX covariance matrix
        and then select the top 25 percentile features
        '''

        C_hat, Q_hat = kfdecoder.KalmanFilter.MLE_obs_model(intended_kin, spike_counts, include_offset=False)
        Q_diag = np.diag(Q_hat)
        self.Q_diag_list.append(Q_diag)

        return Q_diag

    def threshold_features(self, Q_diag, per):
        Q_per = np.percentile(Q_diag,  per)
        return np.argwhere(Q_diag <= Q_per)

class IterativeFeatureSelector(FeatureSelector):

    def measure_features(self, feature_matrix, target_matrix):
        '''
        feature_matrix[np.array]: n_time_points by n_features
        target_matrix[np.array]: n_time_points by n_target fitting vars
        '''
        self.feature_measure_count += 1

        self.determine_change_features()

    def determine_change_features(self):
       
       #assume we selected the first few features.

       train_high_SNR_time = 60

       if self.feature_measure_count <= train_high_SNR_time:
           return


       N_high_SNR = 8
       feature_change_index_time = self.feature_measure_count - train_high_SNR_time
       selected_index = self.N_TOTAL_AVAIL_FEATS - N_high_SNR - feature_change_index_time

       if selected_index < 0: return

       selected_index = np.max(selected_index, 0)

       self._active_feat_set[selected_index:] = True

       self.decoder_change_flag = True
       self.feature_change_flag = True


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


        