"""
this file works with decoder weights
the inputs to this functins are primarily np.arrays 
"""
import numpy as np
from riglib.bmi.kfdecoder import KalmanFilter
import statsmodels.api as sm
import sklearn

# we calculate the smoothness metric 
# copy and paste for now,  we can potentially do this online.

def calculate_feature_smoothness(matrix_feature_by_batch:np.ndarray, mode:str = "compare_to_first_batch") ->  np.ndarray:
    """
    use jaccard similarity score to compare later feature selection batches to the very first one batch
    mode: str: incremental between batches
    """
    # for now, only support two dim array, the array dim is defined as  feature by batch 
    assert matrix_feature_by_batch.ndim == 2
    
    if mode == "incremental":
        smoothness_score = np.zeros(matrix_feature_by_batch.shape[1] - 1)
        for i in range(matrix_feature_by_batch.shape[1] - 1):
            smoothness_score[i] = sklearn.metrics.jaccard_score(matrix_feature_by_batch[:,i], 
                                                                matrix_feature_by_batch[:,i+1])
    else: # otherwise, we compare to the very first one
        initial_batch = matrix_feature_by_batch[:, 0]
        smoothness_score = np.zeros(matrix_feature_by_batch.shape[1])
        for i in range(matrix_feature_by_batch.shape[1]):

            smoothness_score[i] = sklearn.metrics.jaccard_score(matrix_feature_by_batch[:,i], initial_batch)

    return smoothness_score
    
    
def calculate_feature_smoothness_multiple_conditions(matrix_cond_by_feature_by_batch:np.ndarray, **kwargs) -> np.ndarray:

    num_conds, _, _  = matrix_cond_by_feature_by_batch.shape

    smooth_batch_arrays = []

    for i in range(num_conds):
        smooth_batch_arrays.append(calculate_feature_smoothness(matrix_cond_by_feature_by_batch[i, :, :], **kwargs))

    return np.array(smooth_batch_arrays)


def change_target_kalman_filter_with_a_C_mat(target_kf, C:np.matrix, Q:np.matrix = np.nan, debug:bool = True):
    """
    this function replaces the observation matrix of target_kf to C and
    replace the noise model 
    """
    
    if not isinstance(target_kf, KalmanFilter):
        raise Exception(f'{target_kf} is not an instance of riglib.bmi.kfdecoder.KalmanFilter' )
    
    C_before = target_kf.C
    C_xpose_Q_inv_before = target_kf.C_xpose_Q_inv
    C_xpose_Q_inv_C_before = target_kf.C_xpose_Q_inv_C
    
    if Q is np.nan: Q = target_kf.Q
    
    #calculate the new terms
    C_xpose_Q_inv = C.T * np.linalg.pinv(Q)
    C_xpose_Q_inv_C = C.T * np.linalg.pinv(Q) * C
    
    #reassign
    target_kf.C = C
    target_kf.Q = Q
    target_kf.C_xpose_Q_inv = C_xpose_Q_inv
    target_kf.C_xpose_Q_inv_C = C_xpose_Q_inv_C
    
    if debug:
        print('C matrix before')
        print(C_before)
        
        print('C matrix after')
        print(target_kf.C )
        
        print('C_xpose_Q_inv before:')
        print(C_xpose_Q_inv_before)
        
        print('C_xpose_Q_inv_C after:')
        print(target_kf.C_xpose_Q_inv)
              
        print('C_xpose_Q_inv_C_before:')
        print(C_xpose_Q_inv_C_before)
        
        print('C_xpose_Q_inv_C after:')
        print(target_kf.C_xpose_Q_inv_C)
              


def _calc_tuning_angle_2D(mat_kf_c: np.array, vel_ind: tuple = (3,  5)) -> np.array:
    '''
    mat_kf_c: N_Neuron by N_states
    assume with each row with [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, constant]
    '''
    
    N_Neuron, N_States = mat_kf_c.shape
                         
    COL_X, COL_Z = vel_ind
    
    tuning_angles = np.zeros(N_Neuron)
    
    tuning_angles = [np.rad2deg(np.arctan(neuron_row[COL_Z] / neuron_row[COL_X])) for neuron_row in mat_kf_c]
    
    return np.array(tuning_angles)

def calc_tuning_angle(mat_kf_c: np.array, vel_ind: tuple = (3,  5)) -> np.array:
    if len(mat_kf_c.shape) == 2:
        return np.array(_calc_tuning_angle_2D(mat_kf_c, vel_ind = vel_ind))
    elif len(mat_kf_c.shape) ==3:
        tuning_angles = [_calc_tuning_angle_2D(mat_kf_c_slice, vel_ind = vel_ind) for mat_kf_c_slice in mat_kf_c]
        return np.array(tuning_angles)
    else:
        print(f'{__name__}: does not know how to deal with mat C with the shape of {mat_kf_c.shape}')

def calc_p_values_for_spike_batches_use_intended_kin(spike_counts_batch, intended_kin, reg_states =  None):
    """
    given a history of neuron batch  and intended firing rates, calculate its  p-values across neurons and time points
    """

    (N_UPDATE_BATCHES_SPIKE, N_NEURONS, BATCH_SIZE_SPIKE) = spike_counts_batch.shape
    (N_UPDATE_BATCHES_KIN, N_STATES, BATCH_SIZE_KIN) = intended_kin.shape

    if reg_states is None: 
        X_VEL_STATE_IND = 3
        Y_VEL_STATE_IND = 5
        print(f'regress on the states index : {X_VEL_STATE_IND} and {Y_VEL_STATE_IND}' )


    if not (N_UPDATE_BATCHES_SPIKE == N_UPDATE_BATCHES_KIN): 
        raise Exception(f'the spike count length of {N_UPDATE_BATCHES_SPIKE} does not equal to {N_UPDATE_BATCHES_KIN}' )

    if not (BATCH_SIZE_KIN == BATCH_SIZE_SPIKE): 
        raise Exception(f'the batch size  of {BATCH_SIZE_KIN} does not equal to {BATCH_SIZE_SPIKE}' )

    # the number of columns of in the design matrix
    N_TRACKED_STATES = 3 # ( CONST, X_VEL_STATE_IND, Y_VEL_STATE_IND)
    batch_pvalue_matrix = np.empty((N_UPDATE_BATCHES_SPIKE,N_NEURONS, N_TRACKED_STATES))
    batch_pvalue_matrix[:] = np.NaN #set all values to NaN


    #calculate the p-value matrix across the batches
    for batch_of_interest in range(N_UPDATE_BATCHES_SPIKE):
        
        #obtain the batches
        intended_kin_one_batch = intended_kin[batch_of_interest]
        intended_kin_one_batch_2D = intended_kin_one_batch[(X_VEL_STATE_IND, Y_VEL_STATE_IND),:]
        spike_count_one_batch = spike_counts_batch[batch_of_interest]

        #transpose the matrices so that timepoints along the rows
        spike_count_one_batch = spike_count_one_batch.T
        intended_kin_one_batch_2D = intended_kin_one_batch_2D.T

        batch_pvalue_matrix[batch_of_interest, :, :] \
        = calc_single_batch_p_values_by_fitting_kinematics_to_spike_counts(intended_kin_one_batch_2D, spike_count_one_batch)
    
    return batch_pvalue_matrix


    

def calc_single_batch_p_values_by_fitting_kinematics_to_spike_counts(intended_kin_one_batch_2D:np.array,
                                                                        spike_count_one_batch:np.array):
    """
    this function iteratively fits an ols of the states to the firing rates 
    spike_count_one_batch = reg_coeff * intended_kin_one_batch_2D + error
    and returns a neuron_batch_pvalue_matrix of N_NEURONS by (N_STATES + 1) 
    
    inputs:
    intended_kin_one_batch_2D: N_TIME_POINTS by N_NEURONS
    spike_count_one_batch: by N_TIME_POINTS by N_STATES 
    
    output:
    neuron_batch_pvalue_matrix: N_NEURONS by (N_STATES + 1) 
    """

    

    #pre-initialize to speed up
    #after the transpose, we look at how many columns
    (N_TIME_POINTS, N_Neurons) = spike_count_one_batch.shape
    (N_TIME_POINTS_STATES, N_STATES_ORIGINAL) = intended_kin_one_batch_2D.shape

    if not have_same_num_rows(spike_count_one_batch, intended_kin_one_batch_2D):
        raise Exception(f'the time points do not match, neurons have {N_TIME_POINTS} time points and states have {N_TIME_POINTS_STATES}')

    X = intended_kin_one_batch_2D
    N_STATES = N_STATES_ORIGINAL + 1 # add 1 for the intersept

    neuron_batch_pvalue_matrix = np.empty((N_Neurons, N_STATES))
    neuron_batch_pvalue_matrix[:] = np.NaN #set all values to NaN

    for n_roi_neuron in range(N_Neurons):

        spike_count_one_batch_one_neuron = spike_count_one_batch[:,n_roi_neuron]
        
        (neuron_batch_coeff, neuron_batch_pvalue, sig_flag, sig_p_value_indices) = fit_ols(spike_count_one_batch_one_neuron, X)
        neuron_batch_pvalue_matrix[n_roi_neuron,:] = neuron_batch_pvalue
        
    return neuron_batch_pvalue_matrix


def fit_ols(y, X, 
            significance_level = 0.05,
           add_constant_term_flag = True,
           force_add_constant = True):
    """
    fit  X to y in the ordinary least square sense
    and returns  a tuple  of (neuron_batch_coeff, neuron_batch_pvalue, 
                                sig_flag, sig_p_value_indices)
    """
    #pre-fit: check the same number of rows
    if not have_same_num_rows(y, X): 
            raise Exception(f'y and X do not have the same number of rows') 

    #pre-fit: add the constant one to the states (X)'
    if  add_constant_term_flag: 
        if force_add_constant:
            X = sm.add_constant(X, has_constant='add')
        else:
            X = sm.add_constant(X, has_constant='raise')

    #fit using statsmodels.ols')
    ols_spike_from_states = sm.OLS(y, X)
    results_ols_spike_from_states = ols_spike_from_states.fit()

    #post-fit: get the coefficients
    neuron_batch_coeff = results_ols_spike_from_states.params
    #('the p-values test against if the coefficients are zero')
    neuron_batch_pvalue = results_ols_spike_from_states.pvalues

    sig_p_value_indices = np.array(get_significant_index(neuron_batch_pvalue,
                                                significance_level = significance_level))

    #return if any of the indix is empty
    sig_flag = not (sig_p_value_indices.size == 0)

    return (neuron_batch_coeff, 
           neuron_batch_pvalue,
           sig_flag,
           sig_p_value_indices)


def get_significant_index(p_value_array: np.array, 
                          significance_level = 0.05, 
                         compare_mode = 'smaller_or_equal'):
    '''
    compare each element in the p_value_list against signifance level. 
    mode defaults to smaller_than for comparing p_value
    '''
    
    if compare_mode == 'smaller_or_equal': 
        sig_test_list = np.where(p_value_array <= significance_level)
        sig_test_list = np.array(np.squeeze(sig_test_list))
        return sig_test_list
    elif compare_mode == 'larger_or_equal':
        return np.where(p_value_array >= significance_level)
    elif compare_mode == 'equal':
        return np.where(p_value_array == significance_level)
    else:
        raise Exception(f'unknown mode {compare_mode}')


def is_any_pvalue_significant(p_value_array: np.array, 
                              significance_level = 0.05,
                              compare_mode = 'smaller_or_equal'):
    """
    given the p_value_array
    check if any of the element is smaller than the significance_level (default to 0.05)
    """
    sig_test_list = get_significant_index(p_value_array,
                                           significance_level,
                                           compare_mode)
    
    
    return not (len(sig_test_list) == 0)



def have_same_num_rows(x: np.array, y:np.array):
    """
    check if x and y have the same number of rows
    """
    y_num_rows = x.shape[0]
    x_num_rows = y.shape[0]
    return y_num_rows == x_num_rows
    