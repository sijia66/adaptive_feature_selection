import numpy as np
from numpy.lib.shape_base import expand_dims
from riglib.bmi.kfdecoder import KalmanFilter
import statsmodels.api as sm
import sklearn


from sklearn.metrics import r2_score

X_VEL_STATE, Y_VEL_STATE,CONST_STATE  = 3, 5,6

def calc_average_ssm_for_each_X_column(y, X, beta,
                            return_normalized = False):
    """
    returns sum(y - x * beta)**2  / num_x_rows
    default to averaging over the number of rows

    if return_normalized: normalized by the uncentered variance of y

    returns a list of ssms of the shape (X_NUM_COLS, 0)
    """

    (X_NUM_ROWS, X_NUM_COLS) = X.shape

    result = list()

    for i in range(X_NUM_COLS):
        X_col = X[:, (i,)]
        beta_row = beta[(i,), :]
        
        result.append(calc_ssm_y_minus_x_beta(
            y, X_col, beta_row
        ))

    result = np.array(result)

    result = result / X_NUM_ROWS

    if return_normalized:  
        result = result / np.var(y)

    return result

def calc_average_ssm_for_each_y_row(y, C, X,
                            return_normalized = True):
    '''
    assumes a kalman filter observation model
    y = CX
    take each y row 
    the number of times are along the columns
    '''

    n_rows, n_cols = y.shape

    np.sum(y - C@X, axis = 2)



def calc_ssm_y_minus_x_beta(y:np.array, 
                               X: np.array,
                               beta:np.array ):
    """
    returns Sum(y - x * beta)**2
    """
    return np.sum((y - X @ beta)**2)


def calc_a_history_of_matrix_L2norms_along_first_axis(kf_C_history: np.array,  target_C = None,
                                    indices_to_sum:tuple = (3,5), 
                                    debug = False):
    """
    for each row,  calculate the L2 norm as specified by the indices_to_sum
    this function assumes kf_C_history  with a 3D array of the shape (N_TIME_POINTS, N_NEURONS, N_STATES)
    default to along the first axis

    output
    a_history_of_weights: np.array of the shape (N_TIME_POINTS, N_NEUORONS)
    """

    (N_TIME_POINTS, N_NEURONS, N_STATES) = kf_C_history.shape
    if debug: 
        print(f'input matrix shape {kf_C_history.shape}')
        print(kf_C_history)
        print(target_C)
        print()

    if target_C is None:
        target_C = np.copy(kf_C_history[0,:,:])
        print(target_C.shape)


    history_of_L2norms = list()

    for ii in range(N_TIME_POINTS):
        extract_matrix = kf_C_history[ii,:,:] - target_C

        extract_matrix = extract_matrix[:, indices_to_sum]

        history_of_L2norms.append(
            calc_matrix_L2norm_along_first_dimension(extract_matrix)
        )

    return np.array(history_of_L2norms)



def calc_matrix_L2norm_along_first_dimension(kf_C:np.array, debug = False ):

    L2_norm =  list()
    N_ROWS = kf_C.shape[0]

    if debug: print(kf_C)

    for ii in range(N_ROWS):
        row_vec  = kf_C[ii, :]
        L2_norm.append(calc_vector_L2_norm(row_vec))

    return np.array(L2_norm)



def calc_vector_L2_norm(row_vector: np.array,
                        indices_to_sum: tuple = None,
                        debug = False):
    
    if indices_to_sum is None: 
        indices_to_sum = range(len(row_vector))
    if debug: 
        print(f'calc_vector_L2_norm: {row_vector}')
        print(indices_to_sum)

    L2_norm_result = np.linalg.norm(row_vector[indices_to_sum])

    return L2_norm_result





# let's see if we can write the Jaccard score into a function
def calculate_jaccard_score(feature_vec_a: np.ndarray, feature_vec_b:np.ndarray) -> float:
    """
    we try to calcualte the ratio of both true, over both true, not (none of which is true)
    """
    
    assert feature_vec_a.ndim == 1
    assert feature_vec_a.ndim == 1
    assert feature_vec_a.size == feature_vec_b.size 
    
    n_element =  feature_vec_a.size
    
    # the top
    both_array_true = np.logical_and(feature_vec_a,  feature_vec_b)
    count_both_true = np.count_nonzero(both_array_true)
    
    # then both wrong
    both_array_false =  np.logical_and(feature_vec_a == False, feature_vec_b == False)
    count_both_false = np.count_nonzero(both_array_false)
    
    return count_both_true / (n_element -  count_both_false)



    
    
    
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


def calc_R2(spike_counts_batch, intended_velocities, debug = True):
    """
    calculates R2 by fitting spike_counts to intende_velocities.
    it also does type checking that allows handling of input types of either list or np.ndarray
    params:
        spike_counts_batch (num of batch by num units by num of data points per batch)
        intended_velocities (num of batch by num states by intended velocities)
    returns:
        r_values (num_batches)
        coefs: the fitted coefs. 
    """

    if isinstance(spike_counts_batch, list) and isinstance(intended_velocities, list):
        num_batches = len(spike_counts_batch)
    elif isinstance(spike_counts_batch, np.ndarray) and isinstance(intended_velocities, np.ndarray):
        num_batches = spike_counts_batch.shape[0]
    else:
        raise Exception(f"mixed input types of {type(spike_counts_batch)} and {type(intended_velocities)}")
    

    print(num_batches)

    coefs =  list()
    r_values = [None] * num_batches

    linear_reg_model = LinearRegression()

    for i in range(num_batches):

        if isinstance(spike_counts_batch, np.ndarray): 
            spike_counts_one_batch  = spike_counts_batch[i, :,:]
            intended_velocities_one_batch = intended_velocities[i,:,:]
        elif isinstance(spike_counts_batch, list):
            spike_counts_one_batch = spike_counts_batch[i]
            intended_velocities_one_batch = intended_velocities[i]


        linear_reg_model.fit(spike_counts_one_batch.T, intended_velocities_one_batch.T)

        #save the fitted coefs.
        coefs.append(linear_reg_model.coef_.copy())
        #print(linear_reg_model.coef_.shape)

        # calculate R2 values
        predicted_velocities = linear_reg_model.predict(spike_counts_one_batch.T)
        score=r2_score(intended_velocities_one_batch.T, predicted_velocities)

        r_values[i] = score

    return np.array(r_values), coefs


def calc_R2_with_sim_C( spike_counts_batch,intended_velocities, C_mat, active_set, 
                       remove_first_and_last_Batch = True, 
                       c_mat_remove_first_batch = True, 
                       select_only_vel_states = True,
                       select_features_with_active_set = False,
                       debug = True):
    
    
        # then we iterate through the batch sort of thing.
    NUM_LEARNER_BATCHES = intended_velocities.shape[0]
    
    if debug:
        print("intended_velocities", intended_velocities.shape)
        print("spike_counts_batch", len(spike_counts_batch))
        print("C_mat", C_mat.shape)
        print("active_set", active_set.shape)
    
    if remove_first_and_last_Batch:
        active_set = active_set[1:-1, :]
    if c_mat_remove_first_batch:
        C_mat = C_mat[1:, :, : ]
        
    if select_only_vel_states:
        C_mat = C_mat[:,:,(X_VEL_STATE, Y_VEL_STATE,CONST_STATE)]
        
    
    if debug:
        print("intended_velocities", intended_velocities.shape)
        print("spike_counts_batch", len(spike_counts_batch))
        print("C_mat", C_mat.shape)
        print("active_set", active_set.shape)
        

    
    R_2_over_batch = []
    
    for i in range(NUM_LEARNER_BATCHES):
        
        batch_vel =  intended_velocities[i,:,:]
        
        # check if we get the data from the list or the np.ndarray
        if type(spike_counts_batch) == list:
            batch_spike_counts = spike_counts_batch[i]
        else:
            batch_spike_counts = spike_counts_batch[i,:,:]
        
        #  we can only compare to what's being used in the calculation
        if select_features_with_active_set:
            if debug: print(batch_spike_counts.shape)
            batch_spike_counts = batch_spike_counts[active_set[i,:],:].T
        else:
            batch_spike_counts = batch_spike_counts[: ,:].T
        
        batch_c_mat = C_mat[i,:,:]
        
        selected_c_mat = batch_c_mat[active_set[i,:],:]
        

        estimated_spike_counts = selected_c_mat @ batch_vel
        
        score = r2_score(batch_spike_counts, estimated_spike_counts.T)
        
        R_2_over_batch.append(score)
        
    return R_2_over_batch