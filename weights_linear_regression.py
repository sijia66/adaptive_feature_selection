import numpy as np
from riglib.bmi.kfdecoder import KalmanFilter
import statsmodels.api as sm



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


def calc_ssm_y_minus_x_beta(y:np.array, 
                               X: np.array,
                               beta:np.array ):
    """
    returns Sum(y - x * beta)**2
    """
    return np.sum((y - X @ beta)**2)


def calc_a_history_of_matrix_L2norms_along_first_axis(kf_C_history: np.array, 
                                    indices_to_sum:tuple, 
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
        print()

    kf_C_history = kf_C_history[:, :,indices_to_sum]

    history_of_L2norms = list()

    for ii in range(N_TIME_POINTS):
        extract_matrix = kf_C_history[ii,:,:]

        
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