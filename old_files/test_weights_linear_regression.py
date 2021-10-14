import numpy as np
from weights_linear_regression import *


def test_numpy_variance_from_an_array():
    a = np.array([[1,2],
                  [2,4],
                  [3,6]])
    
    var_along_col = a.var(axis = 0)
    print(f'{__name__}: {var_along_col}')
    
    expected_result = [2/3.0, 8 /3.0]
    print(expected_result)
    np.testing.assert_array_almost_equal(var_along_col, 
                                         expected_result)


def test_calc_a_history_of_matrix_L2norms_along_first_axis():
    kf_C = np.array([
        [0,0,0, 0.3, 0, 0.4, 1],
        [0,0,0, 0.5, 0, 1.2, 1]
    ])
    indices_to_sum = (3, 5)
    expected_from_one_matrix = np.array([0.5, 1.3])

    #create a second history point, as a scaled version of this
    SCALE_FAC = 0.01
    kf_C_history = np.expand_dims(kf_C, axis = 0)
    expected_from_one_matrix = np.expand_dims(expected_from_one_matrix, axis = 0)

    kf_C_history = np.concatenate(
        [kf_C_history, 
        kf_C_history * SCALE_FAC],
        axis = 0
    )
    expected_result = np.concatenate(
        (expected_from_one_matrix,
        expected_from_one_matrix * SCALE_FAC),
        axis = 0
    )


    kf_L2norm = calc_a_history_of_matrix_L2norms_along_first_axis(kf_C_history,indices_to_sum)
    assert np.allclose(expected_result, kf_L2norm)


def test_calc_1D_ssm_y_minus_x_beta():
    N_ONES  = 10 
    y = np.ones(N_ONES )

    X = np.ones(N_ONES ).T * 0.9
    X = np.tile(X, (2,1)).T

    beta = np.array([0.5, 0.5])

    expected_resuilt = 0.1

    np.testing.assert_array_almost_equal(expected_resuilt,
                                        calc_ssm_y_minus_x_beta(y, X, beta))

def test_calc_1D_ssm_y_minus_x_beta_two_outputs():
    N_ONES  = 10 
    y = np.ones(N_ONES )
    y = np.tile(y, (2,1)).T

    X = np.ones(N_ONES ).T * 0.9
    X = np.tile(X, (2,1)).T

    beta = np.array([0.5, 0.5])
    beta = np.tile(beta, (2,1)).T 

    expected_resuilt = 0.2

    np.testing.assert_array_almost_equal(expected_resuilt,
                                    calc_ssm_y_minus_x_beta(y, X, beta))

def test_calc_average_ssm_for_each_X_column():
    """
    test for y - x[:,i] * beta[:,i]
    """
    y = np.array([8,7]).reshape(-1,1)
    X = np.array([[1, 2,3],
                 [1,2,3]])
    beta = np.array([1, 1,1]).reshape(-1,1)

    expected_result = np.array([42.5, 30.5, 20.5])

    np.testing.assert_array_almost_equal(expected_result,
                                        calc_average_ssm_for_each_X_column(y, X, beta),
                                        )

def test_calc_average_ssm_for_each_X_column_normalized():

    y = np.array([8,7]).reshape(-1,1)
    X = np.array([[1, 2,3],
                 [1,2,3]])
    beta = np.array([1, 1,1]).reshape(-1,1)

    expected_result = np.array([170., 122.,  82.])

    np.testing.assert_array_almost_equal(expected_result,
                                    calc_average_ssm_for_each_X_column(y, X, beta, return_normalized=True),
                                    )


