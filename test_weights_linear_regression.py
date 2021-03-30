from weights_linear_regression import *

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

if __name__ == "__main__":
    test_calc_a_history_of_matrix_L2norms_along_first_axis()