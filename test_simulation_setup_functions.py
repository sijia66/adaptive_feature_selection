
import numpy as np
np.printoptions(precision=2, suppress=True)
from simulation_setup_functions import *


def test_make_new_sim():
    # Define inputs
    encoder_change_mode = 'drop_half_good_neurons'
    n_neurons = 8
    original_bimodal_weights = [0.8, 0.2]
    old_sim_c = np.zeros((n_neurons, 7))
    norm_var = [1, 0.01]
    norm_var_2 = [5, 0.01]
    random_seed = 42

    # Call the function to get the new sim
    new_sim = make_new_sim(encoder_change_mode, n_neurons, original_bimodal_weights,
                           old_sim_c, norm_var, norm_var_2, random_seed)

    # Check that the new sim has the correct shape
    assert new_sim.shape == old_sim_c.shape
    
    print(old_sim_c- new_sim)

    # Check that the first half of the neurons are the same as the old sim
    number_good_neurons = int(n_neurons * original_bimodal_weights[0])
    number_preserved_neurons = number_good_neurons // 2
    indices = np.arange(number_preserved_neurons)
    
    assert np.allclose(new_sim[indices, :], old_sim_c[indices, :])
    
    indices = np.arange(number_preserved_neurons, number_good_neurons)
    assert not np.allclose(new_sim[indices, :], old_sim_c[indices, :])

    # Check that the bottom half of the neurons are different from the old sim
    indices = np.arange(number_good_neurons, n_neurons)
    assert np.allclose(new_sim[indices, :], old_sim_c[indices, :])
    

import numpy as np

def test_swap_array_rows():
    # create a test array with 4 rows and 3 columns
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    
    # expected result after swapping the rows
    expected = np.array([[7, 8, 9], [10, 11, 12], [1, 2, 3], [4, 5, 6]])
    
    # call the function to swap the rows
    result = swap_array_rows(arr)
    
    # check that the resulting array is equal to the expected one
    assert np.array_equal(result, expected)
    
    # create a test array with 5 rows and 2 columns
    arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    
    # expected result after swapping the rows
    expected = np.array([[7, 8], [9, 10], [1, 2], [3, 4], [5, 6]])
    
    # call the function to swap the rows
    result = swap_array_rows(arr)
    
    print(result)
    
    # check that the resulting array is equal to the expected one
    assert np.array_equal(result, expected)
    

import numpy as np

def test_rotate_good_neurons():
    # Test case 1: small input matrix
    old_sim_c = np.array([[1, 2,3,4,5,6], [3, 4,5,6,7,8], [5, 6,7,8,9,10]])
    n_neurons = 3
    original_bimodal_weights = np.array([0.5, 0.5])
    expected_output = np.array([[1, 2,3,-6,5,4], [3,4,5,-8,7,6], [5,6,7,-10,9,8]])
    print(rotate_good_neurons(old_sim_c, n_neurons, original_bimodal_weights))
    assert np.allclose(rotate_good_neurons(old_sim_c, n_neurons, original_bimodal_weights), 
                       expected_output)
    


if __name__ == '__main__':
    test_make_new_sim()
