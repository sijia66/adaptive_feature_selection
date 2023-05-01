
import numpy as np
np.printoptions(precision=2, suppress=True)
from simulation_setup_functions import *


def test_make_new_sim():
    # Define inputs
    encoder_change_mode = 'drop_half_good_neurons'
    n_neurons = 10
    original_bimodal_weights = [0.8, 0.2]
    old_sim_c = np.random.rand(n_neurons, 7)
    norm_var = [0.1, 0.2]
    norm_var_2 = [0.3, 0.4]
    random_seed = 42

    # Call the function to get the new sim
    new_sim = make_new_sim(encoder_change_mode, n_neurons, original_bimodal_weights,
                           old_sim_c, norm_var, norm_var_2, random_seed)

    # Check that the new sim has the correct shape
    assert new_sim.shape == old_sim_c.shape

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