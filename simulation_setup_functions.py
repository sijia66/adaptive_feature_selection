

import numpy as np

def generate_binary_feature_distribution(percent_high_SNR_noises, n_neurons, fraction_snr):


    num_noises = len(percent_high_SNR_noises)


    percent_high_SNR_noises_labels = [f'{s:.2f}' for s in percent_high_SNR_noises]

    mean_firing_rate_low = 50
    mean_firing_rate_high = 100
    noise_mode = 'fixed_gaussian'

    neuron_types = ['noisy', 'non_noisy']

    n_neurons_no_noise_group = int(n_neurons * fraction_snr)
    n_neurons_noisy_group = n_neurons - n_neurons_no_noise_group


    no_noise_neuron_ind = np.arange(n_neurons_no_noise_group)
    noise_neuron_ind = np.arange(n_neurons_no_noise_group, n_neurons_noisy_group + n_neurons_no_noise_group)

    neuron_type_indices_in_a_list = [
        noise_neuron_ind, 
        no_noise_neuron_ind
    ]


    noise_neuron_list = np.full(n_neurons, False, dtype = bool)
    no_noise_neuron_list = np.full(n_neurons, False, dtype = bool)


    noise_neuron_list[noise_neuron_ind] = True
    no_noise_neuron_list[no_noise_neuron_ind] = True



    N_TYPES_OF_NEURONS = 2

    print('We have two types of indices: ')
    for t,l in enumerate(neuron_type_indices_in_a_list): print(f'{neuron_types[t]}:{l}')



    # make percent of count into a list 
    percent_of_count_in_a_list = list()

    for i in range(num_noises):
        percent_of_count = np.ones(n_neurons)[:, np.newaxis]

        percent_of_count[noise_neuron_ind] = 1
        percent_of_count[no_noise_neuron_ind] = percent_high_SNR_noises[i]
        
        percent_of_count_in_a_list.append(percent_of_count)
    
    return (percent_of_count_in_a_list, no_noise_neuron_ind, noise_neuron_ind, no_noise_neuron_list, noise_neuron_list)


def get_KF_C_Q_from_decoder(first_decoder):
    """
    get the decoder matrices C, Q from the decoder instance
    
    Args:
        first_decoder: riglib.bmi.decoder.
    Returns:
        target_C, target_Q: np.ndarray instances
    """
    target_C = first_decoder.filt.C
    target_Q = np.copy(first_decoder.filt.Q)
    diag_val = 10000
    np.fill_diagonal(target_Q, diag_val)
    
    return (target_C, target_Q)