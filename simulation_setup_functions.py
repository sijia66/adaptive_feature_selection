

import numpy as np
import scipy.stats as ss


def get_enc_setup(sim_mode = 'two_gaussian_peaks', tuning_level = 1, n_neurons = 4, 
                  bimodal_weight = [0.5, 0.5],
                  norm_var = [50,  10], # sufficient statistics for the first gaussian peak,
                  norm_var_2 = [100, 10], # similarly, sufficient stats for the 2nd gaussian peak.
                  ):
    '''
    sim_mode:str 
       std:  mn 20 neurons
       'toy' # mn 4 neurons

    tuning_level: float 
        the tuning level at which a particular direction the firng rate is tuned
        the higher the better
    bimodal_weight: a weight of each of the peak
    '''

    print(f'{__name__}: get_enc_setup has a tuning_level of {tuning_level} \n')

    N_STATES = 7  # 3 positions and 3 velocities and an offset
    if sim_mode == 'toy':
        #by toy, w mn 4 neurons:
            #first 2 ctrl x velo
            #lst 2 ctrl y vel
        # build a observer matrix
        N_NEURONS = 4
        N_STATES = 7  # 3 positions and 3 velocities and an offset
        # build the observation matrix
        sim_C = np.zeros((N_NEURONS, N_STATES))


        # control x positive directions
        sim_C[0, :] = np.array([0, 0, 0, tuning_level, 0, 0, 0])
        sim_C[1, :] = np.array([0, 0, 0, -tuning_level, 0, 0, 0])
        # control z positive directions
        sim_C[2, :] = np.array([0, 0, 0, 0, 0, tuning_level, 0])
        sim_C[3, :] = np.array([0, 0, 0, 0, 0, -tuning_level, 0])
        

    elif sim_mode ==  'std':
        # build a observer matrix
        N_NEURONS = 25
        N_STATES = 7  # 3 positions and 3 velocities and an offset
        # build the observation matrix
        sim_C = np.zeros((N_NEURONS, N_STATES))
        # control x positive directions
        sim_C[0, :] = np.array([0, 0, 0, tuning_level, 0, 0, 0])
        sim_C[1, :] = np.array([0, 0, 0, -tuning_level, 0, 0, 0])
        # control z positive directions
        sim_C[2, :] = np.array([0, 0, 0, 0, 0, tuning_level, 0])
        sim_C[3, :] = np.array([0, 0, 0, 0, 0, -tuning_level, 0])

    elif sim_mode == 'rot_90':
        #the directions are along the four axes
        N_NEURONS = n_neurons
        N_STATES = 7
        sim_C = _get_alternate_encoder_setup_matrix(N_NEURONS, N_STATES, tuning_level)

    elif sim_mode == 'rand':
        N_STATES = 7
        sim_C = _get_rand_encoder_matrix(n_neurons,  N_STATES, tuning_level)
    elif sim_mode == "two_gaussian_peaks":
        feature_weights = generate_bimodal(n_neurons, bimodal_weight, 
                                            norm_var[0], norm_var[1], 
                                            norm_var_2[0], norm_var_2[1])
        # sort the weights in descending order
        feature_weights = np.sort(feature_weights)[::-1]

        sim_C = _get_rand_encoder_matrix(n_neurons, 7, feature_weights)

        return (n_neurons, N_STATES, sim_C, feature_weights)
    else:
        raise Exception(f'not recognized mode {sim_mode}')
    
    return (n_neurons, N_STATES, sim_C)

def _get_alternate_encoder_setup_matrix(N_NEURONS, N_STATES, tuning_level):
    from itertools import cycle
    axial_angle_iterator = cycle([0, np.pi / 2, np.pi, np.pi * 3 / 2])

    X_VEL_IND = 3
    Y_VEL_IND = 5

    sim_C = np.zeros((N_NEURONS, N_STATES))
    x_weights = np.zeros(N_NEURONS)
    y_weights = np.zeros(N_NEURONS)

    for i in range(N_NEURONS):
        current_angle = next(axial_angle_iterator)
        x_weights[i] = np.cos(current_angle) * tuning_level
        y_weights[i] = np.sin(current_angle) * tuning_level

    sim_C[:,X_VEL_IND] = x_weights
    sim_C[:,Y_VEL_IND] = y_weights

    return sim_C

def _get_rand_encoder_matrix(n_neurons,  N_STATES, tuning_level):
    #sample 2 pi:
    prefered_angles_in_rad = np.random.uniform(low = 0, high = 2 * np.pi, size = n_neurons)

    sim_C = np.zeros((n_neurons, N_STATES))

    X_VEL_IND = 3
    Y_VEL_IND = 5

    #calculate the matrices
    sim_C[:,X_VEL_IND] = np.cos(prefered_angles_in_rad) * tuning_level
    sim_C[:,Y_VEL_IND] = np.sin(prefered_angles_in_rad) * tuning_level

    return sim_C



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


def generate_bimodal(N, w, mu_1, var_1, mu_2, var_2, seed = 0, debug = True):
  """
  generate N samples that are distributed by the mixture density
  w * Normal(mu_1, var_1) + (1-w) * Normal(mu_2, var_2)


  """
  # set the seed -> reproducible results
  np.random.seed(seed)
  if debug:
      pass
  
  norm_params = np.array([[mu_1, var_1],
                        [mu_2, var_2]])
  
  n_components = norm_params.shape[0]
    
  
  weights = np.array(w) / sum(w)

  # The indices are sampled by biasing the weights
  mixture_idx = np.random.choice(len(weights), size=N, replace=True, p=weights)

  # y is the mixture sample
  y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                    dtype=np.float64)
  
  return y


def generate_binary_features_by_thresholding(percent_high_SNR_noises,
                                            normal_val, 
                                            norm_val_2, feature_weights):
    """
    calculate the threshold from the given normal distributions and
    and apply the threshold to the feature weights.
    """


    # the average of the mean is the threshold
    norm_val_mean, norm_val_var = normal_val
    norm_val_2_mean, norm_val_2_val = norm_val_2
    threshold = (norm_val_mean + norm_val_2_mean) / 2

    # find the indices for those features above and below the threshold
    no_noise_neuron_ind = np.argwhere(feature_weights >= threshold)
    noise_neuron_ind = np.argwhere(feature_weights < threshold)

    # this is redundent, but also make two boolean arrays to keep track of the noise and non-noise features.
    # prusumbly, this would speed up indicing. 
    n_neurons = len(feature_weights)

    noise_neuron_list = np.full(n_neurons, False, dtype = bool)
    no_noise_neuron_list = np.full(n_neurons, False, dtype = bool)


    noise_neuron_list[noise_neuron_ind] = True
    no_noise_neuron_list[no_noise_neuron_ind] = True

    # set up the noise distribution
    # TODO, make this different for each distri
    # make percent of count into a list 
    percent_of_count_in_a_list = list()
    num_noises = len(percent_high_SNR_noises)

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



def config_feature_selector(FEATURE_SELECTOR_TYPE, feats, feats_2, **kwargs):
    
        # ## feature selector setup
    from feature_selection_feature import FeatureTransformer, TransformerBatchToFit
    from feature_selection_feature import FeatureSelector, LassoFeatureSelector, SNRFeatureSelector
    from feature_selection_feature import IterativeFeatureSelector, IterativeFeatureRemoval
    from feature_selection_feature import ReliabilityFeatureSelector


    #pass the real time limit on clock
    feats.append(FeatureSelector)
    
    if FEATURE_SELECTOR_TYPE == 'IterativeAddition':
        feats_2.append(IterativeFeatureSelector)

        feature_x_meth_arg = [
            ('transpose', None ),
        ]

        kwargs_feature = dict()
        kwargs_feature = {
            'transform_x_flag':False,
            'transform_y_flag':False,
            'n_starting_feats': kwargs["n_neurons"],
            'n_states':  7,
            "train_high_SNR_time": kwargs["train_high_SNR_time"],
        }

        print('kwargs will be updated in a later time')
        print(f'the feature adaptation project is tracking {kwargs_feature.keys()} ')
    elif FEATURE_SELECTOR_TYPE == 'IterativeRemoval':
        
        feats_2.append(IterativeFeatureRemoval)
        feature_x_meth_arg = [
            ('transpose', None ),
        ]

        kwargs_feature = dict()
        kwargs_feature = {
            'transform_x_flag':False,
            'transform_y_flag':False,
            'n_starting_feats': kwargs["n_neurons"],
            'n_states':  7,
            "train_high_SNR_time": kwargs["train_high_SNR_time"], 
            "target_feature_set": kwargs['target_feature_set']
        }
    else:
        raise ValueError('Unsupported feature selector type')

    return (feats, feats_2, feature_x_meth_arg, kwargs_feature)