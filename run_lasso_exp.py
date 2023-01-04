import numpy as np
from afs_clda_1000_neurons import run_iter_feat_addition, run_lasso_sims
from simulation_runs import run_convex_selection


exp_types = ['lasso', 'convex', 'joint_convex', 'joint_convex_init_feature']
exp_types_to_run = ['joint_convex_init_feature']

total_exp_time = 300# in seconds
N_NEURONS = 128

ROUND_DECIMALS = 3


if "lasso" in exp_types_to_run:

    data_dump_folder = \
    '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/theory_dev/'


    #random_seeds = range(10)
    #num_neurons = [32, 128, 512, 1024]

    random_seeds = [0]
    num_neurons = [128]
    lasso_alphas = 1

    #percent_high_SNR_noises = np.arange(0, 1.2, 0.2)
    percent_high_SNR_noises = [0.7]

    for a in lasso_alphas:

        for nn in num_neurons:  
            for noise in percent_high_SNR_noises:

                for s in random_seeds:

                        run_lasso_sims(total_exp_time = 1200, 
                                            lasso_alpha = a, 
                                            n_neurons = nn,
                                            random_seed=s, 
                                            percent_high_SNR_noises = [noise],
                                            data_dump_folder=data_dump_folder)

        print(f'finished exp with random seed {nn} with random seeds {s}')



if "convex" in exp_types_to_run:

    """
    convex basically means that we are doing the convex objective selection
    this setting is set as a string in FEATURE_SELETOR_TYPE
    """

    # noise scan
    data_dump_folder = \
    '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/convex_selection/logical_or/'
    mean_first_peak = 50
    mean_second_peak = 100
    std_of_peaks = 10

    print("********************************************")
    print("********************************************")
    print("********************************************")
    print(f'running experiment convex feature selection')
    run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                        data_dump_folder = data_dump_folder,
                        norm_val= [mean_first_peak, std_of_peaks],
                        norm_var_2= [mean_second_peak, std_of_peaks],
                        train_high_SNR_time = 20, #  60 batches or  1200 times)
                        FEATURE_SELETOR_TYPE='convex'
    )
    print("********************************************")
    print("********************************************")
    print("********************************************")


if "joint_convex" in exp_types_to_run:
    """
    convex basically means that we are doing the joint objective feature selection. 
    this setting is set as a string in FEATURE_SELETOR_TYPE = joint_convex
    """

    # noise scan
    # data_dump_folder = \
    # '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/convex_selection/grid_scan_sparsity_decay/'

    data_dump_folder = \
    '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/encoder_dev/'
    
    # we set up the neural populations
    mean_first_peak = 50
    mean_second_peak = 100
    std_of_peaks = 3

    #sparsity_array = np.arange(0.05, 0.15, 0.01)
    # smoothness_array = np.arange(0, 0.15, 0.025)
    sparsity_array = [0.11]

    smoothness_array = [0.125]
    num_lags_array = [3]
    decay_factor_array  = np.arange(0, 1.2, 0.2)

    # decay_factor_array = np.round(decay_factor_array, ROUND_DECIMALS)
    decay_factor_array = [0.8]


    for sparsity_val in sparsity_array:
        for smoothness_val in smoothness_array:
            for num_lag in num_lags_array:
                for decay_factor in  decay_factor_array:

                    # no one can escape the beauty of python one-liner, granted at the expense of line width
                    sparsity_val, smoothness_val = np.round(sparsity_val, ROUND_DECIMALS), np.round(smoothness_val, ROUND_DECIMALS)
                    
                    print("********************************************")
                    print("********************************************")
                    print("********************************************")
                    print(f'running experiment convex feature selection')
                    run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                        data_dump_folder = data_dump_folder,
                                        norm_val= [mean_first_peak, std_of_peaks],
                                        norm_var_2= [mean_second_peak, std_of_peaks],
                                        train_high_SNR_time
                                         = 10, #  60 batches or  1200 times)
                                        FEATURE_SELETOR_TYPE='joint_convex',
                                        threshold_selection = 0.5,
                                        objective_offset = 1,
                                        sparsity_coef = sparsity_val,
                                        smoothness_coef = smoothness_val,
                                        num_of_lags = num_lag,  #  this is the K in the formulation, the number of batch updated feature scores we expect it to be.
                                        past_batch_decay_factor = decay_factor,
                    )
                    print("********************************************")
                    print("********************************************")
                    print("********************************************")


if "joint_convex_init_feature" in exp_types_to_run:
    """
    convex basically means that we are doing the joint objective feature selection. 
    this setting is set as a string in FEATURE_SELETOR_TYPE = joint_convex
    """

    # noise scan
    # data_dump_folder = \
    # '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/convex_selection/grid_scan_sparsity_decay/'

    data_dump_folder = \
    '/home/sijia66/data/'
    
    # we set up the neural populations
    mean_first_peak = 50
    mean_second_peak = 100
    std_of_peaks = 3

    #sparsity_array = np.arange(0.05, 0.15, 0.01)
    # smoothness_array = np.arange(0, 0.15, 0.025)
    sparsity_array = [0.11]

    smoothness_array = [0.05]
    num_lags_array = [3]
    
    # decay_factor_array  = np.arange(0, 1.2, 0.2)
    # decay_factor_array = np.round(decay_factor_array, ROUND_DECIMALS)
    decay_factor_array = [0.5]


    for sparsity_val in sparsity_array:
        for smoothness_val in smoothness_array:
            for num_lag in num_lags_array:
                for decay_factor in  decay_factor_array:

                    # no one can escape the beauty of python one-liner, granted at the expense of line width
                    sparsity_val, smoothness_val = np.round(sparsity_val, ROUND_DECIMALS), np.round(smoothness_val, ROUND_DECIMALS)
                    
                    print("********************************************")
                    print("********************************************")
                    print("********************************************")
                    print(f'running experiment convex feature selection')
                    run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                        data_dump_folder = data_dump_folder,
                                        norm_val= [mean_first_peak, std_of_peaks],
                                        norm_var_2= [mean_second_peak, std_of_peaks],
                                        train_high_SNR_time
                                         = 10, #  60 batches or  1200 times)
                                        FEATURE_SELETOR_TYPE='joint_convex',
                                        threshold_selection = 0.5,
                                        objective_offset = 1,
                                        sparsity_coef = sparsity_val,
                                        smoothness_coef = smoothness_val,
                                        num_of_lags = num_lag,  #  this is the K in the formulation, the number of batch updated feature scores we expect it to be.
                                        past_batch_decay_factor = decay_factor,
                                        RANDOM_INITIAL_FEATURES=True,
                                        
                    )
                    print("********************************************")
                    print("********************************************")
                    print("********************************************")