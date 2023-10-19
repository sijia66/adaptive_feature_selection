import numpy as np
from simulation_runs import run_convex_selection


exp_types = [
             'feature_gap_scan',
             'encoder_swap',
             'lasso', 
             'convex',
             'joint_convex', 
             'joint_convex_init_feature', 
             'joint_convex_encoder_change',
             'compare_convex_smooth',
             'full_feature_tracking']
exp_types_to_run = ['full_feature_tracking']

total_exp_time = 600# in seconds
N_NEURONS = 128

ROUND_DECIMALS = 3

if "feature_gap_scan" in exp_types_to_run:
    # actually running the experiments
# data saving stuff

    data_dump_folder = \
    '/home/aolab/sijia/data/figure2_simulation_setup/'
    # gap difference
    # exp_type = 'gap_difference'
    mean_first_peak = 50
    std_of_peaks = 3
    mean_differences = np.arange(0, 70,  step = 10 )
    
    mean_differences = [0, 60]
    
    for mean_diff in mean_differences:
        mean_second_peak = mean_first_peak + mean_diff

        print("********************************************")
        print("********************************************")
        print("********************************************")

        print(f'running experiment with second peak at {mean_second_peak}')
        run_convex_selection(total_exp_time = total_exp_time, 
                    data_dump_folder=data_dump_folder,
                    encoder_change_mode = "same", # we don't want to change the encoder
                    FEATURE_SELETOR_TYPE='full', # this is the default setting and does not do anything
                    RANDOM_INITIAL_FEATURES = True,
                    number_of_features = 32,
                    n_neurons = N_NEURONS,   
                    norm_val= [mean_first_peak, std_of_peaks],
                    norm_var_2= [mean_second_peak, std_of_peaks],
                    train_high_SNR_time  = 10, #  60 batches or  1200 times)
                    )


        print("********************************************")
        print("********************************************")
        print("********************************************")
        
if "encoder_swap" in exp_types_to_run:
    # actually running the experiments
# data saving stuff

    data_dump_folder = \
    '/home/aolab/sijia/data/figure2_simulation_setup/'
    # gap difference
    # exp_type = 'gap_difference'
    mean_first_peak = 50
    mean_second_peak = 110
    std_of_peaks = 3
    NUM_INITIAL_FEATURES = 32
    
    ENCODER_CHANGE_MODE = "swap_top_and_bottom"
    change_sim_c_at_cycle = 18000 # 

    run_convex_selection(total_exp_time = total_exp_time, 
                data_dump_folder=data_dump_folder,
                encoder_change_mode = ENCODER_CHANGE_MODE, # we don't want to change the encoder
                change_sim_c_at_cycle = change_sim_c_at_cycle,
                FEATURE_SELETOR_TYPE='full', # this is the default setting and does not do anything
                RANDOM_INITIAL_FEATURES = False,
                RANDOM_INITIAL_FEATURES_COUNT = NUM_INITIAL_FEATURES,
                number_of_features = 32,
                init_feat_first_or_last = "first",
                n_neurons = N_NEURONS,   
                norm_val= [mean_first_peak, std_of_peaks],
                norm_var_2= [mean_second_peak, std_of_peaks],
                train_high_SNR_time  = 10, #  60 batches or  1200 times)
                )


if "full_feature_tracking" in exp_types_to_run:
    
    data_dump_folder = \
    '/home/aolab/sijia/data/figure3_lasso/'
    
    # updater_type = "smooth_batch"
    updater_type = "smooth_batch_with_full_feature"
    
    random_seed = 0
    
    # we set up the neural populations
    mean_first_peak = 50
    mean_second_peak = 100
    std_of_peaks = 3
    
    feature_selector_type = 'joint_convex'


    #sparsity_array = np.arange(0.05, 0.15, 0.01)
    # smoothness_array = np.arange(0, 0.15, 0.025)
    sparsity_array = [0.125]

    smoothness_array = [0.05]
    num_lags_array = [3]

    # decay_factor_array = np.round(decay_factor_array, ROUND_DECIMALS)
    decay_factor_array = [0.5]

    
    number_of_features = 32
    encoder_change_mode = "shuffle_rows"
    change_sim_c_at_cycle = 18000 # 



        # lasso feature selection
    #lasso_alphas = [0.01, 0.1, 1, 10]
    lasso_alphas = [10]
    lasso_thresholds = [0,1]
    number_of_features_array = [32, 64]
    lasso_threshold = 0
    
    feature_selector_type = 'lasso'
    
    updater_type = "smooth_batch"
    for a in lasso_alphas:
        for lasso_threshold, number_of_features in zip(lasso_thresholds, number_of_features_array):
    
            run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                 random_seed=random_seed,
                            data_dump_folder = data_dump_folder,
                            norm_val= [mean_first_peak, std_of_peaks],
                            norm_var_2= [mean_second_peak, std_of_peaks],
                            train_high_SNR_time= 10, #  60 batches or  1200 times)
                            FEATURE_SELETOR_TYPE=feature_selector_type,
                            UPDATER_TYPE = updater_type,
                            lasso_alpha = a, 
                            lasso_threshold = lasso_threshold,
                            number_of_features = number_of_features,
                            RANDOM_INITIAL_FEATURES=False,
                            encoder_change_mode = encoder_change_mode,
                            change_sim_c_at_cycle = change_sim_c_at_cycle,
                            )
    
    updater_type = "smooth_batch_with_full_feature"
    for a in lasso_alphas:
        for lasso_threshold, number_of_features in zip(lasso_thresholds, number_of_features_array):

            run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                 random_seed=random_seed,
                            data_dump_folder = data_dump_folder,
                            norm_val= [mean_first_peak, std_of_peaks],
                            norm_var_2= [mean_second_peak, std_of_peaks],
                            train_high_SNR_time= 10, #  60 batches or  1200 times)
                            FEATURE_SELETOR_TYPE=feature_selector_type,
                            UPDATER_TYPE = updater_type,
                            lasso_alpha = a, 
                            lasso_threshold = lasso_threshold,
                            number_of_features = number_of_features,
                            RANDOM_INITIAL_FEATURES=False,
                            encoder_change_mode = encoder_change_mode,
                            change_sim_c_at_cycle = change_sim_c_at_cycle,
                            )
                    



if "lasso" in exp_types_to_run:

    data_dump_folder = \
    '/home/aolab/sijia/data/'

        # we set up the neural populations
    mean_first_peak = 50
    mean_second_peak = 100
    std_of_peaks = 3

    lasso_alphas = [0.01, 0.1, 1, 10]
    lasso_alphas = [20]
    
    norm_val= [mean_first_peak, std_of_peaks]
    norm_var_2= [mean_second_peak, std_of_peaks]   

    adaptive_lasso = False    
    lasso_thresholds = [0,1,2]
    # lasso_thresholds = [1,2]
    
    for a in lasso_alphas:
        for lasso_threshold in lasso_thresholds:
            run_convex_selection(total_exp_time = total_exp_time, 
                        data_dump_folder=data_dump_folder,
                        FEATURE_SELETOR_TYPE='lasso',
                        RANDOM_INITIAL_FEATURES = True,
                        lasso_alpha = a, 
                        lasso_threshold = lasso_threshold,
                        number_of_features = 32,
                        adaptive_lasso = adaptive_lasso,  
                        n_neurons = N_NEURONS,   
                        norm_val= [mean_first_peak, std_of_peaks],
                        norm_var_2= [mean_second_peak, std_of_peaks],
                        train_high_SNR_time  = 10, #  60 batches or  1200 times)
                        )




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
    '/home/sijia66/data/part2_random_start_bottleneck_num_features/'
    
    # we set up the neural populations
    mean_first_peak = 10
    mean_second_peak = 100
    std_of_peaks = 3
  
        
    sparsity_array = [0.06]
    # sparsity_array = np.arange(0.05, 0.15, 0.01)
    # decay_factor_array  = np.arange(0, 1.2, 0.2)
    # decay_factor_array = np.round(decay_factor_array, ROUND_DECIMALS)
    decay_factor_array = [0.2]


    #smoothness_array =  np.arange(0.025, 0.15, 0.025)
    smoothness_array = [0.1]
    num_lags_array = [3]
    
    random_seeds = [0]
    num_of_features_array  = [16, 32, 64, 96, 128]   # specify how many features we want to use, or None


    for sparsity_val in sparsity_array:
        for smoothness_val in smoothness_array:
            for num_lag in num_lags_array:
                for decay_factor in  decay_factor_array:
                    
                    for random_seed in random_seeds:
                        
                        for num_of_features in num_of_features_array:

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
                                                number_of_features = num_of_features,
                                                RANDOM_INITIAL_FEATURES=True,
                                                random_seed=random_seed
                                                
                            )
                            print("********************************************")
                            print("********************************************")
                            print("********************************************")


if "joint_convex_encoder_change" in exp_types_to_run:
    """
    we are also changing the features of the encoder, and see how the feature selection performs
    """

    # noise scan
    # data_dump_folder = \
    # '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/convex_selection/grid_scan_sparsity_decay/'
    
    data_dump_folder = \
    '/home/sijia66/data/encoder_dev/'
    
    random_seed = 2
    
    # we set up the neural populations
    mean_first_peak = 50
    mean_second_peak = 100
    std_of_peaks = 3
    
    change_sim_c_at_cycle = 18000 # 
    #encoder_change_mode = "drop_half_good_neurons"
    # encoder_change_mode = "swap_tuning"
    # encoder_change_mode = "shuffle_rows"
    encoder_change_mode= "change_to_zeros"
    
    
    
    # # without feature selection
    feature_selector_type = "full"
    
        

    run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                        data_dump_folder = data_dump_folder,
                        norm_val= [mean_first_peak, std_of_peaks],
                        norm_var_2= [mean_second_peak, std_of_peaks],
                        train_high_SNR_time
                            = 10, #  60 batches or  1200 times)
                        FEATURE_SELETOR_TYPE=feature_selector_type,
                        RANDOM_INITIAL_FEATURES=False,
                        encoder_change_mode = encoder_change_mode,
                        change_sim_c_at_cycle = change_sim_c_at_cycle,
                        random_seed=random_seed
    )
    print("********************************************")
    
    

    feature_selector_type = 'joint_convex'
    number_of_features = 32

    #sparsity_array = np.arange(0.05, 0.15, 0.01)
    # smoothness_array = np.arange(0, 0.15, 0.025)
    sparsity_array = [0.125]

    smoothness_array = [0.05]
    num_lags_array = [3]

    # decay_factor_array = np.round(decay_factor_array, ROUND_DECIMALS)
    decay_factor_array = [0.5]
    
    for sparsity_val in sparsity_array:
        for smoothness_val in smoothness_array:
            for num_lag in num_lags_array:
                for decay_factor in  decay_factor_array:

                    # no one can escape the beauty of python one-liner, granted at the expense of line width
                    sparsity_val, smoothness_val = np.round(sparsity_val, ROUND_DECIMALS), np.round(smoothness_val, ROUND_DECIMALS)
                    
                    print("********************************************")
                    print(f'running experiment convex feature selection')
                    run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                        data_dump_folder = data_dump_folder,
                                        norm_val= [mean_first_peak, std_of_peaks],
                                        norm_var_2= [mean_second_peak, std_of_peaks],
                                        train_high_SNR_time
                                         = 10, #  60 batches or  1200 times)
                                        FEATURE_SELETOR_TYPE='joint_convex',
                                        number_of_features = number_of_features,
                                        threshold_selection = 0.5,
                                        objective_offset = 1,
                                        sparsity_coef = sparsity_val,
                                        smoothness_coef = smoothness_val,
                                        num_of_lags = num_lag,  #  this is the K in the formulation, the number of batch updated feature scores we expect it to be.
                                        past_batch_decay_factor = decay_factor,
                                        RANDOM_INITIAL_FEATURES=False,
                                        encoder_change_mode = encoder_change_mode,
                                        change_sim_c_at_cycle = change_sim_c_at_cycle,
                                        random_seed=random_seed
                    )
                    print("********************************************")
    
    # lasso feature selection
    #lasso_alphas = [0.01, 0.1, 1, 10]
    lasso_alphas = [10]
    lasso_thresholds = [2.5]
    
    feature_selector_type = 'lasso'
    
    for a in lasso_alphas:
        for lasso_threshold in lasso_thresholds:
    
            run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                 random_seed=random_seed,
                            data_dump_folder = data_dump_folder,
                            norm_val= [mean_first_peak, std_of_peaks],
                            norm_var_2= [mean_second_peak, std_of_peaks],
                            train_high_SNR_time= 10, #  60 batches or  1200 times)
                            FEATURE_SELETOR_TYPE=feature_selector_type,
                            lasso_alpha = a, 
                            lasso_threshold = lasso_threshold,
                            RANDOM_INITIAL_FEATURES=False,
                            encoder_change_mode = encoder_change_mode,
                            change_sim_c_at_cycle = change_sim_c_at_cycle,
                            )
            



if "compare_convex_smooth" in exp_types_to_run:
    
    data_dump_folder = \
    '/home/sijia66/data/encoder_dev/'
    
    random_seed = 0
    
    # we set up the neural populations
    mean_first_peak = 50
    mean_second_peak = 100
    std_of_peaks = 3
    
    feature_selector_type = 'joint_convex'


    #sparsity_array = np.arange(0.05, 0.15, 0.01)
    # smoothness_array = np.arange(0, 0.15, 0.025)
    sparsity_array = [0.125]

    smoothness_array = [0.05]
    num_lags_array = [3]

    # decay_factor_array = np.round(decay_factor_array, ROUND_DECIMALS)
    decay_factor_array = [0.5]
    
    number_of_features = 32
    
    encoder_change_mode = "shuffle_rows"
    change_sim_c_at_cycle = 18000 # 

    # we don't smooth the matrices
    for sparsity_val in sparsity_array:
        for smoothness_val in smoothness_array:
            for num_lag in num_lags_array:
                for decay_factor in  decay_factor_array:

                    # no one can escape the beauty of python one-liner, granted at the expense of line width
                    sparsity_val, smoothness_val = np.round(sparsity_val, ROUND_DECIMALS), np.round(smoothness_val, ROUND_DECIMALS)
                    
                    print("********************************************")
                    print(f'running experiment convex feature selection')
                    run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                        data_dump_folder = data_dump_folder,
                                        norm_val= [mean_first_peak, std_of_peaks],
                                        norm_var_2= [mean_second_peak, std_of_peaks],
                                        train_high_SNR_time
                                         = 10, #  60 batches or  1200 times)
                                        FEATURE_SELETOR_TYPE='joint_convex',
                                        number_of_features = number_of_features,
                                        threshold_selection = 0.5,   
                                        objective_offset = 1,  
                                        sparsity_coef = sparsity_val,
                                        smoothness_coef = smoothness_val,
                                        num_of_lags = num_lag,  #  this is the K in the formulation, the number of batch updated feature scores we expect it to be.
                                        past_batch_decay_factor = decay_factor,
                                        RANDOM_INITIAL_FEATURES=False,
                                        encoder_change_mode = encoder_change_mode,
                                        change_sim_c_at_cycle = change_sim_c_at_cycle,
                                        random_seed=random_seed
                    )
                    print("********************************************")
                    
    
    # we don't smooth the matrices
    for sparsity_val in sparsity_array:
        for smoothness_val in smoothness_array:
            for num_lag in num_lags_array:
                for decay_factor in  decay_factor_array:

                    # no one can escape the beauty of python one-liner, granted at the expense of line width
                    sparsity_val, smoothness_val = np.round(sparsity_val, ROUND_DECIMALS), np.round(smoothness_val, ROUND_DECIMALS)
                    
                    print("********************************************")
                    print(f'running experiment convex feature selection')
                    run_convex_selection(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                                        data_dump_folder = data_dump_folder,
                                        norm_val= [mean_first_peak, std_of_peaks],
                                        norm_var_2= [mean_second_peak, std_of_peaks],
                                        train_high_SNR_time
                                         = 10, #  60 batches or  1200 times)
                                        FEATURE_SELETOR_TYPE='joint_convex',
                                        number_of_features = number_of_features,
                                        threshold_selection = 0.5,
                                        objective_offset = 1,  
                                        sparsity_coef = sparsity_val,
                                        smoothness_coef = smoothness_val,
                                        num_of_lags = num_lag,  #  this is the K in the formulation, the number of batch updated feature scores we expect it to be.
                                        past_batch_decay_factor = decay_factor,
                                        RANDOM_INITIAL_FEATURES=False,
                                        encoder_change_mode = encoder_change_mode,
                                        change_sim_c_at_cycle = change_sim_c_at_cycle,
                                        random_seed=random_seed,
                                        smooth_the_matrices = False,
                    )
                    print("********************************************")
    
    
    # then you can run the this file using CLI python tools
    
    
    
    
    