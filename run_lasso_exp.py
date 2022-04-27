import numpy as np
from afs_clda_1000_neurons import run_iter_feat_addition, run_lasso_sims
from simulation_runs import run_convex_selection


exp_types = ['lasso', 'convex']
exp_types_to_run = ['convex']

total_exp_time = 240 # in seconds
N_NEURONS = 128




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

        # noise scan
    data_dump_folder = \
    '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/convex_selection/'
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
                        train_high_SNR_time = 10, #  60 batches or  1200 times)

    )

    print("********************************************")
    print("********************************************")
    print("********************************************")