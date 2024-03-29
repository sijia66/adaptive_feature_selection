import numpy as np
from afs_clda_1000_neurons import run_iter_feat_addition


data_dump_folder = \
'/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/theory_dev/'


#random_seeds = range(10)
#num_neurons = [32, 128, 512, 1024]

random_seeds = [0]
num_neurons = [512]

#percent_high_SNR_noises = np.arange(0, 1.2, 0.2)
percent_high_SNR_noises = [0.7]

for nn in num_neurons:  
    for noise in percent_high_SNR_noises:

        for s in random_seeds:

                run_iter_feat_addition(total_exp_time = 2400, n_neurons = nn,
                                    random_seed=s, 
                                    percent_high_SNR_noises = [noise],
                                    data_dump_folder=data_dump_folder)

    print(f'finished exp with random seed {nn} with random seeds {s}')



