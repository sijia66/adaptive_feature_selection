import numpy
from afs_clda_1000_neurons import run_iter_feat_addition


data_dump_folder = \
'/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/test/diff_neurons/'


random_seeds = range(10)
num_neurons = [32, 128, 512, 1024]

for nn in num_neurons:

    for s in random_seeds:

        run_iter_feat_addition(total_exp_time = 2400, n_neurons = nn,
                            random_seed=s, 
                            data_dump_folder=data_dump_folder)

    print(f'finished exp with random seed {nn} with random seeds {s}')

