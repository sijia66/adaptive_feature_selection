import numpy as np

# shows the kind of experiments we can generate
exp_types = ['gap_difference', 'std_difference', 'single_target_reach']
exp_type = 'std_difference'

exp_types_to_run = ['noise_scan']

N_NEURONS = 128

######################################################################
# generic task information
from simulation_runs import run_iter_feat_addition
total_exp_time = 1200 # in seconds
#####################################################################
# noise scan




######################################################################
# actually running the experiments
# data saving stuff
data_dump_folder = \
'/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/gaussian_peaks/2022_01_21_gaussian_peaks/'

# gap difference
# exp_type = 'gap_difference'
mean_first_peak = 50
std_of_peaks = 10
mean_differences = np.arange(0, 60,  step = 10 )

if 'gap_difference' in exp_types_to_run:
    for mean_diff in mean_differences:
        mean_second_peak = mean_first_peak + mean_diff

        print("********************************************")
        print("********************************************")
        print("********************************************")

        print(f'running experiment with second peak at {mean_second_peak}')
        run_iter_feat_addition(total_exp_time = total_exp_time, n_neurons= N_NEURONS,
                            data_dump_folder = data_dump_folder,
                            norm_val= [mean_first_peak, std_of_peaks],
                            norm_var_2= [mean_second_peak, std_of_peaks],
                            train_high_SNR_time = 60 #  60 batches or  1200 times)
        )

        print("********************************************")
        print("********************************************")
        print("********************************************")

###################################################################################3
data_dump_folder = \
'/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/gaussian_peaks/2022_01_21_gaussian_stds/'

mean_first_peak = 50
mean_second_peak = 100
std_differences = np.arange(10, 60, step =  10)

if 'std_difference' in exp_types_to_run:
    for std_val in std_differences:
        std_second_peak = std_val

        print("********************************************")
        print("********************************************")
        print("********************************************")

        print(f'running experiment with second peak at {mean_second_peak}')
        run_iter_feat_addition(total_exp_time = total_exp_time, n_neurons=N_NEURONS,
                            data_dump_folder = data_dump_folder,
                            norm_val= [mean_first_peak, std_of_peaks],
                            norm_var_2= [mean_second_peak, std_second_peak],
                            train_high_SNR_time = 60 #  60 batches or  1200 times)
        )

        print("********************************************")
        print("********************************************")
        print("********************************************")
