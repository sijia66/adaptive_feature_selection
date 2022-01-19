from simulation_runs import run_iter_feat_addition


#####################################################################
# data saving stuff
data_dump_folder = \
'/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/theory_dev/'

######################################################################
# generic task information
total_exp_time = 1200 # in seconds
run_iter_feat_addition(total_exp_time = total_exp_time, 
                       data_dump_folder = data_dump_folder)