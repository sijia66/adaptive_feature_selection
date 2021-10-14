import os

import aopy

data_dir = os.getcwd()

result_dir = '/home/sijia-aw/BMi3D_my/operation_funny_chicken/sim_data/'



files = {

    'hdf':'wo_feature selection'

}



result_filename = 'preprocessed_' + files['hdf']

aopy.preproc.proc_exp(data_dir, files, result_dir, result_filename)