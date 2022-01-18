import aopy
import tables
import os

def load_feature_selection_files(data_dump_folder, exp_conds):
    """
    load feature selection from data dump folder
    
    Args;
    
    Returns:
        exp_data_all: a list of exp_data as parsed by bmi3d
        exp_data_metadata_all: a list of exp_metadata as parsed by bmi3d
    """

    exp_data_all = list()
    exp_data_metadata_all = list()

    for e in exp_conds:
        files = {'hdf':e+'.h5'}
        file_name = files['hdf']
        
        try:
            d,m = aopy.preproc.parse_bmi3d(data_dump_folder, files)

            #also load the clda and feature selection files

            feature_selection_data = aopy.data.load_hdf_group(data_dump_folder, file_name,'feature_selection')
            d['feature_selection'] =  feature_selection_data
            
            exp_data_all.append(d)
            exp_data_metadata_all.append(m)
            
        except:
            print(f'cannot parse {e}')

        
    return (exp_data_all, exp_data_metadata_all)