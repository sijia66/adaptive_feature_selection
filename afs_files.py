import aopy
import tables
import os


import pickle

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
            
            try :
                lasso_data = aopy.data.load_hdf_group(data_dump_folder, 
                                                      file_name,
                                                      'lasso_feature_selection')
                d['lasso_feature_selection'] = lasso_data
            except:
                pass
            
            exp_data_all.append(d)
            exp_data_metadata_all.append(m)
            
        except:
            print(f'cannot parse {e}')

        
    return (exp_data_all, exp_data_metadata_all)



def load_clda_pickle_file(data_dump_folder, exp_conds):

    if type(exp_conds) is not list:
        exp_conds = [exp_conds]

    clda_data_full = []

    for i in exp_conds:
        pickle_file_name = i + ".p"
        with open(data_dump_folder + pickle_file_name, 'rb') as f:
            clda_data =  pickle.load(f)
        clda_data_full.append(clda_data)

    return clda_data_full

def convert_to_list(clda_data):
    """
    convert a list of dictionaries of the keys to a dictionary of lists. 
    clda_data (list)

    Returns:
        clda_dict
    """

    clda_keys = clda_data[0].keys()
    clda_dict = dict()
    for k in clda_keys:
        clda_dict[k] = [c[k] for c in clda_data]

    return clda_dict


def load_and_convert_clda_pickle_files(data_dump_folder, exp_conds):

    clda_data_full = load_clda_pickle_file(data_dump_folder, exp_conds)
    
    return [convert_to_list(clda_data) for clda_data in clda_data_full]