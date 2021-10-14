"""
this python stores tiny tweeny functions that
help with processing hdf files

"""

import tables 
import numpy as np


def open_hdf_file(file_name:str = None, debug:bool = True) -> tables.Table:
    """
    a wrapper around tables.open_file
    with basic IO checking stuff
    that takes file_name e.g. temp_data/one_trial_clda.h5
    debug just print outs stuff,

    returns:
        hdffile a tables
    """

    try:
        hdffile = tables.open_file(file_name, 'r')

        if debug: print(hdffile)
        return hdffile

    except IOError:
        print(f{{__name__}:"{file_name} not accessible"})

    
def open_many_hdf_files(file_names:tuple ,debug:bool = True) -> tuple:
    """
    take a tuple of file names 
    and open_hdf_file on each of it 
    return a tuple of hdffile handles
    """

    hdf_files = [open_hdf_file(h) for h in file_names]
    return tuple(hdf_files)


def get_kf_C_from_hdf(file_name:str, debug:bool = True) ->  np.array:
    """
    this function gets the observation matrix C from hdffil.clda table
    assume the table exits
    """
    if debug:  print(f'{__name__}::assumes to take kf_c from {file_name}')

    hdffile = open_hdf_file(file_name = file_name)
    
    if hdffile is None: raise Exception(f'{file_name} returns none')

    clda_tables = hdffile.root.clda
    if debug: print(f'{__name__}: {clda_tables.description}')

    return clda_tables.col('kf_C')



