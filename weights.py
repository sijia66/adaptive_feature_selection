"""
this file works with decoder weights
the inputs to this functins are primarily np.arrays 
"""
import numpy as np


def _cal_tuning_angle_2D(mat_kf_c: np.array, vel_ind: tuple = (3,  5)) -> np.array:
    '''
    mat_kf_c: N_Neuron by N_states
    assume with each row with [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, constant]
    '''
    
    N_Neuron, N_States = mat_kf_c.shape
                         
    COL_X, COL_Z = vel_ind
    
    tuning_angles = np.zeros(N_Neuron)
    
    tuning_angles = [np.rad2deg(np.arctan(neuron_row[COL_Z] / neuron_row[COL_X])) for neuron_row in mat_kf_c]
    
    return np.array(tuning_angles)

def cal_tuning_angle(mat_kf_c: np.array, vel_ind: tuple = (3,  5)) -> np.array:
    if len(mat_kf_c.shape) == 2:
        return np.array(_cal_tuning_angle_2D(mat_kf_c, vel_ind = vel_ind))
    elif len(mat_kf_c.shape) ==3:
        tuning_angles = [_cal_tuning_angle_2D(mat_kf_c_slice, vel_ind = vel_ind) for mat_kf_c_slice in mat_kf_c]
        return np.array(tuning_angles)
    else:
        print(f'{__name__}: does not know how to deal with mat C with the shape of {mat_kf_c.shape}')