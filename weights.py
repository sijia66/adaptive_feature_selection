"""
this file works with decoder weights
the inputs to this functins are primarily np.arrays 
"""
import numpy as np
from riglib.bmi.kfdecoder import KalmanFilter

# we define a function that does this. 
def replace_kalman_filter(target_kf, C:np.matrix, Q:np.matrix = np.nan, debug:bool = True):
    """
    this function replaces the observation matrix of target_kf to C and
    replace the noise model 
    """
    
    if not isinstance(target_kf, KalmanFilter):
        raise Exception(f'{target_kf} is not an instance of riglib.bmi.kfdecoder.KalmanFilter' )
    
    C_before = target_kf.C
    C_xpose_Q_inv_before = target_kf.C_xpose_Q_inv
    C_xpose_Q_inv_C_before = target_kf.C_xpose_Q_inv_C
    
    if Q is np.nan: Q = target_kf.Q
    
    #calculate the new terms
    C_xpose_Q_inv = C.T * np.linalg.pinv(Q)
    C_xpose_Q_inv_C = C.T * np.linalg.pinv(Q) * C
    
    #reassign
    target_kf.C = C
    target_kf.C_xpose_Q_inv = C_xpose_Q_inv
    target_kf.C_xpose_Q_inv_C = C_xpose_Q_inv_C
    
    if debug:
        print('C matrix before')
        print(C_before)
        
        print('C matrix after')
        print(target_kf.C )
        
        print('C_xpose_Q_inv before:')
        print(C_xpose_Q_inv_before)
        
        print('C_xpose_Q_inv_C after:')
        print(target_kf.C_xpose_Q_inv)
              
        print('C_xpose_Q_inv_C_before:')
        print(C_xpose_Q_inv_C_before)
        
        print('C_xpose_Q_inv_C after:')
        print(target_kf.C_xpose_Q_inv_C)
              


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