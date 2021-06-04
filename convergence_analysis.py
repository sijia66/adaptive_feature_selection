import numpy as np 

class ConvergenceAnalyzer():
    def __init__(self, optimal_thetas = None):
        '''
        setup the analyzer
        INPUTS:
            optimal_values: 1D numpy array
        '''
        #make sure the optimal values has a dim of 1
        if optimal_thetas is not None:
            assert optimal_thetas.ndim == 1

        self.optimal_thetas = optimal_thetas
        self.n_var = self.optimal_thetas.size

    @classmethod
    def calc_mse_along_rows(cls, thetas, optimal_thetas):
        '''
        assume thetas is 3D
        '''
        cls.check_remaining_axes_match(thetas, optimal_thetas)

        n_time = thetas.shape[0]

        mses = np.empty(n_time)

        for i in range(n_time):
            theta_slice = thetas[i,:,:]
            mses[i] = cls.calc_mse_wz_theta_and_optimal(theta_slice, optimal_thetas)

        return mses


    @classmethod
    def calc_mse_wz_theta_and_optimal(cls, thetas, optimal_thetas):
        '''
        
        '''

        return np.linalg.norm(thetas - optimal_thetas) / np.linalg.norm(optimal_thetas)

    @classmethod
    def check_remaining_axes_match(cls,A,B):
        assert A.shape[1:] == B.shape


def calc_cosine_to_target_matrix(matrix_series, target_mat = None, **kwargs):
    '''
    compares matrix_series (num_time by num_features by num_vectors)
    '''

    num_time, num_feats, num_states = matrix_series.shape

    if target_mat is None: target_mat = matrix_series[0,:,:]

    angle_hist = list()

    for i in range(num_time):
        mat_A = matrix_series[i,:,:]
        mat_B = target_mat
        angles = calc_cosine_sim_bet_two_matrices(mat_A, mat_B)

        angle_hist.append(angles)

    return np.array(angle_hist)


def calc_cosine_sim_bet_two_matrices(mat_A, mat_B, selected_cols = (3,5), deg = True):
    
    assert mat_A.shape == mat_B.shape

    num_rows, num_cols = mat_A.shape
    
    angles = np.empty(num_rows)

    for i in range(num_rows):
        angles[i] = calc_cosine_similarity(mat_A[i, selected_cols],
                                       mat_B[i, selected_cols],
                                       deg = deg)

    return angles

from scipy.spatial import distance

def calc_cosine_similarity(x, y, deg = False):
    cos_dist = 1 -  distance.cosine(x, y)
    if deg: 
        return np.rad2deg(np.arccos(cos_dist))
    else:
        return cos_dist


from scipy.optimize import curve_fit

class ExpFitAnalyzer():
    def __init__(self):
        self.fitted_flag = False

    def calc_fitting_params(self, f, x_data, y_data):
        fitting_params, fitting_cov = curve_fit(f = f, xdata = x_data, ydata = y_data)

        self.fitted_flag = True

        #record the thing for other analyses
        self.f = f
        self.fitting_params = fitting_params
        self.x_data = np.copy(x_data)
        self.y_data = y_data
        
                                
        return (self.fitting_params,fitting_cov)

    def  calc_estimated_y(self):
        if not self.fitted_flag: raise Exception('not fitted yet!')

        self.est_y = self.f(self.x_data, *self.fitting_params)
        return self.est_y

    def plot_fitting(self, ax, label_flag = True):
        ax.plot(self.x_data, self.y_data)
        ax.plot(self.x_data, self.est_y)
        

def calc_flipped_shifted_exponential(x, a,b):
    return a*(1-np.exp(-b*x))

def calc_flipped_shifted_exp_from_equilibrium(x, a,b, c):
    return a*(1-np.exp(-b*x)) + c

    