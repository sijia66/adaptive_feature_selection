import matplotlib.pyplot as plt
import numpy as np


#useful ultility function copiee and  
X_VEL_STATE_IND = 3
Y_VEL_STATE_IND = 5
def plot_prefered_directions(C:np.array, plot_states:tuple = (X_VEL_STATE_IND,Y_VEL_STATE_IND),
            ax=None, 
            invert=False, **kwargs):
    '''
    Plot 2D "preferred directions" of features in the Decoder

    Parameters
    ----------
    C: np.array of shape (n_features, n_states)
    ax: matplotlib.pyplot axis, default=None
        axis to plot on. If None specified, a new one is created. 
    plot_indices: a tuple of states in  in the matrix

    invert: bool, default=False
        If true, flip the signs of the arrows plotted
    kwargs: dict
        Keyword arguments for the low-level matplotlib function
    '''

    if ax == None:
        plt.figure()
        ax = plt.subplot(111)

    if C.shape[1] > 2:
        x, z = plot_states
    else:
        x, z = 0, 1

    n_neurons = C.shape[0]
    linestyles = ['-.', '-', '--', ':']

    if invert:
        C = C*-1

    for k in range(n_neurons):
        ax.plot([0, C[k, x]], [0, C[k, z]], 
                linestyle=linestyles[int(k/7) % len(linestyles)], 
                **kwargs)