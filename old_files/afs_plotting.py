import matplotlib.pyplot as plt
import numpy as np

#define constants
GLOBAL_FIGURE_VERTICAL_SIZE = 4


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

def add_center_out_grid(ax, target_seq, radius,circle_alpha = 0.2,range_lim = 15):

    target_origins =  np.unique(target_seq, axis = 0)

    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    
    add_circular_origin(ax, radius)
    add_targets(ax, target_origins, radius, circle_alpha = circle_alpha)

def plot_trial_trajectory(ax, cursor_trajectory):
    X_CURSOR_INDEX = 0
    Z_CURSOR_INDEX = 2
    ax.plot(cursor_trajectory[:, X_CURSOR_INDEX], 
             cursor_trajectory[:, Z_CURSOR_INDEX])
    

def add_circular_origin(ax, radius):
    add_circle_to_ax(ax, (0,0), radius)

def add_targets(ax, targets_origins, radius, circle_alpha = 0.2):

    X_CURSOR = 0
    Z_CURSOR = 2
    #plot the targets
    for origin_t in targets_origins:
        origin = origin_t[0]
        t = origin_t[1]

        cc = plt.Circle((t[X_CURSOR],t[Z_CURSOR] ), 
                        radius = radius,
                        alpha = circle_alpha)

        ax.set_aspect( 1 ) 
        ax.add_artist( cc ) 


def add_circle_to_ax(ax, origin, radius, circle_alpha = 0.2):
    '''
    plot a circle with origin (np.array)

    args:
        origin
        radius
    '''
    
    cc = plt.Circle(origin, 
                radius = radius,
                alpha = circle_alpha)

    ax.add_artist( cc ) 

import matplotlib as mpl

def get_cmap(n_lines, color = None):

    if color is None:
        color = mpl.cm.Blues
    c = np.arange(1, n_lines + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=color)
    cmap.set_array([])

    return cmap