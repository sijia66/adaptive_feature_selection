import matplotlib.pyplot as plt
import numpy as np
import aopy
import string

#define constants
GLOBAL_FIGURE_VERTICAL_SIZE = 4


#useful ultility function copiee and  
X_VEL_STATE_IND = 3
Y_VEL_STATE_IND = 5


def subplots_with_labels(n_rows, n_cols, 
                         custom_labels = None, #TODO, instead of by capital letters, we can use custom labels
                         label_directions = 'row_first', # 'row_first' or 'col_first'
                         return_labeled_axes = False,
                         rel_label_x = -0.25, rel_label_y = 1.1, label_font_size = 11, 
                         constrained_layout = True, **kwargs):


    # if more than 26 subplots, raise an error
    if n_rows*n_cols > 26:
        raise ValueError("More than 26 subplots requested, running out of single letters to label them with!")

    # make a list of letters to use as labels

    alphabets = string.ascii_uppercase
    labels = alphabets[:n_rows*n_cols]

    # tabulate the labels into n_rows by n_cols array
    labels = np.array(list(labels)).reshape((n_rows,n_cols), 
                                            order = 'F' if label_directions == 'col_first' else 'C')


    # make the figure and axes
    fig, labels_axes = plt.subplot_mosaic(labels,
                                        constrained_layout=constrained_layout,
                                        **kwargs)

    

    for n, (key, ax) in enumerate(labels_axes.items()):
        ax.text(rel_label_x, rel_label_y, 
                key, transform=ax.transAxes, 
                size=label_font_size)
        
        # just annotate the axes
    axes = list(labels_axes.values())
    axes = np.array(axes).reshape((n_rows,n_cols))

    if return_labeled_axes:
        return fig, axes, labels_axes
    else:
        return fig, axes
    
def concatenate_encoder_weights_feature_sets(encoder_weight, exp_data):
    num_batches = encoder_weight.shape[0]

    num_blank_columns = 5
    blank_columns = np.ones((num_blank_columns, encoder_weight.shape[0]))

    feature_sets_one_cond = exp_data[0]['feature_selection']['feat_set']
    feature_sets_the_other_cond = exp_data[-1]['feature_selection']['feat_set']
    # make these numpy arrays
    feature_sets_one_cond = np.array(feature_sets_one_cond)
    feature_sets_the_other_cond = np.array(feature_sets_the_other_cond)
    # concatenate the feature sets along the second axis
    feature_sets = np.concatenate((feature_sets_one_cond, blank_columns, feature_sets_the_other_cond), axis=0)

    # add the encoder weights to the feature sets
    encoder_weights = encoder_weight.T
    # normalize to 0 and 1
    encoder_weights = (encoder_weights - encoder_weights.min()) / (encoder_weights.max() - encoder_weights.min())

    encoder_weights_feature_sets = np.concatenate((encoder_weights, blank_columns, feature_sets), axis=0)

    # make columns are neurons and rows are selections
    encoder_weights_feature_sets = encoder_weights_feature_sets.T


    return encoder_weights_feature_sets

def plot_prefered_directions(C:np.ndarray, plot_states:tuple = (X_VEL_STATE_IND,Y_VEL_STATE_IND),
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
        ax.axis('equal')

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


def plot_feature_selection(active_feat_set_list, ax = None, selected_color = 'white', unselected_color = 'black',
                                                 label_x = True, default_x_label = 'Learner batch',
                                                 label_y = True, default_y_label = 'Feature index'):
    """
    plot the selection strategy map. 
    """
    from matplotlib import colors
    
    
        
    active_feat_heat_map = np.array(active_feat_set_list, dtype = np.int32)
    active_feat_heat_map = np.ma.masked_where(active_feat_heat_map == False, active_feat_heat_map)
    

    
    if ax is None: 
        fig, ax = plt.subplots()
        print(type(ax))
        
        #color true to yellow
    cmap = colors.ListedColormap([selected_color])
    ax.imshow(active_feat_heat_map.T, cmap = cmap, aspect = 'auto', interpolation = 'none')

   
    cmap.set_bad(color=unselected_color)

    if label_x: ax.set_xlabel(default_x_label)
    if label_y: ax.set_ylabel(default_y_label)

    return ax

CENTER_TARGET_ON = 16
CURSOR_ENTER_CENTER_TARGET = 80
CENTER_TARGET_OFF = 32
REWARD = 48
DELAY_PENALTY = 66
TIMEOUT_PENALTY = 65
HOLD_PENALTY = 64
TRIAL_END = 239

import functools

def get_all_cursor_trajectories(exp_data_all, start_code = [20], end_codes = [REWARD, HOLD_PENALTY]):
    
    cursor_trajectories_list = list()
        
    for e in exp_data_all:
        (cursor_trajectories, trial_segments, trial_times) = get_cursor_trajectories_from_parsed_data(e, start_code = start_code, end_codes=end_codes)
        cursor_trajectories_list.append(cursor_trajectories)

    
    return cursor_trajectories_list

def get_cursor_trajectories_from_parsed_data(exp_data, start_code = [20], end_codes = [REWARD, HOLD_PENALTY]):
    
    
    events = exp_data['events']
    cursor_kinematics = exp_data['task']['cursor'][:,[0,2]] # cursor (x, z, y) position on each bmi3d cycle

    streamed_code = events['code']
    event_cycles = events['time'] # confusingly, 'time' here refers to cycle count

    trial_segments, trial_times = aopy.preproc.get_trial_segments(streamed_code, event_cycles, start_code,  end_codes)
    trial_segments = np.array(trial_segments)
    trial_indices = [range(t[0], t[1]) for t in trial_times]
    cursor_trajectories = [cursor_kinematics[t] for t in trial_indices]
    
    return (cursor_trajectories, trial_segments, trial_times)



from typing import List, Dict
import seaborn as sns


def plot_cursor_trajectories(cursor_trajectories: List, exp_data:Dict, exp_metadata,ax = None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    trials = exp_data['bmi3d_trials']
    trial_targets = aopy.postproc.get_trial_targets(trials['trial'], trials['target'][:,[0,2]]) # (x, z, y) -> (x, y)
    unique_targets = np.unique(np.concatenate(([t[1] for t in trial_targets], trial_targets[0])), axis=0)


    target_radius =  exp_metadata['target_radius']
    bounds = [-11, 11, -11, 11]
    
    sns.color_palette("dark:salmon_r", as_cmap=True)

    aopy.visualization.plot_trajectories(cursor_trajectories, bounds = bounds, ax = ax)
    aopy.visualization.plot_targets(unique_targets, target_radius, ax = ax)


def calculate_encoder_weight_change(sim_c, new_sim_c=None,
                                     include_new_sim_c=False, nnum_of_repeats_before=30, num_of_repeats_after=30):
    good_features_initial = (np.linalg.norm(sim_c, axis=1))
    if include_new_sim_c:
        good_features_after_shuffled = (np.linalg.norm(new_sim_c, axis=1))
    else:
        good_features_after_shuffled = good_features_initial

    old_features_before_shuffled_repeat = np.repeat(good_features_initial[:, np.newaxis],
                                                    nnum_of_repeats_before, axis = 1)
    new_features_after_shuffled_repeat = np.repeat(good_features_after_shuffled[:, np.newaxis],
                                                    num_of_repeats_after, axis = 1)

    encoder_weight_change = np.concatenate((old_features_before_shuffled_repeat,
                                            new_features_after_shuffled_repeat), axis = 1)
    return encoder_weight_change