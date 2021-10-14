

import numpy as np



def sort_trials(wait_time:list, 
                target_seq:list,
                task_data_hist_np:dict, 
                dict_keys, FRAME_RATE = 60):
    """
    specific to runnng Si Jia's simulation
    accepts a task_data_hist_np with saved data
    and returns a list of dictionaries of targets
    """
    trial_dict = list()
    
    for i,row in enumerate(wait_time):
        start_time = row[0]
        inter_wait_time = row[1]

        start_sample = int(start_time * FRAME_RATE)
        inter_wait_sample = int(inter_wait_time * FRAME_RATE)
        stop_sample = start_sample + inter_wait_sample

        single_trial_dict = dict()

        for k in dict_keys:
            
            requested_type_data = np.squeeze(task_data_hist_np[k])
            single_trial_dict[k] =  requested_type_data[start_sample:stop_sample,
                                                       :]
        #add more info
        single_trial_dict['start_time'] = row[0]
        single_trial_dict['trial_time'] = row[1]
        
        #add target info
        single_trial_dict['targets'] = target_seq[i]

        #add the dictionary to the list
        trial_dict.append(single_trial_dict)
        
    return trial_dict

def segment_trials_in_state_log(state_log, 
                      trial_end_states = ['reward', 'timeout_penalty', 'hold_penalty']):

    segmented_trials = list()

    single_trial_states = list()

    for s in state_log:
        state_name = s[0]
        
        if state_name in trial_end_states:
            single_trial_states.append(s)
            segmented_trials.append(single_trial_states)    
            single_trial_states = list()
            
        else:
            single_trial_states.append(s)

    return segmented_trials

def sort_trials_use_segmented_log(segmented_trials, target_seq:list, task_data_hist_np:dict, 
                dict_keys, FRAME_RATE = 60):
    
    trial_dict = list()

    target_ind = -1 #keeping track which target for each trial
    
    for i,st in enumerate(segmented_trials):
        first_event_name, first_event_time = st[0]
        last_event_name, last_event_time= st[-1]
        trial_time = last_event_time - first_event_time

        start_sample = int(first_event_time * FRAME_RATE)
        inter_wait_sample = int(trial_time * FRAME_RATE)
        stop_sample = start_sample + inter_wait_sample

        single_trial_dict = dict()

        for k in dict_keys:
            
            requested_type_data = np.squeeze(task_data_hist_np[k])
            single_trial_dict[k] =  requested_type_data[start_sample:stop_sample,
                                                       :]
        #add event info
        single_trial_dict['event_log'] = st
        
        #add target info
        #only when there is a wait log
        if first_event_name == 'wait': target_ind += 1
        single_trial_dict['targets'] = target_seq[target_ind]

        #add the dictionary to the list
        trial_dict.append(single_trial_dict)
        
    return trial_dict


def filter_state(state_log:list, state_to_match:str)->list:
    '''
    given a list of states 
    state_log: a list of tuples (state:string, start_time: float)
    state_to_watch
    
    returns a list of element type
    '''
    
    return list(filter(lambda k: k[0] == state_to_match, state_log) )

def calc_inter_trial_times(trial_log: list)-> list:
    """
    state_log: a list of tuples ("wait", start_time: float)
    return a list of tuples: ("wait", start_time: float, diff_time)
    """
    wait_log_with_diff = list()
    for i, wait_state in enumerate(trial_log):
        if i == len(wait_log)-1: #there is nothing to subtract, just put zero.
            wait_log_with_diff.append((wait_state[1],  0))
            
        else:
            finish_time = wait_log[i+1][1]
            wait_log_with_diff.append((wait_state[1],  finish_time - wait_state[1]))
    
    return np.array(wait_log_with_diff[:-1])


def calc_arc_length_from_trial_dict(trial_dict):
     arc_length = [calc_arc_length(sample_trial['cursor']) for sample_trial in trial_dict]
     return np.array(arc_length)

def calc_arc_length_from_cursor(cursor_data, window = 10, frame_rate = 60):

    cursor_data = np.squeeze(cursor_data)

    num_time, num_coor = cursor_data.shape

    block_size = window * frame_rate
    num_block = int(num_time / block_size)

    arc_length_in_blocks =  np.empty(num_block)

    for i in range(num_block):

        time_ind = range(i * block_size, (i+1) * block_size)
        arc_length_in_blocks[i] = calc_arc_length(cursor_data[time_ind, :])

    return arc_length_in_blocks


def calc_arc_length(trial_cursor_trajectory):
    """
    trial_cursor_trajectory[numpy.array]: of n_time by n_coordinates
    returns arc length
    """
    
    trial_cursor_diff = np.diff(trial_cursor_trajectory, axis = 0)
    trial_cursor_arc = np.linalg.norm(trial_cursor_diff, axis = 1)
    trial_arc_length = np.sum(trial_cursor_arc)
    
    return trial_arc_length


def calc_event_rate_from_state_log(state_log, event_name, window_length_in_frames = 60,FRAME_RATE = 60, **kwargs):
    '''
    a wrapper function to calc event rate
    '''
    window_length = window_length_in_frames / FRAME_RATE

    event_record = generate_event_array(state_log, event_name, FRAME_RATE = FRAME_RATE, **kwargs)

    return calc_event_rate(event_record, window_length = window_length, FS = FRAME_RATE)

def calc_moving_event_rate(state_log, event_name, window_length, FRAME_RATE = 60, **kwargs):
    event_record = generate_event_array(state_log, event_name, FRAME_RATE = FRAME_RATE, **kwargs)

    return moving_sum(event_record, FRAME_RATE * window_length)


def generate_event_array(state_log, event_name, total_time = None, FRAME_RATE = 60, **kwargs):
    '''
    given a state log, generate an array in which the corresponding time point has a value of one.
    args:
        state_log(list): a list of tuples of (state_name:str, time:float)
    returns:
        event_log(np.array):
    '''
    if total_time is None:
            #get the last experiment time
        exp_time = state_log[-1][1]
    else:
        exp_time = total_time
    
    total_num_frames = exp_time

    
    event_record = np.zeros(total_num_frames)
    #proc state log
    #each state has a ('state', time in float)
    event_trial_time = [s[1] for s in state_log if s[0] == event_name]
    event_trial_frame = np.array(event_trial_time)
    event_trial_frame = np.array(event_trial_frame, dtype = np.int)
    
    event_record[event_trial_frame] = 1
    
    return event_record

def calc_event_rate(event_record, window_length = 60, FS = 60):
    """
    reward_record; a state log containing succuss trials. 
    sorted_trials_in_a_list
    window_length[s]: default to 1 min
    """

    num_frames = len(event_record)
    #get number of frames
    window_length_in_frames = int(window_length *  FS)

    #then we can reshape
    quotient  = num_frames % window_length_in_frames 
    completed_window_length = int(num_frames - quotient)
    event_vec = event_record[:completed_window_length]

    #folded into the windows  and count how many trials
    event_vec_folded = event_vec.reshape((-1, window_length_in_frames))
    event_rate = np.sum(event_vec_folded, axis = 1) 



    return event_rate 


def moving_sum(x, w):
    return np.convolve(x, np.ones(w), 'valid')

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w