import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import datetime
import os
from typing import Dict, List, Tuple, Optional, Any
from detrend_sliding_window import detrend_sliding_window
from flattenDoppler2D import flattenDoppler2D
from crossvalidate_short import crossvalidate_short
from preprocess import preProcess
import h5py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Neural decoding analysis script - Python conversion of MATLAB code
# TC: for early direction/value tests with Verasonics, Sessions [164 165 166] - 4 corners 2 values
# simplified script - for Bahareh 20240606

# load a dop_trials_struct before proceeding
def load_matlab_v73(filepath):
    """
    Load MATLAB v7.3 files using h5py
    """
    def h5py_to_dict(h5_file, path='/'):
        """Recursively convert h5py file to dictionary"""
        data = {}
        for key in h5_file[path].keys():
            if isinstance(h5_file[path + key], h5py.Dataset):
                # Handle different data types
                dataset = h5_file[path + key]
                if dataset.dtype.kind == 'O':  # Object references
                    # Handle cell arrays or struct arrays
                    try:
                        # Try to dereference object references
                        refs = dataset[:]
                        if refs.ndim == 2 and refs.shape[0] == 1:
                            refs = refs[0]
                        
                        deref_data = []
                        for ref in refs.flat:
                            if ref:
                                deref_data.append(h5_file[ref][:])
                        
                        if len(deref_data) == 1:
                            data[key] = deref_data[0]
                        else:
                            data[key] = deref_data
                    except:
                        # If dereferencing fails, just get the raw data
                        data[key] = dataset[:]
                else:
                    # Regular numeric data
                    arr = dataset[:]
                    # MATLAB stores arrays in column-major order, transpose if needed
                    if arr.ndim > 1:
                        data[key] = arr.T
                    else:
                        data[key] = arr
            elif isinstance(h5_file[path + key], h5py.Group):
                # Recursively handle groups (structs)
                data[key] = h5py_to_dict(h5_file, path + key + '/')
        return data
    
    with h5py.File(filepath, 'r') as f:
        return h5py_to_dict(f)

def debug_data_structure(data_dict, max_depth=3, current_depth=0):
    """Debug function to examine the structure of loaded data"""
    if current_depth > max_depth:
        return
    
    for key, value in data_dict.items():
        indent = "  " * current_depth
        if isinstance(value, dict):
            print(f"{indent}{key}: dict with keys {list(value.keys())}")
            debug_data_structure(value, max_depth, current_depth + 1)
        elif isinstance(value, np.ndarray):
            print(f"{indent}{key}: array shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, list):
            print(f"{indent}{key}: list of length {len(value)}")
            if len(value) > 0 and isinstance(value[0], np.ndarray):
                print(f"{indent}  First element shape: {value[0].shape}")
        else:
            print(f"{indent}{key}: {type(value)}")

# Neural decoding analysis script - Python conversion of MATLAB code
try:
    # First try with h5py for MATLAB v7.3 files
    dop_trials_struct = load_matlab_v73("D:\Downloads\dop_trials_struct_S164_R1_NoRegister_NoDetrend.mat")
    print("Successfully loaded MATLAB v7.3 file with h5py")
except Exception as e:
    print(f"h5py loading failed: {e}")
    try:
        # Fallback to scipy.io.loadmat for older MATLAB files
        from scipy.io import loadmat
        dop_trials_struct = loadmat("D:\Downloads\dop_trials_struct_S164_R1_NoRegister_NoDetrend.mat")
        print("Successfully loaded with scipy.io.loadmat")
    except Exception as e2:
        print(f"Both loading methods failed. h5py error: {e}, scipy error: {e2}")
        raise

# Debug the data structure
print("=== Data Structure Debug ===")
debug_data_structure(dop_trials_struct)

# choose previous N trials for adaptive training set
N = 175
# choose K for K-fold validation
K = 15
# decoding window in seconds
trailing_window = 1

# control for preprocessing steps
time_gain = False
disk_filter = False
z_score = True
detrend = True

# Preprocessing
if detrend:
    detrend_window_length = 50  # seconds
    iDopP = detrend_sliding_window(dop_trials_struct['dop_trials_struct']['iDopP'], 
                                  detrend_window_length * dop_trials_struct['dop_trials_struct']['acquisition_rate'])
else:
    detrend_window_length = 0
    iDopP = dop_trials_struct['dop_trials_struct']['iDopP']

print(f"iDopP shape before preProcess: {iDopP.shape}")
print(f"iDopP dtype: {iDopP.dtype}")
print(f"Contains NaN: {np.any(np.isnan(iDopP))}")
iDopP = preProcess(iDopP, timeGain=time_gain, diskFilter=disk_filter, zScore=z_score)

classifier_string = 'PCA+LDA'
validation_string = 'kFold'
protocol_name = 'ValDir'
n_iter = 1

decoding_results = {}
decoding_results['session'] = dop_trials_struct['dop_trials_struct']['session']
decoding_results['run'] = dop_trials_struct['dop_trials_struct']['run']
decoding_results['protocol_name'] = protocol_name

# indexing labels
only_use_successful_trials = 1
axes_to_decode = []  # [] for both, 1 for horz, 2 for vert

fixation_pos = dop_trials_struct['dop_trials_struct']['fixation_pos']
target_cue_pos = np.squeeze(dop_trials_struct['dop_trials_struct']['target_pos_eachCue']).T
tprime_cue_pos = np.squeeze(dop_trials_struct['dop_trials_struct']['targetPrime_pos_eachCue']).T

success_trial_idx = np.isin(dop_trials_struct['dop_trials_struct']['reached_minimum_state_idx'].flatten(),
                           dop_trials_struct['dop_trials_struct']['success_idx'].flatten())

decoding_results['only_use_successful_trials'] = only_use_successful_trials
decoding_results['axes_to_decode'] = axes_to_decode

for nn in range(n_iter):
    # for saccade direction ipsi/contra or up/down
    label_horz = np.sign(target_cue_pos[:, 0])
    label_vert = np.sign(target_cue_pos[:, 1])

    # set to min of 1
    label_horz = label_horz - np.min(label_horz) + 1
    label_vert = label_vert - np.min(label_vert) + 1

    if only_use_successful_trials:
        label_horz[~success_trial_idx] = np.nan
        label_vert[~success_trial_idx] = np.nan

    # equalize counts of each label combination
    keepidx = []
    mincount_combo = 10000
    
    if len(axes_to_decode) == 0:
        unique_horz = np.unique(label_horz[~np.isnan(label_horz)])
        unique_vert = np.unique(label_vert[~np.isnan(label_vert)])
    elif axes_to_decode == 1:
        unique_horz = np.unique(label_horz[~np.isnan(label_horz)])
        unique_vert = [1]
    elif axes_to_decode == 2:
        unique_horz = [1]
        unique_vert = np.unique(label_vert[~np.isnan(label_vert)])

# Doubt in only_use_fixation_idx variable :

    # find minimum count of combos
    for u1 in range(len(unique_horz)):
        for u2 in range(len(unique_vert)):
                if len(axes_to_decode) == 0:
                    idx = (label_horz == unique_horz[u1]) & (label_vert == unique_vert[u2])
                elif axes_to_decode == 1:
                    idx = label_horz == unique_horz[u1]
                elif axes_to_decode == 2:
                    idx = label_vert == unique_vert[u2]
                mincount_combo = min(mincount_combo, np.sum(idx))

    # create set with equal representation all combos
    for u1 in range(len(unique_horz)):
        for u2 in range(len(unique_vert)):
            if len(axes_to_decode) == 0:
                idx = (label_horz == unique_horz[u1]) & (label_vert == unique_vert[u2])
            elif axes_to_decode == 1:
                idx = label_horz == unique_horz[u1]
            elif axes_to_decode == 2:
                idx = label_vert == unique_vert[u2]

            idx = np.where(idx)[0]
            keepidx.extend(np.random.choice(idx, mincount_combo, replace=False))

    keepidx = np.sort(keepidx)

    train_labels_horz = label_horz[keepidx]
    train_labels_vert = label_vert[keepidx]

    print(f"train_labels_horz shape: {train_labels_horz.shape}")

    # FIX: Handle the lookup_tables_3D_to_4D structure more robustly
    lookup_table = dop_trials_struct['dop_trials_struct']['lookup_tables_3D_to_4D']
    
    print(f"lookup_tables_3D_to_4D type: {type(lookup_table)}")
    if isinstance(lookup_table, list):
        print(f"lookup_tables_3D_to_4D is a list of length: {len(lookup_table)}")
        if len(lookup_table) > 0:
            lookup_array = lookup_table[0]
            print(f"First element shape: {lookup_array.shape}")
        else:
            raise ValueError("lookup_tables_3D_to_4D list is empty")
    else:
        lookup_array = lookup_table
        print(f"lookup_tables_3D_to_4D shape: {lookup_array.shape}")
    
    # Handle different possible shapes
    if lookup_array.ndim == 1:
        # If 1D, assume it's a single trial's time indices
        trial_n_timestamps = len(lookup_array)
        nTrials = 1
        print(f"Detected 1D lookup table: {trial_n_timestamps} timestamps, 1 trial")
    elif lookup_array.ndim == 2:
        trial_n_timestamps = lookup_array.shape[1]
        nTrials = lookup_array.shape[0]
        print(f"Detected 2D lookup table: {trial_n_timestamps} timestamps, {nTrials} trials")
    else:
        raise ValueError(f"Unexpected lookup table dimensions: {lookup_array.shape}")

    yPix, xPix, nWindows = iDopP.shape

    # create doppler struct that is (yPix, xPix, nWindows, nTrials) aligned to event
    train_data = np.full((yPix, xPix, trial_n_timestamps, nTrials), np.nan)
    print(train_data.shape)
    
    for tt in range(nTrials):
        if lookup_array.ndim == 1:
            # Single trial case - use the entire 1D array
            timeidx = lookup_array
        else:
            # Multiple trials case - each row is a trial
            timeidx = lookup_array[tt, :]
            
        if not np.all(np.isnan(timeidx)):
            # Ensure indices are valid
            valid_indices = timeidx[~np.isnan(timeidx)].astype(int)
            # valid_indices = valid_indices[valid_indices < nWindows]  # Bounds check
            
            # if len(valid_indices) > 0:
            train_data[:, :, :len(valid_indices), tt] = iDopP[:, :, valid_indices]

    train_data = train_data.reshape(yPix * xPix, trial_n_timestamps, nTrials)
    
    # Handle case where nTrials might be 1 but we need multiple trials from keepidx
    if nTrials == 1 and len(keepidx) > 1:
        print("Warning: Only 1 trial in lookup table but multiple trials needed for training")
        print("This might indicate a problem with the data structure")
        # Use the available trial for all keepidx indices (not ideal but prevents crash)
        train_data = np.repeat(train_data, len(keepidx), axis=2)
    else:
        train_data = train_data[:, :, keepidx] if len(keepidx) <= nTrials else train_data

    # decode across trial time
    nWindows = train_data.shape[1]
    print(nWindows)

    # Determine dimensionality of cPCA subspace (if used)
    m = len(np.unique(train_labels_horz)) - 1
    print(m)

    # allocate memory
    result_horz = [None] * nWindows
    confusion_horz = [None] * nWindows
    cp_horz = [None] * nWindows
    class_predicted_horz = [None] * nWindows
    p_horz = np.full(nWindows, np.nan)
    percentCorrect_horz = np.full(nWindows, np.nan)
    t_horz = np.full(nWindows, np.nan)

    result_vert = [None] * nWindows
    confusion_vert = [None] * nWindows
    cp_vert = [None] * nWindows
    class_predicted_vert = [None] * nWindows
    p_vert = np.full(nWindows, np.nan)
    percentCorrect_vert = np.full(nWindows, np.nan)
    t_vert = np.full(nWindows, np.nan)

    downsample_interval = 1  # in seconds
    decoding_time_stepsize = 1  # in seconds

    # set decoding start and stop time
    decoding_start_time = decoding_time_stepsize
    decoding_end_time = nWindows

    acquisition_rate = dop_trials_struct['dop_trials_struct']['acquisition_rate']

    if (downsample_interval * acquisition_rate) % 1 != 0:
        raise ValueError('error: the downsample_interval must correspond to an integer number of frames')
    if (decoding_time_stepsize * acquisition_rate) % 1 != 0:
        raise ValueError('error: the decoding_time_stepsize must correspond to an integer number of frames')
    if (trailing_window * acquisition_rate) % 1 != 0:
        raise ValueError('error: the trailingwindow must correspond to an integer number of frames')
    
    trailing_window = int(trailing_window * acquisition_rate)

    trial_time = dop_trials_struct['dop_trials_struct']['trial_timevec'][0]

    print('Time with for loop')
    variance_to_keep = 95
    acq_rate_scalar = float(acquisition_rate)


    for ii in range(int(decoding_start_time * acquisition_rate), 
                    int(decoding_end_time), 
                    int(decoding_time_stepsize * acquisition_rate)):
        
        test_time_i = list(range(max(0, ii - trailing_window), ii + 1))
        # testTimeI = max(range(1, ii - trailing_window + 1), range(ii + 1))

        # prepare training data for this time window
        train_data_window = []

        if len(test_time_i) < (downsample_interval * acquisition_rate):
            # Not enough data — use mean over available
            downsample_data = np.mean(train_data[:, test_time_i, :], axis=1)
            train_data_window.append(flattenDoppler2D(downsample_data, 1))
        else:
            # Main downsampling loop
            for jj in range(test_time_i[0] + int(downsample_interval * acquisition_rate) - 1,
                        test_time_i[-1] + 1,
                        int(downsample_interval * acquisition_rate)):
                downsample_idx = list(range(jj - int(downsample_interval * acquisition_rate) + 1, jj + 1))
                # downsample_data = np.mean(train_data[:, downsample_idx, :], axis=1)

                # # Input to flattenDoppler2D is (yPix, xPix, time, nTrials) — emulate it:
                # yPix_xPix = int(np.sqrt(train_data.shape[0]))
                # downsample_data_reshaped = downsample_data.reshape((yPix_xPix, yPix_xPix, 1, train_data.shape[2]))

                downsample_data = np.mean(train_data[:, test_time_i, :], axis=1)  # shape: (yPix * xPix, nTrials)

                # Reshape properly using original yPix and xPix:
                downsample_data_reshaped = downsample_data.reshape((yPix, xPix, 1, train_data.shape[2]))

                # Now call flattenDoppler2D
                train_data_window.append(flattenDoppler2D(downsample_data_reshaped, 0))


                # flatten
                # train_data_window.append(flattenDoppler2D(downsample_data_reshaped, 1))

        # Concatenate time windows
        if train_data_window:
            train_data_final = np.concatenate(train_data_window, axis=1) if len(train_data_window) > 1 else train_data_window[0]
        else:
            continue

        # === Horizontal decoding ===
        if len(axes_to_decode) == 0 or axes_to_decode == 1:
            cp_horz[ii], p_horz[ii], class_predicted_horz[ii] = crossvalidate_short(
                train_data_final, train_labels_horz,
                classificationMethod=classifier_string, N=N,
                validationMethod=validation_string, K=K,
                m=m, variance_to_keep=variance_to_keep
            )

            percentCorrect_horz[ii] = cp_horz[ii]['CorrectRate'] * 100
            confusion_horz[ii] = cp_horz[ii]['CountingMatrix']

        # === Vertical decoding ===
        if len(axes_to_decode) == 0 or axes_to_decode == 2:
            cp_vert[ii], p_vert[ii], class_predicted_vert[ii] = crossvalidate_short(
                train_data_final, train_labels_vert,
                classificationMethod=classifier_string, N=N,
                validationMethod=validation_string, K=K,
                m=m, variance_to_keep=variance_to_keep
            )
            percentCorrect_vert[ii] = cp_vert[ii]['CorrectRate'] * 100
            confusion_vert[ii] = cp_vert[ii]['CountingMatrix']

        # Save current result
        result_horz[ii] = {
            'percentCorrect': percentCorrect_horz[ii],
            'confusion': confusion_horz[ii],
            'p': p_horz[ii],
            't': ii / acquisition_rate
        }

        result_vert[ii] = {
            'percentCorrect': percentCorrect_vert[ii],
            'confusion': confusion_vert[ii],
            'p': p_vert[ii],
            't': ii / acquisition_rate
        }

        print(f"Processing timepoint: {ii/acq_rate_scalar:.2f} s")

    # ==== Clean up results ====

    elim_idx_downsampling = [i for i, x in enumerate(result_horz) if x is None]

    result_horz = [x for i, x in enumerate(result_horz) if i not in elim_idx_downsampling]
    result_vert = [x for i, x in enumerate(result_vert) if i not in elim_idx_downsampling]

    percentCorrect_horz = np.array([x['percentCorrect'] for x in result_horz])
    confusion_horz = [x['confusion'] for x in result_horz]
    p_horz = np.array([x['p'] for x in result_horz])
    t_horz = np.array([x['t'] for x in result_horz])

    percentCorrect_vert = np.array([x['percentCorrect'] for x in result_vert])
    confusion_vert = [x['confusion'] for x in result_vert]
    p_vert = np.array([x['p'] for x in result_vert])
    t_vert = np.array([x['t'] for x in result_vert])

    # Store results
    decoding_results[f'result_horz_{nn}'] = result_horz
    decoding_results[f'result_vert_{nn}'] = result_vert
    decoding_results[f'percentCorrect_horz_{nn}'] = percentCorrect_horz
    decoding_results[f'confusion_horz_{nn}'] = confusion_horz
    decoding_results[f'p_horz_{nn}'] = p_horz
    decoding_results[f't_horz_{nn}'] = t_horz

    decoding_results[f'percentCorrect_vert_{nn}'] = percentCorrect_vert
    decoding_results[f'confusion_vert_{nn}'] = confusion_vert
    decoding_results[f'p_vert_{nn}'] = p_vert
    decoding_results[f't_vert_{nn}'] = t_vert
    decoding_results[f'train_labels_horz_{nn}'] = train_labels_horz
    decoding_results[f'train_labels_vert_{nn}'] = train_labels_vert
    decoding_results[f'trialTime_{nn}'] = t_horz  # or t_vert, they should be the same


    print(f'done iter {nn}')
    
#     for ii in range(int(decoding_start_time * acquisition_rate), 
#                    int(decoding_end_time), 
#                    int(decoding_time_stepsize * acquisition_rate)):
        
#         test_time_i = list(range(max(0, ii - trailing_window), ii + 1))

#         # prepare the training data for this time window, down-sampled
#         train_data_window = []
        
#         if len(test_time_i) < (downsample_interval * acquisition_rate):
#             downsample_data = np.mean(train_data[:, test_time_i, :], axis=1)
#             train_data_window.append(flattenDoppler2D(downsample_data, 1))
#         else:
#             for jj in range(test_time_i[0] + int(downsample_interval * acquisition_rate) - 1,
#                            test_time_i[-1] + 1,
#                            int(downsample_interval * acquisition_rate)):
#                 # downsample_idx = list(range(jj - int(downsample_interval * acquisition_rate) + 1, jj + 1))
#                 # downsample_data = np.mean(train_data[:, downsample_idx, :], axis=1)
#                 # Input: (yPix, xPix, timePoints, trials)
#                 downsample_data = np.mean(train_data[:, :, downsample_idx, :], axis=2)  # (yPix, xPix, nTrials)
#                 downsample_data = downsample_data[:, :, np.newaxis, :]  # (yPix, xPix, 1, nTrials)

# # Use epochOfInterest = 0 (only one window)
#                 train_data_window.append(flattenDoppler2D(downsample_data, 1))

#         if train_data_window:
#             train_data_final = np.concatenate(train_data_window, axis=1) if len(train_data_window) > 1 else train_data_window[0]
#         else:
#             continue

#         if len(axes_to_decode) == 0 or axes_to_decode == 1:
#             cp_horz[ii], p_horz[ii], class_predicted_horz[ii] = crossValidate_short(
#                 train_data_final, train_labels_horz,
#                 classificationMethod=classifier_string, N=N,
#                 validationMethod=validation_string, K=K,
#                 m=m, variance_to_keep=variance_to_keep
#             )

#             percentCorrect_horz[ii] = cp_horz[ii]['CorrectRate'] * 100
#             confusion_horz[ii] = cp_horz[ii]['CountingMatrix']

#         if len(axes_to_decode) == 0 or axes_to_decode == 2:
#             cp_vert[ii], p_vert[ii], class_predicted_vert[ii] = crossValidate_short(
#                 train_data_final, train_labels_vert,
#                 classificationMethod=classifier_string, N=N,
#                 validationMethod=validation_string, K=K,
#                 m=m, variance_to_keep=variance_to_keep
#             )
#             percentCorrect_vert[ii] = cp_vert[ii]['CorrectRate'] * 100
#             confusion_vert[ii] = cp_vert[ii]['CountingMatrix']

#         # save the data to results
#         result_horz[ii] = {
#             'percentCorrect': percentCorrect_horz[ii],
#             'confusion': confusion_horz[ii],
#             'p': p_horz[ii],
#             't': ii - 1
#         }

#         result_vert[ii] = {
#             'percentCorrect': percentCorrect_vert[ii],
#             'confusion': confusion_vert[ii],
#             'p': p_vert[ii],
#             't': ii - 1
#         }

#         print(f"Processing timepoint: {ii}")

#     # Clean up results
#     elim_idx_downsampling = [i for i, x in enumerate(result_horz) if x is None]

#     result_horz = [x for i, x in enumerate(result_horz) if i not in elim_idx_downsampling]
#     result_vert = [x for i, x in enumerate(result_vert) if i not in elim_idx_downsampling]

#     percentCorrect_horz = percentCorrect_horz[~np.isin(range(len(percentCorrect_horz)), elim_idx_downsampling)]
#     confusion_horz = [x for i, x in enumerate(confusion_horz) if i not in elim_idx_downsampling]
#     p_horz = p_horz[~np.isin(range(len(p_horz)), elim_idx_downsampling)]
#     class_predicted_horz = [x for i, x in enumerate(class_predicted_horz) if i not in elim_idx_downsampling]

#     percentCorrect_vert = percentCorrect_vert[~np.isin(range(len(percentCorrect_vert)), elim_idx_downsampling)]
#     confusion_vert = [x for i, x in enumerate(confusion_vert) if i not in elim_idx_downsampling]
#     p_vert = p_vert[~np.isin(range(len(p_vert)), elim_idx_downsampling)]
#     class_predicted_vert = [x for i, x in enumerate(class_predicted_vert) if i not in elim_idx_downsampling]

#     trial_time = trial_time[~np.isin(range(len(trial_time)), elim_idx_downsampling)]

#     # Store results
#     decoding_results[f'label_horz_{nn}'] = label_horz
#     decoding_results[f'label_vert_{nn}'] = label_vert
#     decoding_results[f'keepidx_{nn}'] = keepidx
#     decoding_results[f'train_labels_horz_{nn}'] = train_labels_horz
#     decoding_results[f'train_labels_vert_{nn}'] = train_labels_vert
#     decoding_results[f'result_horz_{nn}'] = result_horz
#     decoding_results[f'result_vert_{nn}'] = result_vert
#     decoding_results[f'percentCorrect_horz_{nn}'] = percentCorrect_horz
#     decoding_results[f'confusion_horz_{nn}'] = confusion_horz
#     decoding_results[f'p_horz_{nn}'] = p_horz
#     decoding_results[f'class_predicted_horz_{nn}'] = class_predicted_horz
#     decoding_results[f'percentCorrect_vert_{nn}'] = percentCorrect_vert
#     decoding_results[f'confusion_vert_{nn}'] = confusion_vert
#     decoding_results[f'p_vert_{nn}'] = p_vert
#     decoding_results[f'class_predicted_vert_{nn}'] = class_predicted_vert
#     decoding_results[f'trialTime_{nn}'] = trial_time

#     print(f'done iter {nn}')

print('finished decoding!')


# Plotting
percentCorrect_horz = decoding_results['percentCorrect_horz_0']
percentCorrect_vert = decoding_results['percentCorrect_vert_0']
p_horz = decoding_results['p_horz_0']
p_vert = decoding_results['p_vert_0']
train_labels_horz = decoding_results['train_labels_horz_0']
train_labels_vert = decoding_results['train_labels_vert_0']
trial_time = decoding_results['trialTime_0']

p_threshold = 0.05
xdat = trial_time

plt.figure(figsize=(10, 6))

ydat = percentCorrect_horz
pdat = p_horz
chance_level = 100 / len(np.unique(train_labels_horz))
best_idx_horz = np.argmax(ydat)
plt.plot(xdat.flatten(), ydat, linewidth=2, color='k', label='horizontal direction')

if np.any(pdat < p_threshold / len(xdat)):
    significant_points = xdat[pdat < p_threshold / len(xdat)]
    plt.scatter(significant_points, np.full_like(significant_points, 99), 
               marker='*', color='k', s=50)

plt.axhline(y=chance_level, linestyle='--', color='k')
plt.text(15, 20, 'horizontal direction', color='k', fontsize=11)

ydat = percentCorrect_vert
pdat = p_vert
chance_level = 100 / len(np.unique(train_labels_vert))

if chance_level < 100:
    best_idx_vert = np.argmax(ydat)
    plt.plot(xdat.flatten(), ydat, linewidth=2, color='b', label='vertical direction')
    
    if np.any(pdat < p_threshold / len(xdat)):
        significant_points = xdat[pdat < p_threshold / len(xdat)]
        plt.scatter(significant_points, np.full_like(significant_points, 98), 
                   marker='*', color='b', s=50)
    
    plt.axhline(y=chance_level, linestyle='--', color='b')
    plt.text(15, 16, 'vertical direction', color='b', fontsize=11)

plt.xlabel('time (s)', fontsize=15)
plt.ylabel('% correct', fontsize=15)
plt.title(f'session {decoding_results["session"]}', fontsize=15)

plt.xlim([xdat[0], xdat[-1]])
plt.ylim([0, 100])

plt.axvline(x=0, color='b')
plt.text(0.3, 80, 'Cue', color='b', fontsize=11)

plt.tight_layout()
plt.show()

# Uncomment to save results
# path = f'C:/Users/sanvi/Documents/MATLAB/behavioral_decoding/results/{decoding_results["session"]}'
# os.makedirs(path, exist_ok=True)
# savetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# processing = 'NoRegister_NoDetrend'
# savename = f'decodingResults_{protocol_name}_S{decoding_results["session"]}_{savetime}_{processing}'
# full_path = os.path.jo  in(path, savename)
# savemat(full_path + '.mat', {'decodingResults': decoding_results})
# plt.savefig(os.path.join(path, f'{processing}.png'))