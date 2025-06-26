import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from detrend_sliding_window import detrend_sliding_window
from crossvalidate_short import crossvalidate_short
from preprocess import preProcess
from flattenDoppler2D import flattenDoppler2D
import warnings
warnings.filterwarnings('ignore')
import h5py


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
    dop_trials_struct = load_matlab_v73(r"D:\Downloads\dop_trials_struct_S164_R1_NoRegister_NoDetrend.mat")
    print("Successfully loaded MATLAB v7.3 file with h5py")
except Exception as e:
    print(f"h5py loading failed: {e}")
    try:
        # Fallback to scipy.io.loadmat for older MATLAB files
        from scipy.io import loadmat
        dop_trials_struct = loadmat(r"D:\Downloads\dop_trials_struct_S164_R1_NoRegister_NoDetrend.mat")
        print("Successfully loaded with scipy.io.loadmat")
    except Exception as e2:
        print(f"Both loading methods failed. h5py error: {e}, scipy error: {e2}")
        raise

# Debug the data structure
print("=== Data Structure Debug ===")
debug_data_structure(dop_trials_struct)

processing = 'NoRegister_NoDetrend'

# choose previous N trials for adaptive training set
N = 175
# choose K for K-fold validation
K = 15
# decoding window in seconds
trailingwindow = 4

# control for preprocessing steps

# Fits an exponential model to the mean signal across depth and uses the
# model to normalize the data. Makes sure that signals from deeper regions
# are properly scaled.
timeGain = False

# Averaging filter that smooths each 2D slice (maps to each window-trial)
# from input data
diskFilter = False

# zscore is applied to every voxel.
zScore = True

# Removes low frequency trends (tissue movement/environment influences)
# from the Doppler data using a sliding window approach of 50 seconds.
detrend = False

# Question: Are windows overlapping?
# Clarify: We are averaging?
if detrend:
    detrend_window_length = 50  # seconds
    iDopP = detrend_sliding_window(dop_trials_struct['dop_trials_struct']['iDopP'], 
                                  detrend_window_length * dop_trials_struct['dop_trials_struct']['acquisition_rate'])
else:
    detrend_window_length = 0
    iDopP = dop_trials_struct['dop_trials_struct']['iDopP']
    
# if detrend:
#     detrend_window_length = 50  # seconds
#     # Access data based on whether it's h5py loaded or scipy loaded
#     if isinstance(dop_trials_struct, dict) and 'iDopP' in dop_trials_struct:
#         iDopP_data = dop_trials_struct['dop_trials_struct']['iDopP']
#         acq_rate = dop_trials_struct['dop_trials_struct']['acquisition_rate']
#     else:
#         # Fallback for scipy.io loaded data
#         iDopP_data = dop_trials_struct['dop_trials_struct']['iDopP'][0,0]
#         acq_rate = dop_trials_struct['dop_trials_struct']['acquisition_rate'][0,0][0,0]
    
#     iDopP = detrend_sliding_window(iDopP_data, detrend_window_length * acq_rate)
# else:
#     detrend_window_length = 0
#     # Access data based on loading method
#     if isinstance(dop_trials_struct, dict) and 'iDopP' in dop_trials_struct:
#         iDopP = dop_trials_struct['dop_trials_struct']['iDopP']
#     else:
#         iDopP = dop_trials_struct['dop_trials_struct']['iDopP'][0,0]

# Performs these preprocessing steps
print(f"iDopP shape before preProcess: {iDopP.shape}")
print(f"iDopP dtype: {iDopP.dtype}")
print(f"Contains NaN: {np.any(np.isnan(iDopP))}")
iDopP = preProcess(iDopP, timeGain=timeGain, diskFilter=diskFilter, zScore=zScore)

# No Register - this means no motion correction but yes detrend
# No Register + No Detrend - no motion correction and no detrend
# Normcorre - motion correction...no detrend (??)

# Now we do PCA + LDA.

# PCA does dimensionality reduction by transforming the
# original variables to new set of uncorrelated variables that capture most
# variance. PCA also does noise reduction as it focuses on the principal
# components that capture the most variance, and inform the patterns in
# brain signals

# LDA separates the classes in the best way possible as it finds the
# discriminants that maximizes the distance between classes

classifierString = 'PCA+LDA'
validationString = 'kFold'

protocol_name = 'ValDir'
n_iter = 1

decodingResults = {}
decodingResults['session'] = dop_trials_struct['dop_trials_struct'].get('session', 0)
decodingResults['run'] = dop_trials_struct['dop_trials_struct'].get('run', 0)
decodingResults['protocol_name'] = protocol_name

## indexing labels

# manually set
only_use_successful_trials = 1
axes_to_decode = []  # [] for both, 1 for horz, 2 for vert


fixation_pos = dop_trials_struct['dop_trials_struct']['fixation_pos']
target_cue_pos = np.squeeze(dop_trials_struct['dop_trials_struct']['target_pos_eachCue']).T
tprime_cue_pos = np.squeeze(dop_trials_struct['dop_trials_struct']['targetPrime_pos_eachCue']).T
    
# reached_min_state = dop_trials_struct['dop_trials_struct'].get('reached_minimum_state_idx', np.array([]))
# success_idx = dop_trials_struct['dop_trials_struct'].get('success_idx', np.array([]))
    
success_trial_idx = np.isin(dop_trials_struct['dop_trials_struct']['reached_minimum_state_idx'].flatten(),
                           dop_trials_struct['dop_trials_struct']['success_idx'].flatten())

decodingResults['only_use_successful_trials'] = only_use_successful_trials
decodingResults['axes_to_decode'] = axes_to_decode

for nn in range(n_iter):
    # for saccade direction ipsi/contra or up/down
    label_horz = np.sign(target_cue_pos[:, 0])
    label_vert = np.sign(target_cue_pos[:, 1])
    
    # set to min of 1
    label_horz = label_horz - np.min(label_horz) + 1
    label_vert = label_vert - np.min(label_vert) + 1
    
    if only_use_successful_trials:
        # FIX: Handle the indexing properly based on the actual shape of success_trial_idx
        if success_trial_idx.ndim == 1:
            # If success_trial_idx is 1D, use it directly
            label_horz[~success_trial_idx] = np.nan
            label_vert[~success_trial_idx] = np.nan
        elif success_trial_idx.ndim == 2:
            # If success_trial_idx is 2D, use the first column as originally intended
            label_horz[~success_trial_idx[:, 0]] = np.nan
            label_vert[~success_trial_idx[:, 0]] = np.nan
        else:
            print(f"Warning: Unexpected success_trial_idx shape: {success_trial_idx.shape}")
            # Use flattened version as fallback
            success_trial_idx_flat = success_trial_idx.flatten()
            if len(success_trial_idx_flat) == len(label_horz):
                label_horz[~success_trial_idx_flat] = np.nan
                label_vert[~success_trial_idx_flat] = np.nan
            else:
                print("Cannot match success_trial_idx to labels, skipping successful trials filter")


    # if only_use_successful_trials:
    #     label_horz[~success_trial_idx[:, 0]] = np.nan
    #     label_vert[~success_trial_idx[:, 0]] = np.nan
    
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
            
            idx_indices = np.where(idx)[0]
            selected_indices = np.random.choice(idx_indices, mincount_combo, replace=False)
            keepidx.extend(selected_indices)
    
    keepidx = np.sort(keepidx)
    
    train_labels_horz = label_horz[keepidx]
    train_labels_vert = label_vert[keepidx]
    
    print(f"train_labels_horz shape: {train_labels_horz.shape}")
    
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
    
    for tt in range(nTrials):
        # Access lookup tables based on loading method
        if isinstance(dop_trials_struct, dict):
            lookup_tables = dop_trials_struct['dop_trials_struct'].get('lookup_tables_3D_to_4D', [])
            if isinstance(lookup_tables, list) and len(lookup_tables) > 0:
                timeidx = lookup_tables[0][:, tt] - 1  # Convert to 0-based indexing
            else:
                # Create default time indices if lookup tables not available
                timeidx = np.arange(trial_n_timestamps)
        else:
            timeidx = dop_trials_struct['dop_trials_struct']['lookup_tables_3D_to_4D'][0,0][0][:, tt] - 1  # Convert to 0-based indexing
        
        valid_idx = ~np.isnan(timeidx)
        if np.any(valid_idx):
            timeidx = timeidx[valid_idx].astype(int)
            # Make sure indices are within bounds
            timeidx = timeidx[timeidx < nWindows]
            if len(timeidx) > 0:
                train_data[:, :, valid_idx[:len(timeidx)], tt] = iDopP[:, :, timeidx]
    
    train_data = train_data.reshape(yPix * xPix, trial_n_timestamps, nTrials)
    train_data = train_data[:, :, keepidx]
    
    # decode across trial time
    nWindows = train_data.shape[1]
    
    # Determine dimensionality of cPCA subspace (if used)
    # For now, using (number of classes - 1), i.e m = 2 for 3 classes
    m = len(np.unique(train_labels_horz)) - 1
    
    # allocate memory fixation position
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
    
    downsample_interval = 1  # in seconds. average sequential frames within this interval together
    decoding_time_stepsize = 1  # in seconds. instead of decoding every frame, step forward this many second with each loop
    
    # set decoding start and stop time
    decoding_start_time = decoding_time_stepsize  # earliest start
    decoding_end_time = nWindows  # latest stop
    
    # Access acquisition rate based on loading method
    if isinstance(dop_trials_struct, dict):
        acquisition_rate = dop_trials_struct['dop_trials_struct'].get('acquisition_rate', 1.0)
        if isinstance(acquisition_rate, (list, np.ndarray)):
            acquisition_rate = acquisition_rate[0] if len(acquisition_rate) > 0 else 1.0
    else:
        acquisition_rate = dop_trials_struct['dop_trials_struct']['acquisition_rate'][0,0][0,0]
    
    if (downsample_interval * acquisition_rate) % 1 != 0:
        raise ValueError('error: the downsample_interval must correspond to an integer number of frames')
    if (decoding_time_stepsize * acquisition_rate) % 1 != 0:
        raise ValueError('error: the decoding_time_stepsize must correspond to an integer number of frames')
    if (trailingwindow * acquisition_rate) % 1 != 0:
        raise ValueError('error: the trailingwindow must correspond to an integer number of frames')
    
    trailingwindow = int(trailingwindow * acquisition_rate)  # TC. timepoints behind current that are included in decoder
    
    # Access trial time vector based on loading method
    if isinstance(dop_trials_struct, dict):
        trial_timevec = dop_trials_struct['dop_trials_struct'].get('trial_timevec', [np.arange(trial_n_timestamps)])
        if isinstance(trial_timevec, list) and len(trial_timevec) > 0:
            trialTime = np.array(trial_timevec[0]).flatten()
        else:
            trialTime = np.arange(trial_n_timestamps)
    else:
        trialTime = dop_trials_struct['dop_trials_struct']['trial_timevec'][0,0][0].flatten()
    
    print('Time with for loop')
    variance_to_keep = 95

    trainData = []

    for ii in range(
        int(decoding_start_time * acquisition_rate),
        int(decoding_end_time) + 1,
        int(decoding_time_stepsize * acquisition_rate)
    ):
        testTimeI = list(range(max(1, ii - trailingwindow), ii + 1))

        if (max(testTimeI) - min(testTimeI)) < (downsample_interval * acquisition_rate):
            # Use full test window as single epoch
            downsample_data = np.mean(train_data[:, testTimeI, :], axis=1)  # shape: (yPix*xPix, nTrials)
            yPix = 133
            xPix = 128
            downsample_data_reshaped = downsample_data.reshape((yPix, xPix, 1, train_data.shape[2]))
            flattened = flattenDoppler2D(downsample_data_reshaped, epochOfInterest=[0])
            trainData.append(flattened)
        else:
            for jj in range(
                testTimeI[0] - 1 + int(downsample_interval * acquisition_rate),
                testTimeI[-1] + 1,
                int(downsample_interval * acquisition_rate)
            ):
                downsample_idx = list(range(jj - int(downsample_interval * acquisition_rate) + 1, jj + 1))

                # Validate indices
                if min(downsample_idx) < 0 or max(downsample_idx) >= train_data.shape[1]:
                    print(f"Skipping jj={jj} due to invalid downsample_idx: {downsample_idx}")
                    continue

                downsample_data = np.mean(train_data[:, downsample_idx, :], axis=1)  # shape: (yPix*xPix, nTrials)
                # yPix_xPix = int(np.sqrt(train_data.shape[0]))
                # downsample_data_reshaped = downsample_data.reshape((yPix_xPix, yPix_xPix, 1, train_data.shape[2]))
                yPix = 133
                xPix = 128
                downsample_data_reshaped = downsample_data.reshape((yPix, xPix, 1, train_data.shape[2]))

                flattened = flattenDoppler2D(downsample_data_reshaped, epochOfInterest=[0])
                
                if np.isnan(flattened).any():
                    print(f"NaNs detected in flattened data at jj={jj}")
                    continue
                
                trainData.append(flattened)

    # Combine all
    if trainData:
        trainData = np.concatenate(trainData, axis=0) if len(trainData) > 1 else trainData[0]

    
    # want to keep 0.985 percent of variance
    # average frames in last 5 seconds
    # for ii in range(int(decoding_start_time * acquisition_rate), 
    #                 int(decoding_end_time), 
    #                 int(decoding_time_stepsize * acquisition_rate)):
        
    #     testTimeI = np.arange(max(0, ii - trailingwindow), ii + 1)
        
    #     # prepare the training data for this time window, down-sampled
    #     trainData = []
    #     if len(testTimeI) < (downsample_interval * acquisition_rate):
    #         # Not enough data — use mean over available
    #         downsample_data = np.mean(train_data[:, testTimeI, :], axis=1)
    #         trainData.append(flattenDoppler2D(downsample_data, 1))
    #     else:
    #         # Main downsampling loop
       
    #     # if len(testTimeI) < (downsample_interval * acquisition_rate):
    #     #     downsample_data = np.mean(train_data[:, testTimeI, :], axis=1)
    #     #     trainData.append(flattenDoppler2D(downsample_data, axis=1))
    #     # else:
    #         # for jj in range(int(testTimeI[0] + downsample_interval * acquisition_rate - 1), 
    #         #                int(testTimeI[-1] + 1), 
    #         #                int(downsample_interval * acquisition_rate)):
    #         #     downsample_idx = np.arange(int(jj - downsample_interval * acquisition_rate), jj)
    #         #     downsample_data = np.mean(train_data[:, downsample_idx, :], axis=1)
    #         #     trainData.append(flattenDoppler2D(downsample_data, axis=1))
    #         for jj in range(testTimeI[0] + int(downsample_interval * acquisition_rate) - 1,
    #                     testTimeI[-1] + 1,
    #                     int(downsample_interval * acquisition_rate)):
    #             downsample_idx = list(range(jj - int(downsample_interval * acquisition_rate) + 1, jj + 1))
    #             # downsample_data = np.mean(train_data[:, downsample_idx, :], axis=1)

    #             # # Input to flattenDoppler2D is (yPix, xPix, time, nTrials) — emulate it:
    #             # yPix_xPix = int(np.sqrt(train_data.shape[0]))
    #             # downsample_data_reshaped = downsample_data.reshape((yPix_xPix, yPix_xPix, 1, train_data.shape[2]))

    #             # downsample_data = np.mean(train_data[:, testTimeI, :], axis=1)  # shape: (yPix * xPix, nTrials)
    #             downsample_data = np.mean(train_data[:, downsample_idx, :], axis=1)

    #             # Reshape properly using original yPix and xPix:
    #             downsample_data_reshaped = downsample_data.reshape((yPix, xPix, 1, train_data.shape[2]))

    #             # Now call flattenDoppler2D
    #             trainData.append(flattenDoppler2D(downsample_data_reshaped, 0))

            
        
    #     if trainData:
    #         trainData = np.concatenate(trainData, axis=1) if len(trainData) > 1 else trainData[0]
        
        if len(axes_to_decode) == 0 or axes_to_decode == 1:
            cp_horz[ii], p_horz[ii], class_predicted_horz[ii] = crossvalidate_short(
                trainData, train_labels_horz,
                classificationMethod=classifierString, N=N,
                validationMethod=validationString, K=K,
                m=m, variance_to_keep=variance_to_keep)
            
            percentCorrect_horz[ii] = cp_horz[ii]['CorrectRate'] * 100
            confusion_horz[ii] = cp_horz[ii]['CountingMatrix']
            
            best_horz_idx = np.nanargmin(p_horz)
        
        if len(axes_to_decode) == 0 or axes_to_decode == 2:
            cp_vert[ii], p_vert[ii], class_predicted_vert[ii] = crossvalidate_short(
                trainData, train_labels_vert,
                classificationMethod=classifierString, N=N,
                validationMethod=validationString, K=K,
                m=m, variance_to_keep=variance_to_keep)
            
            percentCorrect_vert[ii] = cp_vert[ii]['CorrectRate'] * 100
            confusion_vert[ii] = cp_vert[ii]['CountingMatrix']
        
        # save the data to results
        result_horz[ii] = {
            'percentCorrect': percentCorrect_horz[ii],
            'confusion': confusion_horz[ii],
            'p': p_horz[ii],
            't': ii - 1
        }
        
        result_vert[ii] = {
            'percentCorrect': percentCorrect_vert[ii],
            'confusion': confusion_vert[ii],
            'p': p_vert[ii],
            't': ii - 1
        }
        
        print(f"Processing timepoint: {ii}")
    
    downsample_check_struct = result_horz
    
    # remove the missing timepoints due to time_stepsize downsampling
    elim_idx_downsampling = np.zeros(len(downsample_check_struct), dtype=bool)
    for ii in range(len(downsample_check_struct)):
        if downsample_check_struct[ii] is None:
            elim_idx_downsampling[ii] = True
    
    # Remove eliminated indices
    result_horz = [result_horz[i] for i in range(len(result_horz)) if not elim_idx_downsampling[i]]
    result_vert = [result_vert[i] for i in range(len(result_vert)) if not elim_idx_downsampling[i]]
    
    percentCorrect_horz = percentCorrect_horz[~elim_idx_downsampling]
    confusion_horz = [confusion_horz[i] for i in range(len(confusion_horz)) if not elim_idx_downsampling[i]]
    p_horz = p_horz[~elim_idx_downsampling]
    class_predicted_horz = [class_predicted_horz[i] for i in range(len(class_predicted_horz)) if not elim_idx_downsampling[i]]
    
    percentCorrect_vert = percentCorrect_vert[~elim_idx_downsampling]
    confusion_vert = [confusion_vert[i] for i in range(len(confusion_vert)) if not elim_idx_downsampling[i]]
    p_vert = p_vert[~elim_idx_downsampling]
    class_predicted_vert = [class_predicted_vert[i] for i in range(len(class_predicted_vert)) if not elim_idx_downsampling[i]]
    
    trialTime = trialTime[~elim_idx_downsampling]
    
    # create decoding results struct
    if 'label_horz' not in decodingResults:
        decodingResults['label_horz'] = []
        decodingResults['label_vert'] = []
        decodingResults['keepidx'] = []
        decodingResults['train_labels_horz'] = []
        decodingResults['train_labels_vert'] = []
        decodingResults['result_horz'] = []
        decodingResults['result_vert'] = []
        decodingResults['percentCorrect_horz'] = []
        decodingResults['confusion_horz'] = []
        decodingResults['p_horz'] = []
        decodingResults['class_predicted_horz'] = []
        decodingResults['percentCorrect_vert'] = []
        decodingResults['confusion_vert'] = []
        decodingResults['p_vert'] = []
        decodingResults['class_predicted_vert'] = []
        decodingResults['trialTime'] = []
    
    decodingResults['label_horz'].append(label_horz)
    decodingResults['label_vert'].append(label_vert)
    decodingResults['keepidx'].append(keepidx)
    decodingResults['train_labels_horz'].append(train_labels_horz)
    decodingResults['train_labels_vert'].append(train_labels_vert)
    decodingResults['result_horz'].append(result_horz)
    decodingResults['result_vert'].append(result_vert)
    decodingResults['percentCorrect_horz'].append(percentCorrect_horz)
    decodingResults['confusion_horz'].append(confusion_horz)
    decodingResults['p_horz'].append(p_horz)
    decodingResults['class_predicted_horz'].append(class_predicted_horz)
    decodingResults['percentCorrect_vert'].append(percentCorrect_vert)
    decodingResults['confusion_vert'].append(confusion_vert)
    decodingResults['p_vert'].append(p_vert)
    decodingResults['class_predicted_vert'].append(class_predicted_vert)
    decodingResults['trialTime'].append(trialTime)
    
    print(f'done iter {nn + 1}')

print('finished decoding!')

## create variables from previously saved decoding_results struct, for plotting

percentCorrect_horz = np.array(decodingResults['percentCorrect_horz']).flatten()
percentCorrect_vert = np.array(decodingResults['percentCorrect_vert']).flatten()

p_horz = np.array(decodingResults['p_horz']).flatten()
p_vert = np.array(decodingResults['p_vert']).flatten()

train_labels_horz = np.array(decodingResults['train_labels_horz']).flatten()
train_labels_vert = np.array(decodingResults['train_labels_vert']).flatten()

trialTime = np.array(decodingResults['trialTime']).flatten()

# redo this so can handle multiple iters
confusion_horz = decodingResults['confusion_horz'][0]
confusion_vert = decodingResults['confusion_vert'][0]

## plot results direction

p_threshold = 0.05
xdat = trialTime

plt.figure(figsize=(10, 8))

# Horizontal direction
ydat = np.mean(percentCorrect_horz.reshape(1, -1), axis=0)
pdat = np.mean(p_horz.reshape(1, -1), axis=0)
chance_level = 100 / len(np.unique(train_labels_horz))
bestidx_horz = np.argmax(ydat)

plt.plot(xdat, ydat, linewidth=2, color='k', label='horizontal direction')

# plot p value & labels
significant_points = pdat < (p_threshold / len(xdat))
if np.any(significant_points):
    plt.plot(xdat[significant_points], np.full(np.sum(significant_points), 99), '*k', markersize=6)

plt.axhline(y=chance_level, color='k', linestyle='--')
plt.text(15, 20, 'horizontal direction', color='k', fontsize=11)

# Vertical direction
ydat_vert = np.mean(percentCorrect_vert.reshape(1, -1), axis=0)
pdat_vert = np.mean(p_vert.reshape(1, -1), axis=0)
chance_level_vert = 100 / len(np.unique(train_labels_vert))

if chance_level_vert < 100:
    bestidx_vert = np.argmax(ydat_vert)
    plt.plot(xdat, ydat_vert, linewidth=2, color='b', label='vertical direction')
    
    # plot p value & labels
    significant_points_vert = pdat_vert < (p_threshold / len(xdat))
    if np.any(significant_points_vert):
        plt.plot(xdat[significant_points_vert], np.full(np.sum(significant_points_vert), 98), '*b', markersize=6)
    
    plt.axhline(y=chance_level_vert, color='b', linestyle='--')
    plt.text(15, 16, 'vertical direction', color='b', fontsize=11)

plt.xlabel('time (s)')
plt.ylabel('% correct')
plt.title(f"session {decodingResults['session']}")
plt.xlim([xdat[0], xdat[-1]])
plt.ylim([0, 100])
plt.gca().set_aspect('equal')

plt.axvline(x=0, color='b')
plt.text(0.3, 80, 'Cue', color='b', fontsize=11)

plt.tick_params(labelsize=15)
plt.show()

# Uncomment to save results
# savetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# savename = f"decodingResults_{protocol_name}_S{decodingResults['session']}_{savetime}_{processing}"
# savemat(savename + '.mat', {'decodingResults': decodingResults})