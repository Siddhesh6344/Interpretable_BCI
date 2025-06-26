import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import ndimage
from scipy.interpolate import interp2d
from scipy.stats import zscore
from skimage.transform import resize
from scipy.signal import correlate2d
from statsmodels.tsa.arima.model import ARIMA

def preProcess(dopIn, interp=False, uninterp=False, timeGain=False, zScore=False,
               diskFilter=2, downsample=False, preWhiten=False, crop=False,
               detrend=False, verbose=False):
    
    original_3d = False
    if len(dopIn.shape) == 3:
        original_3d = True
        dopIn = dopIn[:, :, :, np.newaxis]
        if verbose:
            print("Input was 3D, converted to 4D")
    elif len(dopIn.shape) != 4:
        raise ValueError(f"Input must be 3D or 4D array, got {len(dopIn.shape)}D")
    
    if isinstance(diskFilter, bool) and diskFilter:
        warnings.warn('DiskFilter boolean deprecated, using filter size 2.')
        diskFilter = 2

    # Apply preprocessing pipeline
    if interp:
        dopIn = interDop(dopIn)
    if uninterp:
        dopIn = uninterpDop(dopIn)
    if timeGain:
        dopIn = timeGainCompensation(dopIn, verbose)
    if zScore:
        dopIn = dopZ(dopIn)
    if diskFilter:
        dopIn = diskFilterFunc(dopIn, diskFilter)
    if downsample:
        dopIn = downsampleFunc(dopIn, downsample)
    if preWhiten:
        dopIn = white(dopIn)
    
    if original_3d:
        return dopIn[:, :, :, 0]
    else:
        return dopIn

def dopZ(dataIn):
    if len(dataIn.shape) == 3:
        dataIn = dataIn[:, :, :, np.newaxis]
        return dopZ(dataIn)[:, :, :, 0]
    
    n_depth, n_width, n_windows, n_trials = dataIn.shape
    dataOut = np.empty_like(dataIn)
    
    for trial in range(n_trials):
        flat = dataIn[:, :, :, trial].reshape(-1, n_windows)
        z = zscore(flat, axis=1, nan_policy='omit')
        z = np.nan_to_num(z, nan=0.0)
        dataOut[:, :, :, trial] = z.reshape(n_depth, n_width, n_windows)
    
    return dataOut

def diskFilterFunc(dataIn, filter_size):
    if len(dataIn.shape) == 3:
        dataIn = dataIn[:, :, :, np.newaxis]
        return diskFilterFunc(dataIn, filter_size)[:, :, :, 0]
    
    n_depth, n_width, n_windows, n_trials = dataIn.shape
    dataOut = np.full_like(dataIn, np.nan)

    y, x = np.ogrid[-filter_size:filter_size+1, -filter_size:filter_size+1]
    mask = x*x + y*y <= filter_size*filter_size
    h = mask.astype(float) / mask.sum()

    for w in range(n_windows):
        for t in range(n_trials):
            if np.all(np.isnan(dataIn[:, :, w, t])):
                continue
            dataOut[:, :, w, t] = correlate2d(dataIn[:, :, w, t], h, mode='same', boundary='symm')
    
    return dataOut

def timeGainCompensation(dataIn, verbose=False):
    if len(dataIn.shape) == 3:
        dataIn = dataIn[:, :, :, np.newaxis]
        return timeGainCompensation(dataIn, verbose)[:, :, :, 0]

    n_depth, n_width, n_windows, n_trials = dataIn.shape
    depthInds = np.arange(1, n_depth + 1)
    indicesForModeling = np.arange(int(0.1 * n_depth), n_depth)

    dataIn_flat = dataIn.reshape(n_depth, n_width, -1)
    depthMeanSignal = np.nanmean(np.nanmean(dataIn_flat, axis=1), axis=1)

    valid = ~np.isnan(depthMeanSignal[indicesForModeling])
    if np.sum(valid) < 2:
        warnings.warn("Insufficient valid data for time gain.")
        return dataIn

    log_signal = np.log(np.maximum(depthMeanSignal[indicesForModeling][valid], 1e-10))
    x_fit = depthInds[indicesForModeling][valid]

    try:
        b, log_a = np.polyfit(x_fit, log_signal, 1)
        a = np.exp(log_a)
        depthMean = a * np.exp(b * depthInds)

        if verbose:
            plt.plot(depthInds, depthMeanSignal, 'r.', label="Raw Mean")
            plt.plot(depthInds, depthMean, 'k-', label="Model")
            plt.xlabel('Depth')
            plt.ylabel('Signal')
            plt.legend()
            plt.show()

        depthMean_safe = np.maximum(depthMean, 1e-10)
        norm = depthMean_safe[:, np.newaxis, np.newaxis, np.newaxis]
        dataOut = dataIn / norm
        dataOut *= 100
        return dataOut

    except Exception as e:
        warnings.warn(f"TGC fit failed: {e}")
        return dataIn

def interDop(dataIn):
    raise NotImplementedError("Interpolation not used in this example.")

def uninterpDop(dataIn):
    raise NotImplementedError("Uninterpolation not used in this example.")

def downsampleFunc(dataIn, downsample_size):
    raise NotImplementedError("Downsampling not used in this example.")

def white(dataIn):
    raise NotImplementedError("Prewhitening not used in this example.")
