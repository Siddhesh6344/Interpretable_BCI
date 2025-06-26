import numpy as np
import warnings
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def detrend_sliding_window(dop, window_length=30):
    """
    Detrend Doppler data using a sliding window approach.
    
    Parameters:
    -----------
    dop : numpy.ndarray
        4-D array of dimensions (yPixels, xPixels, nWindows, nTrials)
        Can also be 3-D: (yPixels, xPixels, time)
    window_length : int or float, optional
        Window length used in the sliding window (default: 30)
        Will be converted to integer
    
    Returns:
    --------
    numpy.ndarray
        Detrended Doppler data with same dimensions as input
    """
    
    # Ensure window_length is integer
    window_length = int(np.round(window_length))
    
    # Get dimensions
    if dop.ndim == 4:
        yPixels, xPixels, nWindows, nTrials = dop.shape
        # Reshape from 4D to 3D
        dop = dop.reshape(yPixels, xPixels, nTrials * nWindows)
        reshape_flag = True
    elif dop.ndim == 3:
        yPixels, xPixels, _ = dop.shape
        reshape_flag = False
    else:
        warnings.warn('Doppler data is badly sized!')
        return dop
    
    # Process each pixel
    for y in range(yPixels):
        for x in range(xPixels):
            this_vector = dop[y, x, :]
            
            # Compute moving average using causal window (no future samples)
            window_avg = moving_average_causal(this_vector, window_length)
            
            # Compute overall average for this pixel
            avg_avg = np.mean(this_vector)
            
            # Detrend and add back the overall average
            dop[y, x, :] = this_vector - window_avg + avg_avg
    
    # Reshape back to 4D if necessary
    if reshape_flag:
        dop = dop.reshape(yPixels, xPixels, nWindows, nTrials)
    
    return dop


def detrend_sliding_window_parallel(dop, window_length=30, n_workers=None):
    """
    Parallel version of detrend_sliding_window for better performance.
    
    Parameters:
    -----------
    dop : numpy.ndarray
        4-D array of dimensions (yPixels, xPixels, nWindows, nTrials)
        Can also be 3-D: (yPixels, xPixels, time)
    window_length : int or float, optional
        Window length used in the sliding window (default: 30)
        Will be converted to integer
    n_workers : int, optional
        Number of parallel workers (default: number of CPU cores)
    
    Returns:
    --------
    numpy.ndarray
        Detrended Doppler data with same dimensions as input
    """
    
    # Ensure window_length is integer
    window_length = int(np.round(window_length))
    
    # Get dimensions
    original_shape = dop.shape
    if dop.ndim == 4:
        yPixels, xPixels, nWindows, nTrials = dop.shape
        # Reshape from 4D to 3D
        dop = dop.reshape(yPixels, xPixels, nTrials * nWindows)
        reshape_flag = True
    elif dop.ndim == 3:
        yPixels, xPixels, _ = dop.shape
        reshape_flag = False
    else:
        warnings.warn('Doppler data is badly sized!')
        return dop
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Create list of pixel coordinates
    pixel_coords = [(y, x) for y in range(yPixels) for x in range(xPixels)]
    
    # Process pixels in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit jobs
        futures = {
            executor.submit(process_pixel, dop[y, x, :], window_length): (y, x)
            for y, x in pixel_coords
        }
        
        # Collect results
        for future in as_completed(futures):
            y, x = futures[future]
            try:
                dop[y, x, :] = future.result()
            except Exception as exc:
                print(f'Pixel ({y}, {x}) generated an exception: {exc}')
    
    # Reshape back to original dimensions if necessary
    if reshape_flag:
        dop = dop.reshape(original_shape)
    
    return dop


def process_pixel(pixel_vector, window_length):
    """
    Process a single pixel's time series data.
    
    Parameters:
    -----------
    pixel_vector : numpy.ndarray
        1D array representing the time series for one pixel
    window_length : int
        Window length for moving average
    
    Returns:
    --------
    numpy.ndarray
        Detrended pixel time series
    """
    # Compute moving average using causal window
    window_avg = moving_average_causal(pixel_vector, window_length)
    
    # Compute overall average
    avg_avg = np.mean(pixel_vector)
    
    # Detrend and add back the overall average
    return pixel_vector - window_avg + avg_avg


def moving_average_causal(data, window_length):
    """
    Compute causal moving average (no future samples).
    Equivalent to MATLAB's movmean with [window_length 0] specification.
    
    Parameters:
    -----------
    data : numpy.ndarray
        1D input data
    window_length : int
        Window length (number of past samples to include)
    
    Returns:
    --------
    numpy.ndarray
        Moving average with same length as input
    """
    if len(data) == 0:
        return data
    
    # Pad the beginning with the first value
    padded_data = np.concatenate([np.full(window_length-1, data[0]), data])
    
    # Use uniform filter for moving average
    weights = np.ones(window_length) / window_length
    moving_avg = np.convolve(padded_data, weights, mode='valid')
    
    return moving_avg
