import numpy as np
import matplotlib.pyplot as plt
import warnings


def flattenDoppler2D(dopIn, epochOfInterest=None, verbose=False, threeD=False):
    """
    flattenDoppler2D is designed to reshape doppler data into arrays that are
    typically used in the BCI training algorithms.
    
    example use:
    dopOut = flattenDoppler2D(dopIn, epochOfInterest=[0, 1], 
                             verbose=False, threeD=True)
    
    Returns:
    dopOut: flattened 2D doppler. If threeD is False (default) this is...
            (nTrials x nPixels*nWindowsUsed). 
            If threeD is True this is...
            (nPixels x nWindows x nTrials)
    
    Parameters:
    dopIn: 4D doppler data (yPix x xPix x nWindows x nTrials)
           can also pass 3D flattened data, e.g. (yPix*xPix) x nWindows x nTrials
    
    epochOfInterest: time indices to flatten over, e.g. [4, 5, 6] (0-indexed)
    
    verbose: boolean - show plots and info
    threeD: boolean - return 3D array instead of 2D
    """
    
    # Handle default epochOfInterest
    if epochOfInterest is None:
        epochOfInterest = list(range(dopIn.shape[2]))
        if verbose:
            warnings.warn('Epoch of interest not given. Passing all data')
    
    # Convert to list if single value
    if isinstance(epochOfInterest, int):
        epochOfInterest = [epochOfInterest]
    
    # Handle 3D input by adding singleton dimension
    if dopIn.ndim == 3:
        dopIn = dopIn.reshape((1, dopIn.shape[0], dopIn.shape[1], dopIn.shape[2]))
    
    # Get data dimension sizes
    yPix, xPix, nWindows, nTrials = dopIn.shape
    epochLength = len(epochOfInterest)
    
    # Stack subsequent windows side by side: 4D->3D images of (yPix, xPix*nWindows, nTrials)
    if epochLength == 1:
        # Single doppler measurement - squeeze that dimension
        # This leaves a (yPix, xPix, nTrials) array
        dop3D = np.squeeze(dopIn[:, :, epochOfInterest[0], :])
        if dopIn.shape[0] == 1:
            dop3D = dop3D.reshape((1, dop3D.shape[0], dop3D.shape[1]))
    elif threeD:
        # Preserves the 4D structure, just selects the epoch
        dop4D = dopIn[:, :, epochOfInterest, :]
    else:
        dopTmp = dopIn[:, :, epochOfInterest, :]
        # Stack the images in the epoch side by side
        dop3D = dopTmp.reshape((yPix, epochLength * xPix, nTrials))
    
    # Plot the mean figures
    if verbose and not threeD and dopIn.ndim == 4:
        map_data = np.mean(dop3D, axis=2)
        map_data = (map_data - np.min(map_data)) / (np.max(map_data) - np.min(map_data))
        map_data = np.power(map_data, 1/4)  # 4th root
        
        plt.figure()
        plt.imshow(map_data, aspect='auto')
        plt.title(f'Mean across {dop3D.shape[2]} images')
        plt.colorbar()
        
        # Print size of images
        print(f'{dop3D.shape[2]} images found. Image Size: {dop3D.shape[0]} x {dop3D.shape[1]}')
        input('Press Enter to continue...')
        plt.close('all')
    
    # Reduce dimension
    if threeD:
        # Resize the images into 3D
        nWindowsKept = dop4D.shape[2]
        dopOut = dop4D.reshape((yPix * xPix, nWindowsKept, nTrials))
    else:
        # Handle case where original input was 3D
        if dopIn.ndim == 3:
            dop3D = dop3D.reshape((1, dop3D.shape[0], dop3D.shape[1]))
        
        # Resize the images into 2D
        nImages = dop3D.shape[2]
        nPixels = dop3D.shape[0] * dop3D.shape[1]
        dopOut = np.zeros((nImages, nPixels))
        
        # dopOut is size nImages x (xPixels*yPixels)
        for i in range(dop3D.shape[2]):
            currentImage = dop3D[:, :, i]
            dopOut[i, :] = currentImage.reshape(1, nPixels)
    
    return dopOut