import numpy as np
import warnings
from scipy.stats import zscore
from skimage import transform
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
import cv2
from normcorre_doppler import normcorre_doppler

def correct_motion_3d(dop_in, core_params, verbose=False, template=None):
    """
    This function uses rigid body transforms to correct errant motion in the
    fUS doppler sequence. It does this on a single-trial basis, i.e. the
    data passed to this function is of the size X x Y x nTimepoints.
    
    Parameters:
    -----------
    dop_in : numpy.ndarray
        xPixels x yPixels x nTimepoints
    core_params : dict
        The coreparams structure used in createDoppler. Should contain
        core_params['method'] of a valid choice: 'imregister', 'rigid', or 'normcorre'
    verbose : bool, optional
        Default False
    template : numpy.ndarray, optional
        Template for normcorre method
        
    Returns:
    --------
    dop_out : numpy.ndarray
        Corrected array xPixels x yPixels x nTimepoints
    core_params : dict
        Updated with reference frame and trial info
    template : numpy.ndarray
        Template (for normcorre method)
    """
    
    # Check the registration method
    if 'method' not in core_params:
        warnings.warn('defaulting to normcorre method!')
        method = 'normcorre'
    else:
        method = core_params['method']
    
    if method not in ['imregister', 'rigid', 'normcorre']:
        raise ValueError(f"Invalid method: {method}")
    
    use_template = template is not None
    
    # Select reference frame and window
    n_timepoints = dop_in.shape[2]
    
    if method in ['imregister', 'rigid']:
        ref_frame = int(np.ceil(n_timepoints / 4))
        fixed_frame = zscore(dop_in[:, :, ref_frame], axis=None)
    elif method == 'normcorre':
        ref_trial = 0
        ref_frame = 0
    
    if method != 'normcorre':
        print(f'Reference: timepoint {ref_frame+1}/{dop_in.shape[2]}')
    
    # Save out updated core_params structure
    core_params['motionCorrection_refFrame'] = ref_frame
    
    # Initialize output data
    dop_out = np.zeros_like(dop_in)
    
    # IMREGISTER METHOD (using scikit-image)
    if method == 'imregister':
        interpolation_method = 'linear'
        
        # Set up a matrix to monitor for failed registrations
        failed_registrations = np.zeros(dop_out.shape[2], dtype=bool)
        
        # For each frame in the sequence
        for timepoint in range(n_timepoints):
            # Select the current frame to work on
            moving_frame = dop_in[:, :, timepoint]
            moving_frame_zscore = zscore(moving_frame, axis=None)
            
            try:
                # Use phase cross correlation for registration
                shift, error, diffphase = phase_cross_correlation(
                    fixed_frame, moving_frame_zscore, upsample_factor=100
                )
                
                # Create affine transform
                tform = AffineTransform(translation=shift[::-1])  # reverse for x,y order
                
                # Apply transform
                dop_out[:, :, timepoint] = warp(
                    moving_frame,
                    tform.inverse,
                    output_shape=fixed_frame.shape,
                    preserve_range=True,
                    cval=0
                )
                
            except Exception as e:
                if verbose:
                    print(f"Registration failed for timepoint {timepoint}: {e}")
                failed_registrations[timepoint] = True
                dop_out[:, :, timepoint] = moving_frame
        
        # Fix the failed registrations by taking the closest successful transform
        for timepoint in range(len(failed_registrations)):
            if failed_registrations[timepoint]:
                # Iteratively find closest good timepoint
                found_replacement_timepoint = False
                counter = 0
                
                while not found_replacement_timepoint:
                    counter += 1
                    if (timepoint - counter >= 0 and 
                        not failed_registrations[timepoint - counter]):
                        found_replacement_timepoint = True
                        counter = -counter
                    elif (timepoint + counter < len(failed_registrations) and 
                          not failed_registrations[timepoint + counter]):
                        found_replacement_timepoint = True
                
                # Find the image transform for that new frame and apply to the misaligned frame
                reference_frame = zscore(dop_in[:, :, timepoint + counter], axis=None)
                moving_frame = dop_in[:, :, timepoint]
                
                try:
                    shift, error, diffphase = phase_cross_correlation(
                        fixed_frame, reference_frame, upsample_factor=100
                    )
                    tform = AffineTransform(translation=shift[::-1])
                    
                    dop_out[:, :, timepoint] = warp(
                        moving_frame,
                        tform.inverse,
                        output_shape=fixed_frame.shape,
                        preserve_range=True,
                        cval=0
                    )
                except:
                    dop_out[:, :, timepoint] = moving_frame
    
    # RIGID METHOD (using OpenCV)
    elif method == 'rigid':
        print('Correcting motion using rigid transform...')
        
        for timepoint in range(n_timepoints):
            # Select the current frame to work on
            moving_frame = dop_in[:, :, timepoint]
            
            # Convert to uint8 for OpenCV
            fixed_uint8 = ((fixed_frame - fixed_frame.min()) / 
                          (fixed_frame.max() - fixed_frame.min()) * 255).astype(np.uint8)
            moving_uint8 = ((moving_frame - moving_frame.min()) / 
                           (moving_frame.max() - moving_frame.min()) * 255).astype(np.uint8)
            
            # Find transform using OpenCV
            try:
                warp_matrix = cv2.estimateRigidTransform(
                    moving_uint8, fixed_uint8, fullAffine=False
                )
                
                if warp_matrix is not None:
                    # Apply transform
                    dop_out[:, :, timepoint] = cv2.warpAffine(
                        moving_frame, warp_matrix, 
                        (moving_frame.shape[1], moving_frame.shape[0])
                    )
                else:
                    dop_out[:, :, timepoint] = moving_frame
                    
            except Exception as e:
                if verbose:
                    print(f"Rigid registration failed for timepoint {timepoint}: {e}")
                dop_out[:, :, timepoint] = moving_frame
    
    # NORMCORRE METHOD
    elif method == 'normcorre':
        print('Correcting motion using normcorre')
        if use_template:
            dop_out, template = normcorre_doppler(dop_in, verbose, template)
        else:
            dop_out, template = normcorre_doppler(dop_in, verbose)
    
    return dop_out, core_params, template