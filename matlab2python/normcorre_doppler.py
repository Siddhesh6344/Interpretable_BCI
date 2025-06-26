import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
from skimage import registration
from skimage.transform import warp, AffineTransform
import time
import cv2
import os
from datetime import datetime

def normcorre_doppler(imageIn, verbose=False, template=None):
    """
    Motion correction for imaging data using NoRMCorre algorithm
    
    Parameters:
    imageIn: numpy array - input image data (3D or 4D)
    verbose: bool - whether to show detailed output and plots
    template: numpy array - optional template for registration
    
    Returns:
    imageOut: numpy array - motion corrected image data
    template1: numpy array - registration template
    """
    
    # Handle template parameter
    if template is not None:
        useTemplate = True
    else:
        useTemplate = False
    
    # Handle different input dimensions
    if imageIn.ndim == 4:
        yPixels, xPixels, nWindows, nTrials = imageIn.shape
        Y = imageIn.reshape(yPixels, xPixels, nWindows * nTrials)
    elif imageIn.ndim == 3:
        yPixels, xPixels, nWindows = imageIn.shape
        Y = imageIn.copy()
    else:
        raise ValueError("Input must be 3D or 4D array")
    
    # Convert to float32 and normalize
    Y = Y.astype(np.float32)
    Y = Y - np.min(Y)
    
    # Set parameters
    normcorre_binwidth = 1000
    print(f"normcorre_binwidth = {normcorre_binwidth}")
    time.sleep(2)
    
    # Set up options for non-rigid correction
    options_nonrigid = {
        'd1': Y.shape[0],
        'd2': Y.shape[1], 
        'grid_size': [32, 32],
        'mot_uf': 4,
        'bin_width': normcorre_binwidth,
        'max_shift': 15,
        'max_dev': 3,
        'us_fac': 50,
        'init_batch': normcorre_binwidth,
        'shifts_method': 'cubic'
    }
    
    # Perform motion correction
    if useTemplate:
        print("Using provided template for registration...")
        start_time = time.time()
        M1, shifts1, template1, options_rigid = normcorre(Y, options_nonrigid, template)
        print(f"Motion correction completed in {time.time() - start_time:.2f} seconds")
    else:
        print("Performing batch motion correction...")
        start_time = time.time()
        M1, shifts1, template1, options_nonrigid = normcorre_batch(Y, options_nonrigid)
        print(f"Motion correction completed in {time.time() - start_time:.2f} seconds")
    
    # Additional non-rigid correction for comparison if verbose
    if verbose:
        print("Also testing non-rigid motion correction (verbose is selected)")
        start_time = time.time()
        options_rigid = options_nonrigid.copy()  # Use same options for comparison
        M2, shifts2, template2, options_rigid = normcorre(Y, options_rigid)
        print(f"Additional correction completed in {time.time() - start_time:.2f} seconds")
    
    # Compute metrics and plot if verbose
    if verbose:
        nnY = np.quantile(Y, 0.005)
        mmY = np.quantile(Y, 0.995)
        
        cY, mY, vY = motion_metrics(Y, 10)
        cM1, mM1, vM1 = motion_metrics(M1, 10)
        cM2, mM2, vM2 = motion_metrics(M2, 10)
        T = len(cY)
        
        # Normalize frames for prettier plotting
        mYp = normalize_frame(mY, np.min(mY), np.max(mY))
        mM1p = normalize_frame(mM1, np.min(mM1), np.max(mM1))
        mM2p = normalize_frame(mM2, np.min(mM2), np.max(mM2))
        
        # Plot metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0,0].imshow(mYp, vmin=0, vmax=1, cmap='gray')
        axes[0,0].set_title('mean raw data', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(mM2p, vmin=0, vmax=1, cmap='gray')
        axes[0,1].set_title('mean rigid corrected', fontsize=14, fontweight='bold')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(mM1p, vmin=0, vmax=1, cmap='gray')
        axes[0,2].set_title('mean non-rigid corrected', fontsize=14, fontweight='bold')
        axes[0,2].axis('off')
        
        axes[1,0].plot(range(1, T+1), cY, label='raw data')
        axes[1,0].plot(range(1, T+1), cM1, label='non-rigid')
        axes[1,0].plot(range(1, T+1), cM2, label='rigid')
        axes[1,0].legend()
        axes[1,0].set_title('correlation coefficients', fontsize=14, fontweight='bold')
        
        axes[1,1].scatter(cY, cM1)
        min_val = 0.9 * min(np.min(cY), np.min(cM1))
        max_val = 1.05 * max(np.max(cY), np.max(cM1))
        axes[1,1].plot([min_val, max_val], [min_val, max_val], '--r')
        axes[1,1].set_xlabel('raw data', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('non-rigid corrected', fontsize=14, fontweight='bold')
        axes[1,1].set_aspect('equal')
        
        axes[1,2].scatter(cM1, cM2)
        axes[1,2].plot([min_val, max_val], [min_val, max_val], '--r')
        axes[1,2].set_xlabel('non-rigid corrected', fontsize=14, fontweight='bold')
        axes[1,2].set_ylabel('rigid corrected', fontsize=14, fontweight='bold')
        axes[1,2].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Plot shifts
        if 'shifts2' in locals():
            # Extract shifts data
            shifts_r = np.array([s['shifts'] for s in shifts1])
            shifts_nr = np.array([s['shifts'] for s in shifts2])
            
            if shifts_nr.ndim == 2:
                shifts_x = shifts_nr[:, 0:1].T
                shifts_y = shifts_nr[:, 1:2].T
            else:
                shifts_x = shifts_nr[:, 0].reshape(1, -1)
                shifts_y = shifts_nr[:, 1].reshape(1, -1)
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            axes[0].plot(range(1, T+1), cY, label='raw data')
            axes[0].plot(range(1, T+1), cM1, label='non-rigid')
            axes[0].plot(range(1, T+1), cM2, label='rigid')
            axes[0].legend()
            axes[0].set_title('correlation coefficients', fontsize=14, fontweight='bold')
            axes[0].set_xticks([])
            
            axes[1].plot(shifts_x.T)
            if shifts_r.ndim > 1:
                axes[1].plot(shifts_r[:, 0], '--k', linewidth=2)
            axes[1].set_title('displacements along x', fontsize=14, fontweight='bold')
            axes[1].set_xticks([])
            
            axes[2].plot(shifts_y.T)
            if shifts_r.ndim > 1:
                axes[2].plot(shifts_r[:, 1], '--k', linewidth=2)
            axes[2].set_title('displacements along y', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('timestep', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        
        # Create video if requested
        save_video = input('Would you like to create a video? (y/n): ')
        if save_video.lower() == 'y':
            create_motion_correction_video(Y, M1, T)
    
    # Return the result
    print('returning rigid-body motion corrected data')
    if imageIn.ndim == 4:
        imageOut = M1.reshape(yPixels, xPixels, nWindows, nTrials)
    else:
        imageOut = M1
    
    return imageOut, template1


def normalize_frame(frameIn, minAct, maxAct):
    """
    Normalize frame data between [0, 1] with nonlinear scaling
    """
    # Normalize data [0 1]
    frameOut = (frameIn - minAct) / (maxAct - minAct)
    # Nonlinear scale between [0 1]
    frameOut = np.power(frameOut, 1/4)  # 4th root
    return frameOut


# Motion correction functions using scikit-image and scipy

def normcorre(Y, options, template=None):
    """
    Motion correction using scikit-image phase correlation
    """
    print("Performing motion correction using scikit-image...")
    
    T = Y.shape[2]
    M = np.zeros_like(Y)
    shifts = []
    
    # Create template if not provided
    if template is None:
        template = np.mean(Y, axis=2)
    
    max_shift = options.get('max_shift', 15)
    
    for t in range(T):
        frame = Y[:, :, t]
        
        # Use phase correlation for registration
        shift_coords = registration.phase_cross_correlation(
            template, frame, 
            upsample_factor=options.get('us_fac', 50)//10
        )[0]
        
        # Limit shifts
        shift_coords = np.clip(shift_coords, -max_shift, max_shift)
        
        # Apply shift
        M[:, :, t] = shift(frame, shift_coords, mode='constant', cval=0)
        
        # Store shifts
        shifts.append({'shifts': shift_coords})
    
    return M, shifts, template, options


def normcorre_batch(Y, options):
    """
    Batch motion correction using template matching
    """
    print("Performing batch motion correction...")
    
    bin_width = options.get('bin_width', 200)
    T = Y.shape[2]
    
    # Create initial template from first bin
    template = np.mean(Y[:, :, :min(bin_width, T)], axis=2)
    
    # Perform motion correction
    M, shifts, template_final, _ = normcorre(Y, options, template)
    
    return M, shifts, template_final, options


def motion_metrics(Y, max_shift_th=10):
    """
    Compute motion correction metrics
    """
    T = Y.shape[2]
    
    # Compute mean frame
    mean_frame = np.mean(Y, axis=2)
    
    # Initialize arrays
    corr_coeffs = np.zeros(T)
    variance_vals = np.zeros(T)
    
    # Flatten mean frame for correlation computation
    mean_flat = mean_frame.flatten()
    mean_flat = mean_flat - np.mean(mean_flat)
    mean_norm = np.linalg.norm(mean_flat)
    
    for t in range(T):
        # Get current frame
        frame = Y[:, :, t]
        frame_flat = frame.flatten()
        frame_flat = frame_flat - np.mean(frame_flat)
        frame_norm = np.linalg.norm(frame_flat)
        
        # Compute correlation coefficient
        if frame_norm > 0 and mean_norm > 0:
            corr_coeffs[t] = np.dot(mean_flat, frame_flat) / (mean_norm * frame_norm)
        else:
            corr_coeffs[t] = 0
        
        # Compute variance
        variance_vals[t] = np.var(frame)
    
    return corr_coeffs, mean_frame, variance_vals


def create_motion_correction_video(Y, M1, T):
    """
    Create a video showing raw vs motion corrected data
    """
    plt.figure(figsize=(18, 7))
    
    minAct = np.min(Y)
    maxAct = np.max(Y)
    minActM2 = np.min(M1)
    maxActM2 = np.max(M1)
    
    print("Displaying motion correction comparison...")
    
    for t in range(T):
        plt.clf()
        
        rawFrame = normalize_frame(Y[:, :, t], minAct, maxAct)
        correctedFrame = normalize_frame(M1[:, :, t], minActM2, maxActM2)
        
        plt.subplot(1, 2, 1)
        plt.imshow(rawFrame, vmin=0, vmax=1, cmap='bone')
        plt.xlabel('Raw Data', fontsize=14, fontweight='bold')
        plt.title(f'Frame {t+1} out of {T}', fontweight='bold', fontsize=14)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1, 2, 2)
        plt.imshow(correctedFrame, vmin=0, vmax=1, cmap='bone')
        plt.xlabel('Motion Corrected', fontsize=14, fontweight='bold')
        plt.title(f'Frame {t+1} out of {T}', fontweight='bold', fontsize=14)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        plt.pause(0.02)
    
    # Ask for video saving
    save_dir = input("Enter directory path to save video (or press Enter to skip): ")
    if save_dir and os.path.exists(save_dir):
        print('Creating video...')
        
        # Create video using opencv
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = f'normcore_registration_{datetime.now().strftime("%Y-%m-%d")}.mp4'
        video_path = os.path.join(save_dir, video_name)
        
        height, width = Y.shape[:2]
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (width*2, height))
        
        for t in range(T):
            rawFrame = normalize_frame(Y[:, :, t], minAct, maxAct)
            correctedFrame = normalize_frame(M1[:, :, t], minActM2, maxActM2)
            
            # Convert to uint8 and stack horizontally
            raw_uint8 = (rawFrame * 255).astype(np.uint8)
            corr_uint8 = (correctedFrame * 255).astype(np.uint8)
            
            # Stack frames horizontally
            combined = np.hstack([raw_uint8, corr_uint8])
            combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            
            out.write(combined_color)
        
        out.release()
        print(f'Video saved to: {video_path}')
        print('Done.')
    else:
        print('Video creation skipped.')



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import time
# import os
# from datetime import datetime
# import cv2
# import caiman

# def normcorre_doppler(imageIn, verbose=False, template=None):
#     """
#     Python conversion of MATLAB normcorre_doppler function
    
#     Parameters:
#     imageIn: numpy array - input image data
#     verbose: bool - whether to show detailed output and plots
#     template: numpy array - optional template for motion correction
    
#     Returns:
#     imageOut: numpy array - motion corrected image data
#     template1: numpy array - template used for correction
#     """
    
#     # Check if template is provided
#     if template is not None:
#         useTemplate = True
#     else:
#         useTemplate = False
    
#     # Handle different input dimensions
#     if imageIn.ndim == 4:
#         yPixels, xPixels, nWindows, nTrials = imageIn.shape
#         Y = imageIn.reshape(yPixels, xPixels, nWindows * nTrials)
#     elif imageIn.ndim == 3:
#         yPixels, xPixels, nWindows = imageIn.shape
#         Y = imageIn.copy()
#     else:
#         raise ValueError("Input must be 3D or 4D array")
    
#     # Convert to float32 and normalize
#     Y = Y.astype(np.float32)
#     Y = Y - np.min(Y)
    
#     # Set parameters
#     normcorre_binwidth = 1000
#     print(f"normcorre_binwidth = {normcorre_binwidth}")
#     time.sleep(2)
    
#     # Set options for non-rigid motion correction
#     options_nonrigid = {
#         'd1': Y.shape[0],
#         'd2': Y.shape[1], 
#         'grid_size': [32, 32],
#         'mot_uf': 4,
#         'bin_width': normcorre_binwidth,
#         'max_shift': 15,
#         'max_dev': 3,
#         'us_fac': 50,
#         'init_batch': normcorre_binwidth,
#         'shifts_method': 'cubic'
#     }
    
#     # Perform motion correction
#     if useTemplate:
#         print("Performing motion correction with provided template...")
#         start_time = time.time()
#         # This would call: normcorre(Y, options_rigid, template)
#         M1, shifts1, template1, options_rigid = normcorre(Y, options_nonrigid, template)
#         print(f"Motion correction completed in {time.time() - start_time:.2f} seconds")
#     else:
#         print("Performing batch motion correction...")
#         start_time = time.time()
#         # This would call: normcorre_batch(Y, options_nonrigid)
#         M1, shifts1, template1, options_nonrigid = normcorre_batch(Y, options_nonrigid)
#         print(f"Motion correction completed in {time.time() - start_time:.2f} seconds")
    
#     # Non-rigid motion correction for comparison (if verbose)
#     if verbose:
#         print("Also testing rigid motion correction (verbose is selected)")
#         start_time = time.time()
#         # This would call: normcorre(Y, options_rigid)
#         M2, shifts2, template2, options_rigid = normcorre(Y, options_nonrigid)  # Note: using same options for simplicity
#         print(f"Rigid motion correction completed in {time.time() - start_time:.2f} seconds")
    
#     # Compute metrics and plot results if verbose
#     if verbose:
#         nnY = np.quantile(Y, 0.005)
#         mmY = np.quantile(Y, 0.995)
        
#         # Compute motion metrics
#         cY, mY, vY = motion_metrics(Y, 10)
#         cM1, mM1, vM1 = motion_metrics(M1, 10)
#         cM2, mM2, vM2 = motion_metrics(M2, 10)
#         T = len(cY)
        
#         # Normalize frames for prettier plotting
#         mYp = normalize_frame(mY, np.min(mY), np.max(mY))
#         mM1p = normalize_frame(mM1, np.min(mM1), np.max(mM1))
#         mM2p = normalize_frame(mM2, np.min(mM2), np.max(mM2))
        
#         # Plot metrics
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
#         # Mean images
#         im1 = axes[0, 0].imshow(mYp, vmin=0, vmax=1, cmap='gray')
#         axes[0, 0].set_title('Mean Raw Data', fontsize=14, fontweight='bold')
#         axes[0, 0].axis('off')
        
#         im2 = axes[0, 1].imshow(mM2p, vmin=0, vmax=1, cmap='gray')
#         axes[0, 1].set_title('Mean Rigid Corrected', fontsize=14, fontweight='bold')
#         axes[0, 1].axis('off')
        
#         im3 = axes[0, 2].imshow(mM1p, vmin=0, vmax=1, cmap='gray')
#         axes[0, 2].set_title('Mean Non-rigid Corrected', fontsize=14, fontweight='bold')
#         axes[0, 2].axis('off')
        
#         # Correlation coefficients
#         axes[1, 0].plot(range(1, T+1), cY, label='raw data')
#         axes[1, 0].plot(range(1, T+1), cM1, label='non-rigid')
#         axes[1, 0].plot(range(1, T+1), cM2, label='rigid')
#         axes[1, 0].legend()
#         axes[1, 0].set_title('Correlation Coefficients', fontsize=14, fontweight='bold')
        
#         # Scatter plots
#         min_val = 0.9 * min(np.min(cY), np.min(cM1))
#         max_val = 1.05 * max(np.max(cM1), np.max(cM2))
        
#         axes[1, 1].scatter(cY, cM1)
#         axes[1, 1].plot([min_val, max_val], [min_val, max_val], '--r')
#         axes[1, 1].set_xlabel('Raw Data', fontsize=14, fontweight='bold')
#         axes[1, 1].set_ylabel('Non-rigid Corrected', fontsize=14, fontweight='bold')
#         axes[1, 1].set_aspect('equal')
        
#         axes[1, 2].scatter(cM1, cM2)
#         axes[1, 2].plot([min_val, max_val], [min_val, max_val], '--r')
#         axes[1, 2].set_xlabel('Non-rigid Corrected', fontsize=14, fontweight='bold')
#         axes[1, 2].set_ylabel('Rigid Corrected', fontsize=14, fontweight='bold')
#         axes[1, 2].set_aspect('equal')
        
#         plt.tight_layout()
#         plt.show()
        
#         # Plot shifts
#         shifts_r = np.array([shift['shifts'] for shift in shifts1])
#         shifts_nr = np.array([shift['shifts'] for shift in shifts2])
        
#         if shifts_nr.ndim == 3:
#             shifts_nr = shifts_nr.reshape(-1, shifts_nr.shape[-1], T)
        
#         shifts_x = shifts_nr[:, 0, :].T
#         shifts_y = shifts_nr[:, 1, :].T
        
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
#         ax1.plot(range(1, T+1), cY, label='raw data')
#         ax1.plot(range(1, T+1), cM1, label='non-rigid')
#         ax1.plot(range(1, T+1), cM2, label='rigid')
#         ax1.legend()
#         ax1.set_title('Correlation Coefficients', fontsize=14, fontweight='bold')
#         ax1.set_xticks([])
        
#         ax2.plot(shifts_x)
#         ax2.plot(shifts_r[:, 0], '--k', linewidth=2)
#         ax2.set_title('Displacements along X', fontsize=14, fontweight='bold')
#         ax2.set_xticks([])
        
#         ax3.plot(shifts_y)
#         ax3.plot(shifts_r[:, 1], '--k', linewidth=2)
#         ax3.set_title('Displacements along Y', fontsize=14, fontweight='bold')
#         ax3.set_xlabel('Timestep', fontsize=14, fontweight='bold')
        
#         plt.tight_layout()
#         plt.show()
        
#         # Create video if requested
#         save_video = input('Would you like to create a video? (y/n): ')
#         if save_video.lower() == 'y':
#             create_motion_correction_video(Y, M1, T)
    
#     # Return results
#     print("Returning rigid-body motion corrected data")
#     if imageIn.ndim == 4:
#         imageOut = M1.reshape(yPixels, xPixels, nWindows, nTrials)
#     else:
#         imageOut = M1
    
#     return imageOut, template1


# def normalize_frame(frameIn, minAct, maxAct):
#     """
#     Normalize frame data [0, 1] with nonlinear scaling
#     """
#     # Normalize data [0, 1]
#     frameOut = (frameIn - minAct) / (maxAct - minAct)
#     # Nonlinear scale between [0, 1]
#     frameOut = np.power(frameOut, 1/4)
#     return frameOut


# def create_motion_correction_video(Y, M1, T):
#     """
#     Create a video showing motion correction results
#     """
#     plt.figure(figsize=(18, 7))
    
#     minAct = np.min(Y)
#     maxAct = np.max(Y)
#     minActM2 = np.min(M1)
#     maxActM2 = np.max(M1)
    
#     frames = []
    
#     for t in range(T):
#         plt.clf()
        
#         rawFrame = normalize_frame(Y[:, :, t], minAct, maxAct)
#         correctedFrame = normalize_frame(M1[:, :, t], minActM2, maxActM2)
        
#         plt.subplot(1, 2, 1)
#         plt.imshow(rawFrame, vmin=0, vmax=1, cmap='bone')
#         plt.xlabel('Raw Data', fontsize=14, fontweight='bold')
#         plt.title(f'Frame {t+1} out of {T}', fontweight='bold', fontsize=14)
#         plt.axis('equal')
#         plt.xticks([])
#         plt.yticks([])
        
#         plt.subplot(1, 2, 2)
#         plt.imshow(correctedFrame, vmin=0, vmax=1, cmap='bone')
#         plt.xlabel('Rigid Corrected', fontsize=14, fontweight='bold')
#         plt.title(f'Frame {t+1} out of {T}', fontweight='bold', fontsize=14)
#         plt.axis('equal')
#         plt.xticks([])
#         plt.yticks([])
        
#         plt.tight_layout()
#         plt.pause(0.01)
        
#         # Note: Video saving would require additional setup with matplotlib animation
#         # or opencv for actual video file creation
    
#     print("Video display completed")


# # Motion correction functions using existing Python packages

# def normcorre(Y, options, template=None):
#     """
#     Motion correction using CaImAn's normcorre implementation
    
#     Install CaImAn: pip install caiman
#     """
#     try:
#         from caiman.motion_correction import normcorre
#         from caiman.motion_correction import MotionCorrect
        
#         # Create MotionCorrect object
#         mc = MotionCorrect(Y, **options)
        
#         # Perform motion correction
#         if template is not None:
#             mc.template = template
        
#         mc.motion_correct()
        
#         return mc.mmap_file, mc.shifts, mc.template, options
        
#     except ImportError:
#         print("CaImAn not installed. Install with: pip install caiman")
#         print("Using JAX implementation as fallback...")
#         return normcorre_jax(Y, options, template)


# def normcorre_batch(Y, options):
#     """
#     Batch motion correction using CaImAn
#     """
#     try:
#         from caiman.motion_correction import motion_correct_batch_rigid
        
#         # Perform batch motion correction
#         fname_new, shifts, template, options = motion_correct_batch_rigid(
#             Y, **options
#         )
        
#         return fname_new, shifts, template, options
        
#     except ImportError:
#         print("CaImAn not installed. Install with: pip install caiman")
#         print("Using JAX implementation as fallback...")
#         return normcorre_batch_jax(Y, options)


# def motion_metrics(Y, max_shift_th=None):
#     """
#     Compute motion metrics using CaImAn
#     """
#     try:
#         from caiman.motion_correction import compute_metrics_motion_correction
        
#         # Compute motion correction metrics
#         corr_coeffs, mean_frame, variance_vals = compute_metrics_motion_correction(
#             Y, max_shift_th
#         )
        
#         return corr_coeffs, mean_frame, variance_vals
        
#     except ImportError:
#         print("CaImAn not installed. Using basic implementation...")
#         return motion_metrics_basic(Y, max_shift_th)


# # JAX implementation fallbacks
# def normcorre_jax(Y, options, template=None):
#     """
#     JAX-based motion correction using jnormcorre
    
#     Install jnormcorre: pip install jnormcorre
#     """
#     try:
#         import jnormcorre
        
#         # Convert options to jnormcorre format
#         jax_options = {
#             'max_shift': options.get('max_shift', 15),
#             'upsample_factor': options.get('us_fac', 50),
#             'bin_width': options.get('bin_width', 200)
#         }
        
#         # Perform motion correction
#         if template is not None:
#             M, shifts = jnormcorre.normcorre(Y, template=template, **jax_options)
#         else:
#             M, shifts = jnormcorre.normcorre(Y, **jax_options)
        
#         # Compute template
#         template_out = jnormcorre.compute_template(M)
        
#         return M, shifts, template_out, options
        
#     except ImportError:
#         print("jnormcorre not installed. Install with: pip install jnormcorre")
#         raise ImportError("No motion correction package available")


# def normcorre_batch_jax(Y, options):
#     """
#     JAX batch motion correction
#     """
#     try:
#         import jnormcorre
        
#         jax_options = {
#             'max_shift': options.get('max_shift', 15),
#             'upsample_factor': options.get('us_fac', 50),
#             'bin_width': options.get('bin_width', 200)
#         }
        
#         M, shifts = jnormcorre.normcorre_batch(Y, **jax_options)
#         template = jnormcorre.compute_template(M)
        
#         return M, shifts, template, options
        
#     except ImportError:
#         print("jnormcorre not installed. Install with: pip install jnormcorre")
#         raise ImportError("No motion correction package available")


# def motion_metrics_basic(Y, max_shift_th=None):
#     """
#     Basic motion metrics implementation
#     """
#     T = Y.shape[2]
#     mean_frame = np.mean(Y, axis=2)
    
#     corr_coeffs = np.zeros(T)
#     variance_vals = np.zeros(T)
    
#     mean_flat = mean_frame.flatten()
#     mean_flat = mean_flat - np.mean(mean_flat)
#     mean_norm = np.linalg.norm(mean_flat)
    
#     for t in range(T):
#         frame_flat = Y[:, :, t].flatten()
#         frame_flat = frame_flat - np.mean(frame_flat)
#         frame_norm = np.linalg.norm(frame_flat)
        
#         if frame_norm > 0 and mean_norm > 0:
#             corr_coeffs[t] = np.dot(mean_flat, frame_flat) / (mean_norm * frame_norm)
#         else:
#             corr_coeffs[t] = 0
        
#         variance_vals[t] = np.var(Y[:, :, t])
    
#     return corr_coeffs, mean_frame, variance_vals




