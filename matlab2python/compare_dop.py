import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat files
x = sio.loadmat("../no_data_from_bahareh/binfiles_mat/thierriS164_binfiles.mat")
dop_from_bahareh_bin = x['iDOP']

y = sio.loadmat("../dop_trials_struct_S164_R1_NoRegister_NoDetrend.mat")
dop_from_thierri = y['dop_trials_struct']['iDopP'][0, 0]

z = sio.loadmat("../no_data_from_bahareh/finetunedS164_thierriS164_32frames_lowres_interp_steps_799_dop_offset0.mat")
dop_no = z['iDOP']

# Function to visualize frames at different indices
def plot_frames(index):
    plt.figure(figsize=(12, 4))

    # Frame from Bahareh bin
    plt.subplot(1, 3, 1)
    plt.imshow(dop_from_bahareh_bin[index, :, :], cmap='gray')
    plt.title(f"Bahareh frame {index+1}")
    plt.axis('off')

    # Frame from Thierri
    plt.subplot(1, 3, 2)
    plt.imshow(dop_from_thierri[:, :, index], cmap='gray')
    plt.title("Thierri")
    plt.axis('off')

    # Frame from Bahareh NOfUS
    plt.subplot(1, 3, 3)
    plt.imshow(dop_no[index, :, :], cmap='gray')
    plt.title("Bahareh NOfUS")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Plot for frames 1, 500, and 9000
plot_frames(0)     # MATLAB's i=1 → Python's index 0
plot_frames(499)   # i=500
plot_frames(8999)  # i=9000
