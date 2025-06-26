import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load MATLAB .mat files
dop_struct = sio.loadmat("20230228/dop_trials_struct_S187_R1.mat")
dop_thierri = dop_struct['dop_trials_struct']['iDopP'][0, 0]

dop_no = sio.loadmat("thierri20230228_32frames_lowres_interp_steps_3199_dop.mat")['iDOP']
dop_no = np.transpose(dop_no, (1, 2, 0))

dop_bin = sio.loadmat("thierri20230228_from_bin_dop.mat")['iDOP']
dop_bin = np.transpose(dop_bin, (1, 2, 0))

# Extract specific frames
id = 0  # Python is 0-indexed
x = dop_thierri[:, :, -1]   # Last frame
x_no = dop_no[:, :, id]
x_bin = dop_bin[:, :, id]

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(x, cmap='gray')
plt.title("iDOP")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(x_no, cmap='gray')
plt.title("NO")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(x_bin, cmap='gray')
plt.title("Bin")
plt.axis('off')

plt.tight_layout()
plt.show()
