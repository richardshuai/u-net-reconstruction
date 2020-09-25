import tensorflow as tf
import numpy as np
from PIL import Image

# Input filepaths
comps_path = '/home/rshuai/research/u-net-reconstruction/data/PSFs/SVD_2_5um_PSF_5um_1_ds2_components_green_SubAvg.mat'
weights_path = '/home/rshuai/research/u-net-reconstruction/data/PSFs/SVD_2_5um_PSF_5um_1_ds2_weights_interp_green_SubAvg.mat'

## Parameters
rank = 24
ds_psf = 2
z_slices = 1

obj_dims = (648, 486)

# Read in Matlab v7.3 files with h5py.
# Stores output as comps and weights.
comps_file = h5py.File(comps_path, 'r')
for k, v in comps_file.items():
    comps = np.array(v)
    
print("Comps loaded in")

weights_file = h5py.File(weights_path, 'r')
for k, v in weights_file.items():
    weights = np.array(v)

print("Weights loaded in")
    
# Transpose axes to get both arrays into shape (r, z, y, x)
comps = comps.transpose(1, 0, 2, 3)
weights = weights.transpose(1, 0, 2, 3)

r_orig, z_orig, y_orig, x_orig = comps.shape

# Downsampling shape calculation. Must be flipped for use in PIL.Image.resize()
ds_shape = (np.array(comps.shape[2:]) / ds_psf).astype(int)

# Convert each image slice into PIL image and store in array of shape (r, z)
PIL_comps_image_array = [[Image.fromarray(comps[r, z]) for z in range(z_orig)] for r in range(r_orig)]
PIL_weights_image_array = [[Image.fromarray(weights[r, z])for z in range(z_orig)] for r in range(r_orig)]

# Downsample each PIL image in the array
PIL_comps_image_array = [[PIL_comps_image_array[r][z].resize(np.flip(ds_shape), resample=Image.BOX) 
                             for z in range(z_orig)] for r in range(r_orig)]
PIL_weights_image_array = [[PIL_weights_image_array[r][z].resize(np.flip(ds_shape), resample=Image.BOX) 
                             for z in range(z_orig)] for r in range(r_orig)]


print("Downsampling completed")

# Convert each PIL image back into numpy array, return to numpy array of shape (r, z, y, x)
comps = np.array([[np.array(PIL_comps_image_array[r][z]) for z in range(z_orig)] for r in range(r_orig)])
weights = np.array([[np.array(PIL_weights_image_array[r][z]) for z in range(z_orig)] for r in range(r_orig)])

# % Normalize weights to have maximum sum through rank of 1
_, _, y_new, x_new = weights.shape
weights_norm = np.max(np.sum(weights[:, :, y_new//2, x_new//2], axis=0, keepdims=True), axis=1)

# Other normalizations
weights = weights / weights_norm
comps = comps/(np.linalg.norm(comps.ravel()))


# PSF Preprocessing...
comps = comps[:rank]
weights = weights[:rank]
comps = comps[:, :z_slices]
weights = weights[:, :z_slices]

np.save('/home/rshuai/research/u-net-reconstruction/data/PSFs/processed/comps', comps)
np.save('/home/rshuai/research/u-net-reconstruction/data/PSFs/processed/weights', weights)

print("Done!")