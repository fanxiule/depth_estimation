import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import glob

# remove existing test images
path = './test_samples/'  # path to save test images
files = glob.glob(path + '*')
for f in files:
    os.remove(f)

# read the complete test dataset
filename = 'nyu_depth_v2_labeled.mat'
f = h5py.File(filename, 'r')
image_data = f['images'][0:1449]
depth_data = f['depths'][0:1449]

# sample and save images
img_indices = np.random.choice(1449, 30, False)
for ind in img_indices:
    rgb_img = np.copy(image_data[ind, :, :, :])
    dep_img = np.copy(depth_data[ind, :, :])
    rgb_img = np.transpose(rgb_img, (2, 1, 0))
    dep_img = np.transpose(dep_img)
    rgb_dir = path + str(ind) + '_rgb.jpg'
    dep_dir = path + str(ind) + '_dep.txt'
    plt.imsave(rgb_dir, rgb_img)
    np.savetxt(dep_dir, dep_img, delimiter=', ')
