import numpy as np
import argparse
import h5py
import urllib
import os
import cv2

parser = argparse.ArgumentParser(description="NYU labeled data options")
parser.add_argument("--save_path",
                    type=str,
                    help="Path to save the converted img and depth data",
                    default="/home/xfan/Documents/Datasets/NYU_labeled")
parser.add_argument("--mat_filepath",
                    type=str,
                    help="Where is the .mat file for labeled data saved",
                    default="/home/xfan/Documents/Avidbots/Current_Approach/depth_estimation/monodepth2/datasets/")
parser.add_argument("--data_num",
                    type=int,
                    help="How many pairs of img + depth data to be converted and saved",
                    choices=range(0, 1450),
                    default=1449)
options = parser.parse_args()

# make path if not exist
path = options.save_path  # path to save test images
depth_path = os.path.join(path, "depth")
rgb_path = os.path.join(path, "rgb")
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(depth_path):
    os.makedirs(depth_path)
if not os.path.exists(rgb_path):
    os.makedirs(rgb_path)
# remove all existing images in the path
rgb_img = os.listdir(rgb_path)
dep_img = os.listdir(depth_path)
for i in rgb_img:
    i_full = os.path.join(rgb_path, i)
    os.remove(i_full)
for i in dep_img:
    i_full = os.path.join(depth_path, i)
    os.remove(i_full)

# download and read the .mat file
mat_filename = "nyu_depth_v2_labeled.mat"
mat_path = os.path.join(options.mat_filepath, mat_filename)
if not os.path.exists(mat_path):
    print("Downloading data...")
    urllib.request.urlretrieve("http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
                               filename=mat_filename)
    print("Data downloaded")

print("Reading .mat file")
f = h5py.File(mat_path, 'r')
image_data = f['images'][0:1449]
depth_data = f['depths'][0:1449]

# Convert data from .mat to img files
img_indices = np.random.choice(1449, options.data_num, False)
i = 1
print("Start data conversion")
for ind in img_indices:
    print("Current progress: %d/%d" % (i, options.data_num))
    rgb_img = np.copy(image_data[ind, :, :, :])
    dep_img = np.copy(depth_data[ind, :, :])
    rgb_img = np.transpose(rgb_img, (2, 1, 0))
    dep_img = 255.0 * np.transpose(dep_img) / 10.0
    dep_img = dep_img.astype(int)
    rgb_dir = os.path.join(rgb_path, str(ind) + '.jpg')
    dep_dir = os.path.join(depth_path, str(ind) + '.png')
    cv2.imwrite(rgb_dir, rgb_img)
    cv2.imwrite(dep_dir, dep_img)
    i += 1
