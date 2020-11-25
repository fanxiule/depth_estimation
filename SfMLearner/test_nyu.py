from __future__ import division
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from SfMLearner_NYU import SfMLearner
import glob
import time

matplotlib.use('Agg')


def cal_rmse(pred_dep, gt_dep):
    diff = gt_dep - pred_dep
    diff = np.square(diff)
    diff = np.sum(diff)
    sz = pred_dep.size
    rmse = np.sqrt(diff / sz)
    return rmse


def cal_delta(pred_dep, gt_dep, threshold):
    ratio1 = np.divide(pred_dep, gt_dep)
    ratio2 = np.divide(gt_dep, pred_dep)
    max_ratio = np.maximum(ratio1, ratio2)
    passed_ratio = max_ratio < threshold
    num_passed_pix = np.count_nonzero(passed_ratio)
    sz = pred_dep.size
    return num_passed_pix / sz


img_height = 480
img_width = 640
max_depth = 10

ckpt_file = 'checkpoints/NYU_fresh/model.latest'
# ckpt_file = 'models/model-190532' # KITTI model

# get list of test files
test_dir = 'nyu_test/test_samples/'
img_files = glob.glob(test_dir + '*.jpg')
dep_files = glob.glob(test_dir + '*.txt')
img_files.sort()
dep_files.sort()

# open and store images
rgb_img = np.empty((1, 1, img_height, img_width, 3))  # weird 5D shape so that it agrees with the input of the model
dep_img = np.empty((1, img_height, img_width))
for ind, _ in enumerate(img_files):
    img = plt.imread(img_files[ind])
    img = np.reshape(img, (1, 1, img_height, img_width, 3))

    dep = np.loadtxt(dep_files[ind], delimiter=', ')
    dep = np.reshape(dep, (1, img_height, img_width))

    if ind == 0:
        rgb_img = img
        dep_img = dep
    else:
        rgb_img = np.concatenate((rgb_img, img), axis=0)
        dep_img = np.concatenate((dep_img, dep), axis=0)

# define the network
sfm = SfMLearner()
sfm.setup_inference(img_height,
                    img_width,
                    mode='depth')

# inference
pred_dep = np.empty((1, img_height, img_width))
saver = tf.train.Saver([var for var in tf.model_variables()])
start_time = time.time()
with tf.Session() as sess:
    saver.restore(sess, ckpt_file)
    for ind, _ in enumerate(img_files):
        pred = sfm.inference(rgb_img[ind, :, :, :, :], sess, mode='depth')
        depth = pred['depth']
        if ind == 0:
            pred_dep = depth
        else:
            pred_dep = np.concatenate((pred_dep, depth), axis=0)

# calculate fps
end_time = time.time()
fps = len(img_files) / (end_time - start_time)
print('Frame rate: %.6f' % fps)

# scale the predicted depth
pred_dep = pred_dep[:, :, :, 0]
pred_dep = np.clip(pred_dep, 0, max_depth)
for ind in np.arange(pred_dep.shape[0]):
    gt_dep_frame = dep_img[ind, :, :]
    pred_dep_frame = pred_dep[ind, :, :]
    scale = np.median(gt_dep_frame) / np.median(pred_dep_frame)
    pred_dep[ind, :, :] = scale * pred_dep_frame

# calculate error
rmse = cal_rmse(pred_dep, dep_img)
delta1 = cal_delta(pred_dep, dep_img, 1.25)
delta2 = cal_delta(pred_dep, dep_img, 1.25 * 1.25)
delta3 = cal_delta(pred_dep, dep_img, 1.25 * 1.25 * 1.25)
print('RMSE: %.6f, delta<1.25: %.6f, delta<1.25^2: %.6f, delta<1.25^3: %.6f' % (rmse, delta1, delta2, delta3))

# save results
result_path = 'nyu_test/test_results/'
files = glob.glob(result_path + '*')
for f in files:
    os.remove(f)

img_ind = [f.split('/')[-1] for f in img_files]
img_ind = [f.split('_')[0] for f in img_ind]

for ind, _ in enumerate(img_files):
    fig = plt.figure(ind+1, figsize=(10, 15))
    plt.subplot(3, 1, 1)
    plt.imshow(rgb_img[ind, 0, :, :, :])
    plt.subplot(3, 1, 2)
    plt.imshow(pred_dep[ind, :, :])
    plt.subplot(3, 1, 3)
    plt.imshow(dep_img[ind, :, :])
    fig_path = result_path + img_ind[ind] + '.jpg'
    plt.savefig(fig_path)
    plt.close(fig)
