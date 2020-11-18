from __future__ import division
import argparse
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="/home/xiule/Downloads/ppm/", help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, default="NYU_depth", help="which dataset needs to be prepared")
parser.add_argument("--dump_root", type=str,
                    default="/home/xiule/Programming/p_workspace/depth_esitmation_lit_reviews/depth_estimation/SfMLearner/data/NYU/formatted_data/",
                    help="Where to dump the data")
parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=480, help="image height")
parser.add_argument("--img_width", type=int, default=640, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()


def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def dump_example(n, args):
    # if n % 2000 == 0:
    print('Progress %d/%d....' % (n, data_loader.num_train - 2))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    folder_name = example['file_name'].split('/')[0]
    dump_dir = args.dump_root + folder_name
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    file_name = example['file_name'].split('/')[1]
    file_name = file_name.split('.ppm')[0]
    dump_img_file = dump_dir + '/%s.jpg' % file_name
    plt.imsave(dump_img_file, image_seq)
    dump_cam_file = dump_dir + '/%s_cam.txt' % file_name
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))


def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    from NYU.NYU_loader import NYU_loader
    data_loader = NYU_loader(args.dataset_dir,
                             img_height=args.img_height,
                             img_width=args.img_width,
                             seq_length=args.seq_length)

    for n in range(1, data_loader.num_train - 1):
        dump_example(n, args)

    # Split into train/val
    np.random.seed(8964)  ##
    subfolders = os.listdir(args.dump_root)
    subfolders.sort()

    for sub in subfolders:
        # remove first and last images in each scene because they are usually mixed with images from a different scene
        img_files = glob(args.dump_root + sub + '/*.jpg')
        cam_files = glob(args.dump_root + sub + '/*.txt')
        img_files.sort()
        cam_files.sort()
        first_img_files = img_files[0]
        last_img_files = img_files[-1]
        first_cam_files = cam_files[0]
        last_cam_files = cam_files[-1]
        os.remove(first_img_files)
        os.remove(last_img_files)
        os.remove(first_cam_files)
        os.remove(last_cam_files)

    with open(args.dump_root + 'train.txt', 'w') as tf:
        with open(args.dump_root + 'val.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.jpg')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))


main()
