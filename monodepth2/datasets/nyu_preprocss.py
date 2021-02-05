import os
import argparse
import math
import random

parser = argparse.ArgumentParser(description="Preprocess NYU data")
parser.add_argument("--nyu_path",
                    type=str,
                    help="path to the NYU dataset",
                    default="/home/xfan/Documents/Datasets/NYU")
parser.add_argument("--interval",
                    type=int,
                    help="interval between adjacent input frames to the pose net",
                    default=5)
parser.add_argument("--split_path",
                    type=str,
                    help="path to the split txt file",
                    default="/home/xfan/Documents/Avidbots/Current_Approach/depth_estimation/monodepth2/splits/nyu")
options = parser.parse_args()

nyu_datapath = options.nyu_path
nyu_split_path = options.split_path
if not os.path.exists(nyu_split_path):
    os.makedirs(nyu_split_path)

train_file = os.path.join(nyu_split_path, "train_files.txt")
val_file = os.path.join(nyu_split_path, "val_files.txt")

scenarios = os.listdir(nyu_datapath)
scenarios.sort()
scene_list = []
for scenario in scenarios:
    scenes = os.listdir(os.path.join(nyu_datapath, scenario))
    scenes.sort()
    for scene in scenes:
        scene_list.append([scenario, scene])

for scene in scene_list:
    scene_path = os.path.join(scene[0], scene[1])
    complete_scene_path = os.path.join(nyu_datapath, scene_path)
    try:  # open the INDEX.txt file to identify files to preprocess
        txt_file = open(os.path.join(complete_scene_path, 'INDEX.txt'), 'r')
    except OSError as e:
        print("No text file found. Either the dataset is corrupted or it has been preprocessed")
        continue
    lines = txt_file.readlines()
    ppm_indicator = 0  # 0 no file read yet, 1 ppm read, 2 pgm read
    file_list = []  # to store index and all valid files
    ppm_file = None
    pgm_file = None
    ind = 0

    for line in lines:
        line = line.strip()  # remove the '\n' character
        if ppm_indicator == 0:  # read first file
            if line[-4:] == '.ppm':
                ppm_indicator = 1
                ppm_file = line
            elif line[-4:] == '.pgm':
                ppm_indicator = 2
                pgm_file = line

        else:  # if at least 1 file has been read previously
            if line[-4:] == '.ppm' and ppm_indicator == 2:
                ppm_file = line  # previously read file is pgm
                ppm_indicator = 1
            elif line[-4:] == '.pgm' and ppm_indicator == 1:
                pgm_file = line  # previously read file is ppm
                ppm_indicator = 2

        if ppm_file is not None and pgm_file is not None:
            file_list.append([ind, ppm_file, pgm_file])
            ind += 1
            ppm_file = None
            pgm_file = None

    # total number of valid samples
    max_num = len(file_list)

    # rename files to 1.pgm, 1.ppm, 2.pgm, 2.ppm, ... for easier manipulation during training
    for files in file_list:
        ind, ppm_file, pgm_file = files
        old_ppm_file = os.path.join(complete_scene_path, ppm_file)
        old_pgm_file = os.path.join(complete_scene_path, pgm_file)
        new_ppm_file = os.path.join(complete_scene_path, '%d.ppm' % ind)
        new_pgm_file = os.path.join(complete_scene_path, '%d.pgm' % ind)
        os.rename(old_ppm_file, new_ppm_file)
        os.rename(old_pgm_file, new_pgm_file)

    # delete unnecessary files, i.e. the ones that haven't been renamed
    file_list = os.listdir(complete_scene_path)
    for file in file_list:
        file_name_ext = os.path.splitext(file)
        file_name_lower = file_name_ext[0].lower()
        if file_name_lower.islower():
            os.remove(os.path.join(complete_scene_path, file))

    min_file_name = 0
    max_file_name = max_num-1
    min_allow_file_name = min_file_name + options.interval
    max_allow_file_name = max_file_name - options.interval
    file_name_range = list(range(min_allow_file_name, max_allow_file_name + 1))
    random.shuffle(file_name_range)
    split = math.floor(0.8 * len(file_name_range))
    train_split = file_name_range[:split]
    val_split = file_name_range[split:]
    # write training files and validation files to txt
    with open(train_file, 'a') as f:
        for t_file in train_split:
            f.write("%s %d\n" % (scene_path, t_file))
    with open(val_file, 'a') as f:
        for v_file in val_split:
            f.write("%s %d\n" % (scene_path, v_file))
