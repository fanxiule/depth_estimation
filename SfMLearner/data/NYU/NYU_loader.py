from __future__ import division
import json
import os
import numpy as np
from PIL import Image
from glob import glob


class NYU_loader(object):
    def __init__(self, dataset_dir, split='train', img_height=480, img_width=640, seq_length=3):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        assert seq_length % 2 != 0, 'seq_length must be odd'
        self.frames = self.collect_frames()
        self.num_frames = len(self.frames)
        if split == 'train':
            self.num_train = self.num_frames
        else:
            self.num_test = self.num_frames
        print('Total frames collected: %d' % self.num_frames)

    def collect_frames(self):
        img_dir = self.dataset_dir
        scene_list = os.listdir(img_dir)
        scene_list.sort()
        frames = []
        for scene in scene_list:
            img_files = glob(img_dir + scene + '/*.ppm')
            img_files.sort()
            for f in img_files:
                try:
                    # perform these two lines of code to identify truncated image
                    img = Image.open(f)
                    _ = img.resize((self.img_width, self.img_height))

                    frame_id = os.path.basename(f)
                    frame_id = scene + '/' + frame_id
                    frames.append(frame_id)
                except OSError:
                    os.remove(f)
                    pass
        return frames

    def load_image_sequence(self, tg_idx, seq_length):
        half_offset = int((seq_length - 1) / 2)
        image_seq = []
        zoom_x = zoom_y = 1
        for o in range(-half_offset, half_offset + 1):
            curr_local_id = tg_idx + o
            curr_image_file = self.dataset_dir + self.frames[curr_local_id]
            curr_img = Image.open(curr_image_file)
            np_raw_img = np.array(curr_img)
            raw_shape = np.copy(np_raw_img.shape)
            if o == 0:
                zoom_y = self.img_height/raw_shape[0]
                zoom_x = self.img_width/raw_shape[1]
            curr_img = curr_img.resize((self.img_width, self.img_height))
            np_curr_img = np.array(curr_img)
            image_seq.append(np_curr_img)
        return image_seq, zoom_x, zoom_y

    def scale_intrinsics(self, intrinsic, sx, sy):
        out = np.copy(intrinsic)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out

    def load_example(self, tgt_frame_id):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(tgt_frame_id, self.seq_length)
        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)

        example = {'intrinsics': intrinsics, 'image_seq': image_seq, 'folder_name': self.dataset_dir,
                   'file_name': self.frames[tgt_frame_id]}
        return example

    def get_train_example_with_idx(self, tgt_idx):
        # tgt_frame_id = self.frames[tgt_idx]
        example = self.load_example(tgt_idx)
        return example
