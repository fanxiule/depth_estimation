import os
import random
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class NYUDataset(data.Dataset):
    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales, is_train=False, img_ext='.ppm'):
        super(NYUDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.full_res_shape = (640, 480)
        self.K = np.array([[519 / self.full_res_shape[0], 0, 326 / self.full_res_shape[0], 0],
                           [0, 519 / self.full_res_shape[1], 254 / self.full_res_shape[1], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # follow the same augmentation protocol as the KITTI implementation
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)

        self.load_depth = self.check_depth()

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_filename = os.path.join(self.data_path, scene_name, "%d.pgm" % frame_index)
        return os.path.isfile(depth_filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1])

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

    def get_image_path(self, folder, frame_index):
        f_str = "%d%s" % (frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def get_depth(self, folder, frame_index, do_flip):
        depth_path = self.get_depth_path(folder, frame_index)
        depth = Image.open(depth_path)
        # convert depth from 16-bit pgm image to absolute depth
        depth = np.array(depth)
        depth = depth / 65535 * 10.0
        # depth = 351.3 / (1092.5 - depth)
        # depth = np.clip(depth, 0, 10.0)

        if do_flip:
            depth = np.fliplr(depth)
        return depth

    def get_depth_path(self, folder, frame_index):
        f_str = "%d%s" % (frame_index, ".pgm")
        depth_path = os.path.join(self.data_path, folder, f_str)
        return depth_path
