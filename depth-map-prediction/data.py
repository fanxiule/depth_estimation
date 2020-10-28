from __future__ import print_function
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import torch.nn as nn
import torch

class TransposeDepthInput(object):
    def __call__(self, depth):
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth)
        depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
        depth = nn.functional.interpolate(depth, size = (55, 74), mode='bilinear', align_corners=False)
        depth = torch.log(depth)
        return depth[0]

rgb_data_transforms = transforms.Compose([
    transforms.Resize((228, 304)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

depth_data_transforms = transforms.Compose([
    TransposeDepthInput(),
])

input_for_plot_transforms = transforms.Compose([
    transforms.Resize((55, 74)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

class NYUDataset(Dataset):
    def calculate_mean(self, images):
        mean_image = np.mean(images, axis=0)
        return mean_image

    def __init__(self, filename, type, rgb_transform = None, depth_transform = None):
        f = h5py.File(filename, 'r')

        images_data = f['images'][0:1449]
        depths_data = f['depths'][0:1449]

        if type == "training":
            self.images = images_data[0:1024]
            self.depths = depths_data[0:1024]
        elif type == "validation":
            self.images = images_data[1024:1248]
            self.depths = depths_data[1024:1248]
        elif type == "test":
            self.images = images_data[1248:]
            self.depths = depths_data[1248:]

        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.mean_image = self.calculate_mean(images_data[0:1449])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.transpose((2, 1, 0))
        image = Image.fromarray(image)
        if self.rgb_transform:
            image = self.rgb_transform(image)

        depth = self.depths[idx]
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((2, 1, 0))
        if self.depth_transform:
            depth = self.depth_transform(depth)
        sample = {'image': image, 'depth': depth}
        return sample
