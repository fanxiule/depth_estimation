import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import cv2
from torchvision import transforms


class NYUEvalDataset(data.Dataset):
    def __init__(self, rgb_path, depth_path, height, width):
        super(NYUEvalDataset, self).__init__()
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.height = height
        self.width = width
        self.rgb_files = os.listdir(rgb_path)
        self.depth_files = os.listdir(depth_path)
        self.rgb_files.sort()
        self.depth_files.sort()
        self.total_img = len(self.rgb_files)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        rgb_file = self.rgb_files[index]
        depth_file = self.depth_files[index]
        rgb = cv2.imread(os.path.join(self.rgb_path, rgb_file), cv2.IMREAD_COLOR)
        depth = cv2.imread(os.path.join(self.depth_path, depth_file), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.resize(rgb, (self.width, self.height))
        rgb_torch = transforms.ToTensor()(rgb)
        return rgb_torch, depth

    def get_total_img_num(self):
        return self.total_img
