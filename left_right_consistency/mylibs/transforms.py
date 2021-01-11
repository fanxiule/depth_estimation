import numpy as np
import torch
import torchvision.transforms.functional as tF


class ToTensor(object):
    def __call__(self, left_img, right_img):
        left_im = tF.to_tensor(left_img)
        right_im = tF.to_tensor(right_img)
        return left_im, right_im


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, left_img, right_img):
        left_im = tF.five_crop(left_img, self.size)[4]
        right_im = tF.five_crop(right_img, self.size)[4]
        return left_im, right_im


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, left_img, right_img):
        if torch.rand(1) < self.p:
            left_im = tF.hflip(left_img)
            right_im = tF.hflip(right_img)
            return right_im, left_im  # after flipping, swap left and right images
        else:
            return left_img, right_img


class NormalizeImg(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):  # use mean and std dev from ImageNet
        self.mean = mean
        self.std = std

    def __call__(self, left_img, right_img):
        left_im = tF.normalize(left_img, self.mean, self.std)
        right_im = tF.normalize(right_img, self.mean, self.std)
        return left_im, right_im
