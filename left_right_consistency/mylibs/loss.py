import numpy as np
import torch
from torch import nn


class Loss(object):
    def __init__(self, left_img, right_img, left_disp, right_disp, scale, use_GPU):
        self.left_img = left_img
        self.right_img = right_img
        self.left_disp = left_disp  # right image uses left disparity to generate left image
        self.right_disp = right_disp  # left image uses right disparity to generate right image
        self.scale = scale
        self.use_GPU = use_GPU
        size = list(left_img.size())
        self.batch_num = size[0]
        self.channel = size[1]
        self.height = size[2]
        self.width = size[3]
        self.num_pixels = self.batch_num * self.height * self.width

    def __call__(self):
        return self.cal_loss()

    def cal_loss(self):
        # weights
        alpha_ap = 3
        alpha_ds = 0.1 * self.scale
        alpha_lr = 0.75

        # loss
        appear_match_L = self.cal_appear_match(self.left_img, self.right_img, -self.left_disp)
        disp_smooth_L = self.cal_disp_smoothness(self.left_img, self.left_disp)
        disp_LR_L = self.cal_LR_consistency(self.left_disp, self.right_disp, True)

        appear_match_R = self.cal_appear_match(self.right_img, self.left_img, self.right_disp)
        disp_smooth_R = self.cal_disp_smoothness(self.right_img, self.right_disp)
        disp_LR_R = self.cal_LR_consistency(self.right_disp, self.left_disp, False)

        total_loss = alpha_ap * (appear_match_R + appear_match_L) + alpha_ds * (
                disp_smooth_R + disp_smooth_L) + alpha_lr * (disp_LR_R + disp_LR_L)
        return total_loss

    def cal_appear_match(self, orig_img, img_for_reconstruct, disparity):
        alpha = 0.85
        rec_img = self.reconstruct_fea(img_for_reconstruct,
                                       disparity)  # reconstruct image based on disparity using bilinear sampling

        img_diff = self.img_diff(orig_img, rec_img)
        SSIM_loss = self.SSIM(orig_img, rec_img)
        total_loss = alpha * SSIM_loss + (1 - alpha) * img_diff
        return total_loss / self.num_pixels

    def reconstruct_fea(self, features, disparity):
        # x_grid and y_grid define the column and row index to sample from the original feature
        x_grid = np.arange(self.width)
        x_grid = np.reshape(x_grid, [1, self.width])
        x_grid = np.tile(x_grid, [self.height, 1])
        x_grid_t = torch.from_numpy(x_grid)
        if self.use_GPU:
            x_grid_t = x_grid_t.cuda()
        x_grid_t = torch.reshape(x_grid_t, [1, self.height, self.width])
        x_grid_t = x_grid_t.repeat(self.batch_num, 1, 1)
        disparity_rec = torch.reshape(disparity, [self.batch_num, self.height, self.width])
        x_grid_t = x_grid_t + disparity_rec
        x_grid_t = torch.reshape(x_grid_t, [self.batch_num, self.height, self.width, 1])
        x_grid_t = 2 / (self.width - 1) * x_grid_t - 1  # normalized to [-1 1]

        y_grid = np.arange(self.height)
        y_grid = np.reshape(y_grid, [1, self.height])
        y_grid = np.tile(np.transpose(y_grid), [1, self.width])
        y_grid_t = torch.from_numpy(y_grid)
        if self.use_GPU:
            y_grid_t = y_grid_t.cuda()
        y_grid_t = torch.reshape(y_grid_t, [1, self.height, self.width])
        y_grid_t = y_grid_t.repeat(self.batch_num, 1, 1)
        y_grid_t = torch.reshape(y_grid_t, [self.batch_num, self.height, self.width, 1])
        y_grid_t = 2 / (self.height - 1) * y_grid_t - 1

        grid = torch.cat((x_grid_t, y_grid_t), 3)
        grid = torch.clip(grid, -1, 1)  # ensure all values in grid are between -1 to 1
        reconstructed_fea = nn.functional.grid_sample(features, grid, padding_mode='border', align_corners=False)
        return reconstructed_fea

    def img_diff(self, original_img, reconstructed_img):
        # L2 norm for img difference
        diff = original_img - reconstructed_img
        total_diff = self.cal_norm(diff)
        total_diff = torch.sum(total_diff)
        return total_diff

    def SSIM(self, original_image, reconstructed_img):
        C1 = 0.01 ** 2
        C2 = 0.02 ** 2

        # use average pooling to generate mu and sigma in SSIM
        param_generator = nn.AvgPool2d(kernel_size=(7, 7), stride=1)
        mu_orig = param_generator(original_image)
        mu_rec = param_generator(reconstructed_img)
        sigma_orig = param_generator(torch.square(original_image)) - torch.square(mu_orig)
        sigma_rec = param_generator(torch.square(reconstructed_img)) - torch.square(mu_rec)
        sigma_orig_rec = param_generator(torch.mul(original_image, reconstructed_img)) - torch.mul(mu_orig, mu_rec)

        SSIM_n1 = 2 * torch.mul(mu_orig, mu_rec) + C1
        SSIM_n2 = 2 * sigma_orig_rec + C2
        SSIM_d1 = torch.square(mu_orig) + torch.square(mu_rec) + C1
        SSIM_d2 = torch.square(sigma_orig) + torch.square(sigma_rec) + C2

        SSIM = torch.div(torch.mul(SSIM_n1, SSIM_n2), torch.mul(SSIM_d1, SSIM_d2))
        SSIM_loss = (1 - SSIM) / 2
        SSIM_loss = torch.clip(SSIM_loss, 0, 1)
        return torch.sum(SSIM_loss)

    def cal_disp_smoothness(self, image, disparity):
        im_x_grad = self.cal_gradient_x(image)
        im_y_grad = self.cal_gradient_y(image)
        disp_x_grad = self.cal_gradient_x(disparity)
        disp_y_grad = self.cal_gradient_y(disparity)

        im_x_grad = self.cal_norm(im_x_grad)
        im_y_grad = self.cal_norm(im_y_grad)
        im_x_grad = torch.reshape(im_x_grad, [self.batch_num, 1, self.height, self.width])
        im_y_grad = torch.reshape(im_y_grad, [self.batch_num, 1, self.height, self.width])
        im_x_grad = torch.exp(-im_x_grad)
        im_y_grad = torch.exp(-im_y_grad)
        smoothness_loss = torch.mul(torch.abs(disp_x_grad), im_x_grad) + torch.mul(torch.abs(disp_y_grad), im_y_grad)
        smoothness_loss = torch.sum(smoothness_loss)
        return smoothness_loss / self.num_pixels

    def cal_gradient_x(self, volume):
        volume_plus = torch.roll(volume, -1, 3)
        volume_minus = torch.roll(volume, 1, 3)
        gradient_x = (volume_plus - volume_minus) / 2
        return gradient_x

    def cal_gradient_y(self, volume):
        volume_plus = torch.roll(volume, -1, 2)
        volume_minus = torch.roll(volume, 1, 2)
        gradient_y = (volume_plus - volume_minus) / 2
        return gradient_y

    def cal_norm(self, volume):
        # L2 norm
        norm = torch.square(volume)
        norm = torch.sum(norm, dim=1)
        norm = torch.sqrt(norm + 0.0000000001)  # add a small number to avoid autograd causing nan
        return norm

    def cal_LR_consistency(self, orig_disp, disp_to_reconst, lr):
        # if lr = True, then we construct consistency loss with left_disp as the basis
        # if lr = False, then we construct consistency loss with right_disp as the basis
        if lr:
            reconstructed_disp = self.reconstruct_fea(disp_to_reconst, -orig_disp)
        else:
            reconstructed_disp = self.reconstruct_fea(disp_to_reconst, orig_disp)
        consistency_loss = orig_disp - reconstructed_disp
        consistency_loss = torch.abs(consistency_loss)
        consistency_loss = torch.sum(consistency_loss)
        return consistency_loss / self.num_pixels
