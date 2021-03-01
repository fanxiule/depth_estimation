import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import networks
import datasets
from utils import download_model_if_doesnt_exist

parser = argparse.ArgumentParser(description="Monodepth2 evaluation")
parser.add_argument("--data_path",
                    type=str,
                    help="Data path to hold the rgb and ground truth depth maps for the data to be evaluated",
                    default="/home/xfan/Documents/Datasets/NYU_labeled")
parser.add_argument("--model_path",
                    type=str,
                    help="File path where the model is saved",
                    default="/home/xfan/Documents/Avidbots/Current_Approach/depth_estimation/monodepth2/models")
parser.add_argument("--orig_flag",
                    help="A flag to indicate if the original model which has been trained on KITTI should be evaluated",
                    action="store_false")
parser.add_argument("--tuned_flag",
                    help="A flag to indicate if the fine tuned model should be evaluated",
                    action="store_false")
parser.add_argument("--tuned_model_name",
                    type=str,
                    help="What is the model name for the fine tuned model",
                    default="nyu_fine_tuned_lr1e-7")
parser.add_argument("--tuned_models",
                    type=list,
                    help="The model number/epoch to be evaluated within the available fine tuned models",
                    default=[0, 4, 14])
parser.add_argument("--use_GPU",
                    help="A flag to indicate if GPU should be used for evaluation",
                    action="store_false")
parser.add_argument("--batch_size",
                    type=int,
                    help="Batch size for running the evaluation",
                    default=6)
options = parser.parse_args()

IMG_HEIGHT = 480
IMG_WIDTH = 640

if options.use_GPU:
    device = 'cuda'
else:
    device = 'cpu'


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = np.sum(np.mean((thresh < 1.25), axis=(1, 2)))
    a2 = np.sum(np.mean((thresh < 1.25 ** 2), axis=(1, 2)))
    a3 = np.sum(np.mean((thresh < 1.25 ** 3), axis=(1, 2)))

    rmse = (gt - pred) ** 2
    rmse = np.sum(np.sqrt(np.mean(rmse, axis=(1, 2))))

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sum(np.sqrt(np.mean(rmse_log, axis=(1, 2))))

    abs_rel = np.abs(gt - pred) / gt
    abs_rel = np.sum(np.mean(abs_rel, axis=(1, 2)))

    sq_rel = ((gt - pred) ** 2) / gt
    sq_rel = np.sum(np.mean(sq_rel, axis=(1, 2)))

    return a1, a2, a3, rmse, rmse_log, abs_rel, sq_rel


def build_model(enc_path, dec_path):  # build a model using parameters from previous training
    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(enc_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict_dec = torch.load(dec_path, map_location='cpu')
    decoder.load_state_dict(loaded_dict_dec)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    img_height = loaded_dict_enc['height']
    img_width = loaded_dict_enc['width']
    return encoder, decoder, img_height, img_width


def slice_img(img, pixel_num=10, rgb=False):  # remove depth/rgb image edges since they are just white edges
    # when rgb is False, the input is a depth map in [batch_num, height, width]
    # when rgb is True, the input is an RGB image in [height, width, channel]
    shape = np.shape(img)
    if len(shape) == 2:
        sliced_img = img[pixel_num:-pixel_num, pixel_num:-pixel_num]
    else:
        if not rgb:
            sliced_img = img[:, pixel_num:-pixel_num, pixel_num:-pixel_num]  # remove pixel_num pixels from the borders
        else:
            sliced_img = img[pixel_num:-pixel_num, pixel_num:-pixel_num, :]
    return sliced_img


def evaluate(encoder, decoder, rgb_input, gt_depth):
    with torch.no_grad():
        rgb_input = rgb_input.to(device)
        features = encoder(rgb_input)
        outputs = decoder(features)
        pred_disp = outputs[("disp", 0)]
        pred_disp = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))(pred_disp)

        # calculate losses
        gt_depth = gt_depth / 255.0 * 10.0  # convert depth images from 256 bit intensity to depth
        pred_depth = 1 / pred_disp
        gt_depth = gt_depth.squeeze().to('cpu').numpy()
        pred_depth = pred_depth.squeeze().to('cpu').numpy()
        # remove the borders
        gt_depth = slice_img(gt_depth)
        pred_depth = slice_img(pred_depth)
        # scale the median of pred_depth to match gt_depth
        ratio = np.median(gt_depth, axis=(1, 2)) / np.median(pred_depth, axis=[1, 2])
        shape = np.shape(gt_depth)
        ratio_sz = ratio.size
        # pred_depth = np.clip(pred_depth, 0.001, 10.0)
        ratio = np.reshape(ratio, (ratio_sz, 1, 1))
        ratio = np.repeat(ratio, shape[1], axis=1)
        ratio = np.repeat(ratio, shape[2], axis=2)
        pred_depth = np.multiply(pred_depth, ratio)
        gt_depth = np.clip(gt_depth, 0.001, 10.0)  # limit the depth within the limit 0.001 m to 10 m
        pred_depth = np.clip(pred_depth, 0.001, 10.0)
        losses = compute_errors(gt_depth, pred_depth)
        return pred_depth, losses


def get_single_sample(dataset, index):
    rgb, gt_depth = dataset.__getitem__(index)
    rgb = rgb.unsqueeze(0)
    gt_depth = gt_depth / 255.0 * 10.0
    return rgb, gt_depth


def pred_single_img(encoder, decoder, rgb_input, gt_depth):
    with torch.no_grad():
        rgb_input = rgb_input.to(device)
        features = encoder(rgb_input)
        outputs = decoder(features)
        pred_disp = outputs[("disp", 0)]
        pred_disp = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))(pred_disp)

        # calculate losses
        pred_depth = 1 / pred_disp
        pred_depth = pred_depth.squeeze().to('cpu').numpy()
        # remove the borders
        gt_depth = slice_img(gt_depth)
        pred_depth = slice_img(pred_depth)
        # scale the median of pred_depth to match gt_depth
        # pred_depth = np.clip(pred_depth, 0.001, 10.0)
        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth = np.multiply(pred_depth, ratio)
        pred_depth = np.clip(pred_depth, 0.001, 10.0)
        return pred_depth


class LossStack:
    def __init__(self):
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.rmse = 0
        self.rmse_log = 0
        self.abs_rel = 0
        self.sq_rel = 0

    def add_loss(self, losses):
        self.a1 += losses[0]
        self.a2 += losses[1]
        self.a3 += losses[2]
        self.rmse += losses[3]
        self.rmse_log += losses[4]
        self.abs_rel += losses[5]
        self.sq_rel += losses[6]

    def get_all_loss(self):
        return [self.a1, self.a2, self.a3, self.rmse, self.rmse_log, self.abs_rel, self.sq_rel]


def main():
    encoder_list = {}
    decoder_list = {}
    img_H_list = {}
    img_W_list = {}

    if options.orig_flag:  # load the original monodepth2 model
        model_name = "mono_640x192"
        download_model_if_doesnt_exist(model_name)
        print("Loading original monodepth2 model...")
        encoder_path = os.path.join(options.model_path, model_name, "encoder.pth")
        decoder_path = os.path.join(options.model_path, model_name, "depth.pth")
        encoder, decoder, height, width = build_model(encoder_path, decoder_path)
        encoder_list['mono'] = encoder
        decoder_list['mono'] = decoder
        img_H_list['mono'] = height
        img_W_list['mono'] = width

    if options.tuned_flag:  # load the fine tuned model
        model_folder = options.tuned_model_name + "/models"
        tuned_model_path = os.path.join(options.model_path, model_folder)
        for i in options.tuned_models:
            print("Loading fine tuned model from epoch %s" % str(i + 1))
            model_name = "weights_%s" % str(i)
            encoder_path = os.path.join(tuned_model_path, model_name, "encoder.pth")
            decoder_path = os.path.join(tuned_model_path, model_name, "depth.pth")
            encoder, decoder, height, width = build_model(encoder_path, decoder_path)
            encoder_list['tuned_%d' % i] = encoder
            decoder_list['tuned_%d' % i] = decoder
            img_H_list['tuned_%d' % i] = height
            img_W_list['tuned_%d' % i] = width

    # build datasets and dataloaders for evaluation images for evaluation
    rgb_path = os.path.join(options.data_path, 'rgb')
    depth_path = os.path.join(options.data_path, 'depth')
    dataloaders = {}
    if options.orig_flag:  # build two datasets, one for original monodepth2 model and one for fine tuned model
        mono_dataset = datasets.NYUEvalDataset(rgb_path, depth_path, img_H_list['mono'], img_W_list['mono'])
        mono_loader = DataLoader(mono_dataset, options.batch_size, True)
        dataloaders['mono'] = mono_loader
        total_imgs = mono_dataset.get_total_img_num()
    if options.tuned_flag:
        tuned_dataset = datasets.NYUEvalDataset(rgb_path, depth_path, img_H_list['tuned_%d' % options.tuned_models[0]],
                                                img_W_list['tuned_%d' % options.tuned_models[0]])
        tuned_loader = DataLoader(tuned_dataset, options.batch_size, True)
        dataloaders['tuned'] = tuned_loader
        total_imgs = mono_dataset.get_total_img_num()

    # evaluation
    '''
    loss_list = {}
    if options.orig_flag:
        loss_list['mono'] = LossStack()
        print("Evaluating pretrained monodepth2 model...")
        for batch_ind, inputs in enumerate(mono_loader):
            rgb_input, gt_depth = inputs
            _, losses = evaluate(encoder_list['mono'], decoder_list['mono'], rgb_input, gt_depth)
            loss_list['mono'].add_loss(losses)

    if options.tuned_flag:
        for i in options.tuned_models:
            loss_list['tuned_%d' % i] = LossStack()
            print("Evaluating model fine tuned with %d epochs" % (i + 1))
            for batch_ind, inputs in enumerate(tuned_loader):
                rgb_input, gt_depth = inputs
                _, losses = evaluate(encoder_list['tuned_%d' % i], decoder_list['tuned_%d' % i], rgb_input, gt_depth)
                loss_list['tuned_%d' % i].add_loss(losses)

    # print loss
    for model, loss_stack in loss_list.items():
        total_losses = loss_stack.get_all_loss()
        avg_losses = [loss / total_imgs for loss in total_losses]
        print("%s losses: a1=%.4f a2=%.4f a3=%.4f rmse=%.4f rmse_log=%.4f abs_rel=%.4f sq_rel=%.4f" % (
            model, avg_losses[0], avg_losses[1], avg_losses[2], avg_losses[3], avg_losses[4], avg_losses[5],
            avg_losses[6]))
    '''
    # generate a sample output
    # img_ind = random.randint(0, total_imgs)
    img_ind = 283
    pred_depth_list = {}
    if options.orig_flag:
        rgb_input, gt_depth = get_single_sample(mono_dataset, img_ind)
        pred_depth = pred_single_img(encoder_list['mono'], decoder_list['mono'], rgb_input, gt_depth)
        pred_depth_list['mono'] = pred_depth
    if options.tuned_flag:
        rgb_input, gt_depth = get_single_sample(tuned_dataset, img_ind)
        for i in options.tuned_models:
            pred_depth = pred_single_img(encoder_list['tuned_%d' % i], decoder_list['tuned_%d' % i], rgb_input,
                                         gt_depth)
            pred_depth_list['tuned_%d' % i] = pred_depth

    rgb_input = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))(rgb_input)
    rgb_input_np = rgb_input.squeeze().cpu().numpy()
    rgb_input_np = np.moveaxis(rgb_input_np, 0, -1)
    rgb_input_np = (255 * rgb_input_np).astype(int)
    # rgb_input_np = slice_img(rgb_input_np, True)

    # plotting
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(rgb_input_np)
    axs[0, 0].set_title('RGB Input')
    axs[0, 1].imshow(gt_depth, cmap='viridis', vmin=0, vmax=10)
    axs[0, 1].set_title('GT Depth')
    if options.orig_flag:
        axs[0, 2].imshow(pred_depth_list['mono'], cmap='viridis', vmin=0, vmax=10)
        axs[0, 2].set_title('Pred Depth - Orig Mono2')
    if options.tuned_flag:
        num_models = 0
        for i in options.tuned_models:
            if num_models > 2:
                break
            axs[1, num_models].imshow(pred_depth_list['tuned_%d' % i], cmap='viridis', vmin=0, vmax=10)
            axs[1, num_models].set_title('Tuned for %d epochs' % (i + 1))
            num_models += 1
    plt.show()


if __name__ == "__main__":
    main()
