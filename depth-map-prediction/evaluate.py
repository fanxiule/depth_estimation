import matplotlib
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from model import coarseNet, fineNet
import pdb
import numpy as np

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='PyTorch depth prediction evaluation script')
parser.add_argument('--model_folder', type=str,
                    default='/home/xiule/Programming/p_workspace/depth_esitmation_lit_reviews/depth_estimation/depth-map-prediction/',
                    metavar='F',
                    help='In which folder have you saved the model')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model_no', type=int, default=500, metavar='N',
                    help='Which model no to evaluate (default: 1(first model))')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')

args = parser.parse_args()

output_height = 55
output_width = 74

coarse_state_dict = torch.load(args.model_folder + "models" + "/coarse_model_" + str(args.model_no) + ".pth")
fine_state_dict = torch.load(args.model_folder + "models" + "/fine_model_" + str(args.model_no) + ".pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coarse_model = coarseNet()
fine_model = fineNet()
coarse_model.to(device)
fine_model.to(device)

coarse_model.load_state_dict(coarse_state_dict)
fine_model.load_state_dict(fine_state_dict)
coarse_model.eval()
fine_model.eval()

dtype = torch.cuda.FloatTensor

from data import NYUDataset, input_for_plot_transforms, rgb_data_transforms, depth_data_transforms

test_loader = torch.utils.data.DataLoader(NYUDataset('nyu_depth_v2_labeled.mat',
                                                     'test',
                                                     rgb_transform=rgb_data_transforms,
                                                     depth_transform=depth_data_transforms),
                                          batch_size=args.batch_size,
                                          shuffle=False, num_workers=0)

input_for_plot_loader = torch.utils.data.DataLoader(NYUDataset('nyu_depth_v2_labeled.mat',
                                                               'test',
                                                               rgb_transform=input_for_plot_transforms,
                                                               depth_transform=depth_data_transforms),
                                                    batch_size=args.batch_size,
                                                    shuffle=False, num_workers=0)


def plot_grid(fig, plot_input, coarse_output, fine_output, actual_output, row_no):
    grid = ImageGrid(fig, 141, nrows_ncols=(row_no, 4), axes_pad=0.05, label_mode="1")
    for i in range(row_no):
        for j in range(4):
            if (j == 0):
                grid[i * 4 + j].imshow(np.transpose(plot_input[i].detach().cpu().numpy(), (1, 2, 0)),
                                       interpolation="nearest")
            if (j == 1):
                grid[i * 4 + j].imshow(np.transpose(coarse_output[i][0].detach().cpu().numpy(), (0, 1)),
                                       interpolation="nearest")
            if (j == 2):
                grid[i * 4 + j].imshow(np.transpose(fine_output[i][0].detach().cpu().numpy(), (0, 1)),
                                       interpolation="nearest")
            if (j == 3):
                grid[i * 4 + j].imshow(np.transpose(actual_output[i][0].detach().cpu().numpy(), (0, 1)),
                                       interpolation="nearest")


batch_idx = 0
for batch_idx, (data, plot_data) in enumerate(zip(test_loader, input_for_plot_loader)):
    rgb, depth = (data['image'].to(device)).clone().detach().requires_grad_(False), (
        data['depth'].to(device)).clone().detach().requires_grad_(False)
    print(rgb)
    print(rgb.shape)
    plot_input, actual_output = (plot_data['image'].to(device)).clone().detach().requires_grad_(False), (
        plot_data['depth'].to(device)).clone().detach().requires_grad_(False)
    print('evaluating batch:' + str(batch_idx))
    coarse_output = coarse_model(rgb.type(dtype))
    fine_output = fine_model(rgb.type(dtype), coarse_output.type(dtype))
    depth_dim = list(depth.size())
    F = plt.figure(1, (30, 60))
    F.subplots_adjust(left=0.05, right=0.95)
    # pdb.set_trace()
    # plot_input = torch.exp(plot_input)-1
    # coarse_output = torch.exp(coarse_output)-1
    # fine_output = torch.exp(fine_output)-1
    # actual_output = torch.exp(actual_output)-1

    plot_grid(F, plot_input, coarse_output, fine_output, actual_output, depth_dim[0])
    plt.savefig(args.model_folder + "new_plots/" + "depth_" + str(args.model_no) + "_" + str(batch_idx) + ".pdf")
    # batch_idx = batch_idx + 1
    # if batch_idx == 1: break
