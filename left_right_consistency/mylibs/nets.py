import torch
from torch import nn
import torchvision.models as models
from .loss import Loss


class Net(nn.Module):
    def __init__(self, use_GPU):
        super(Net, self).__init__()
        self.use_GPU = use_GPU
        resnet50_model = models.resnet50(pretrained=True)

        # encoder
        self.enc_lay0 = nn.Sequential(
            resnet50_model.conv1,
            resnet50_model.bn1,
            resnet50_model.relu,
            resnet50_model.maxpool
        )
        self.enc_lay1 = resnet50_model.layer1
        self.enc_lay2 = resnet50_model.layer2
        self.enc_lay3 = resnet50_model.layer3
        self.enc_lay4 = resnet50_model.layer4

        # decoder
        # decoder layer 1
        self.dec_elu = nn.ELU()
        self.dec_lay1_UC = nn.ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), dilation=1, padding=1,
                                              output_padding=1)
        self.dec_lay1_dispL = self.disp_gen(2048, 0.3 * 38)
        self.dec_lay1_dispR = self.disp_gen(2048, 0.3 * 38)
        self.dec_lay1_C = nn.Conv2d(2050, 1024, kernel_size=(3, 3), padding=1, stride=1)
        self.dec_lay1_bn = nn.BatchNorm2d(1024)

        # decoder layer 2
        self.dec_lay2_UC = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), dilation=1, padding=1,
                                              output_padding=1)
        self.dec_lay2_dispL = self.disp_gen(1024, 0.3 * 76)
        self.dec_lay2_dispR = self.disp_gen(1024, 0.3 * 76)
        self.dec_lay2_C = nn.Conv2d(1026, 512, kernel_size=(3, 3), padding=1, stride=1)
        self.dec_lay2_bn = nn.BatchNorm2d(512)

        # decoder layer 3
        self.dec_lay3_UC = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), dilation=1, padding=1,
                                              output_padding=1)
        self.dec_lay3_dispL = self.disp_gen(512, 0.3 * 152)
        self.dec_lay3_dispR = self.disp_gen(512, 0.3 * 152)
        self.dec_lay3_C = nn.Conv2d(514, 256, kernel_size=(3, 3), padding=1, stride=1)
        self.dec_lay3_bn = nn.BatchNorm2d(256)

        # decoder layer 4
        self.dec_lay4_C1 = nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), dilation=1, padding=1)
        self.dec_lay4_dispL = self.disp_gen(256, 0.3 * 304)
        self.dec_lay4_dispR = self.disp_gen(256, 0.3 * 304)
        self.dec_lay4_C2 = nn.Conv2d(130, 64, kernel_size=(3, 3), padding=1, stride=1)
        self.dec_lay4_bn = nn.BatchNorm2d(64)

        # decoder layer 5
        self.dec_lay5_UC1 = nn.ConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(2, 2), dilation=1, padding=1,
                                               output_padding=1)
        self.dec_lay5_bn1 = nn.BatchNorm2d(16)
        self.dec_lay5_UC2 = nn.ConvTranspose2d(16, 4, kernel_size=(3, 3), stride=(2, 2), dilation=1, padding=1,
                                               output_padding=1)
        self.dec_lay5_bn2 = nn.BatchNorm2d(4)
        self.dec_lay5_dispL = self.disp_gen(4, 0.3 * 1216)
        self.dec_lay5_dispR = self.disp_gen(4, 0.3 * 1216)

    def forward(self, left_img, right_img=None):
        img_sz = list(left_img.size())
        # encoder
        enc0 = self.enc_lay0(left_img)  # 64xH/4XW/4
        enc1 = self.enc_lay1(enc0)  # 256xH/4xW/4
        enc2 = self.enc_lay2(enc1)  # 512xH/8xW/8
        enc3 = self.enc_lay3(enc2)  # 1024xH/16XW/16
        enc4 = self.enc_lay4(enc3)  # 2048xH/32xW/32

        # decoder
        # decoder layer 1
        # dispL - used by right_img to reconstruct left image
        # dispR - used by left_img to reconstruct right image
        enc4_sz = list(enc4.size())
        dec1 = self.dec_lay1_UC(enc4)  # 1024xH/16xW/16
        dec1 = self.dec_elu(dec1)  # 1024xH/16xW/16
        # limit/scale disparity to max = 0.3*width of the image at the given scale
        dec1_dispL = 0.3 * enc4_sz[3] * self.dec_lay1_dispL(enc4)  # 1xH/32xW/32
        dec1_dispR = 0.3 * enc4_sz[3] * self.dec_lay1_dispR(enc4)  # 1xH/32xW/32
        enc3_sz = list(enc3.size())
        dec1_dispL_up = nn.Upsample(size=(enc3_sz[2], enc3_sz[3]))(dec1_dispL)  # 1xH/16xW/16
        dec1_dispR_up = nn.Upsample(size=(enc3_sz[2], enc3_sz[3]))(dec1_dispR)  # 1xH/16xW/16
        dec1 = torch.cat((dec1, enc3, dec1_dispR_up, dec1_dispL_up), 1)  # 2050xH/16xW/16
        dec1 = self.dec_lay1_C(dec1)  # 1024xH/16xW/16
        dec1 = self.dec_elu(dec1)  # 1024xH/16xW/16

        # decoder layer 2
        dec2 = self.dec_lay2_UC(dec1)  # 512xH/8xW/8
        dec2 = self.dec_elu(dec2)  # 512xH/8xW/8
        dec2_dispL = 0.3 * enc3_sz[3] * self.dec_lay2_dispL(dec1)  # 1xH/16xW/16
        dec2_dispR = 0.3 * enc3_sz[3] * self.dec_lay2_dispR(dec1)  # 1xH/16xW/16
        enc2_sz = list(enc2.size())
        dec2_dispL_up = nn.Upsample(size=(enc2_sz[2], enc2_sz[3]))(dec2_dispL)  # 1xH/8xW/8
        dec2_dispR_up = nn.Upsample(size=(enc2_sz[2], enc2_sz[3]))(dec2_dispR)  # 1xH/8xW/8
        dec2 = torch.cat((dec2, enc2, dec2_dispR_up, dec2_dispL_up), 1)  # 1026xH/8xW/8
        dec2 = self.dec_lay2_C(dec2)  # 512xH/8xW/8
        dec2 = self.dec_elu(dec2)  # 512xH/8xW/8

        # decoder layer 3
        dec3 = self.dec_lay3_UC(dec2)  # 256xH/4xW/4
        dec3 = self.dec_elu(dec3)  # 256xH/4xW/4
        dec3_dispL = 0.3 * enc2_sz[3] * self.dec_lay3_dispL(dec2)  # 1xH/8xW/8
        dec3_dispR = 0.3 * enc2_sz[3] * self.dec_lay3_dispR(dec2)  # 1xH/8xW/8
        enc1_sz = list(enc1.size())
        dec3_dispL_up = nn.Upsample(size=(enc1_sz[2], enc1_sz[3]))(dec3_dispL)  # 1xH/4xW/4
        dec3_dispR_up = nn.Upsample(size=(enc1_sz[2], enc1_sz[3]))(dec3_dispR)  # 1xH/4xW/4
        dec3 = torch.cat((dec3, enc1, dec3_dispR_up, dec3_dispL_up), 1)  # 514xH/4xW/4
        dec3 = self.dec_lay3_C(dec3)  # 256xH/4xW/4
        dec3 = self.dec_elu(dec3)  # 256xH/4xW/4

        # decoder layer 4
        dec4 = self.dec_lay4_C1(dec3)  # 64xH/4xW/4
        dec4 = self.dec_elu(dec4)  # 64xH/4xW/4
        dec4_dispL = 0.3 * enc1_sz[3] * self.dec_lay4_dispL(dec3)  # 1xH/4xW/4
        dec4_dispR = 0.3 * enc1_sz[3] * self.dec_lay4_dispR(dec3)  # 1xH/4xW/4
        dec4 = torch.cat((dec4, enc0, dec4_dispR, dec4_dispL), 1)  # 130xH/4xW/4
        dec4 = self.dec_lay4_C2(dec4)  # 64xH/4xW/4
        dec4 = self.dec_elu(dec4)  # 64xH/4xW/4

        # decoder layer 5
        dec5 = self.dec_lay5_UC1(dec4)  # 16xH/2xW/2
        dec5 = self.dec_elu(dec5)  # 16xH/2xW/2
        dec5 = self.dec_lay5_UC2(dec5)  # 4xHxW
        dec5 = self.dec_elu(dec5)  # 4xHxW
        dec5_dispL = 0.3 * img_sz[3] * self.dec_lay5_dispL(dec5)  # 1xHxW
        dec5_dispR = 0.3 * img_sz[3] * self.dec_lay5_dispR(dec5)  # 1xHxW

        # loss calculation
        # only calculate loss when both images are available
        if right_img is not None:
            # decoder layer1 loss
            dec1_scale = 32
            dec1_imL, dec1_imR = self.scale_img(left_img, right_img, dec1_scale)
            dec1_Loss = Loss(dec1_imL, dec1_imR, dec1_dispL, dec1_dispR, dec1_scale, self.use_GPU)
            dec1_loss = dec1_Loss()

            # decoder layer2 loss
            dec2_scale = 16
            dec2_imL, dec2_imR = self.scale_img(left_img, right_img, dec2_scale)
            dec2_Loss = Loss(dec2_imL, dec2_imR, dec2_dispL, dec2_dispR, dec2_scale, self.use_GPU)
            dec2_loss = dec2_Loss()

            # decoder layer3 loss
            dec3_scale = 8
            dec3_imL, dec3_imR = self.scale_img(left_img, right_img, dec3_scale)
            dec3_Loss = Loss(dec3_imL, dec3_imR, dec3_dispL, dec3_dispR, dec3_scale, self.use_GPU)
            dec3_loss = dec3_Loss()

            # decoder layer4 loss
            dec4_scale = 4
            dec4_imL, dec4_imR = self.scale_img(left_img, right_img, dec4_scale)
            dec4_Loss = Loss(dec4_imL, dec4_imR, dec4_dispL, dec4_dispR, dec4_scale, self.use_GPU)
            dec4_loss = dec4_Loss()

            # decoder layer5 loss
            dec5_scale = 1
            dec5_Loss = Loss(left_img, right_img, dec5_dispL, dec5_dispR, dec5_scale, self.use_GPU)
            dec5_loss = dec5_Loss()

            total_loss = dec1_loss + dec2_loss + dec3_loss + dec4_loss + dec5_loss

        else:
            total_loss = None

        if self.training:
            return total_loss
        else:
            return total_loss, dec5_dispL

    def disp_gen(self, input_channel, alpha=1):
        # note disparity is calculated by sigmoid here.
        # so its range is between [0, 1], which needs to be scaled
        # when calculating loss, we may need to (-1)*left_disparity if needed
        # left_disparity is used by right image to reconstruct left_image
        disp_lay = nn.Sequential(
            nn.Conv2d(input_channel, 1, kernel_size=(1, 1), stride=(1, 1)),
	    # nn.ELU(alpha = alpha)
            nn.Sigmoid()
        )
        return disp_lay

    def scale_img(self, left_img, right_img, scale):
        left_im = nn.functional.interpolate(left_img, scale_factor=1 / scale, mode='bilinear', align_corners=False)
        right_im = nn.functional.interpolate(right_img, scale_factor=1 / scale, mode='bilinear', align_corners=False)
        return left_im, right_im
