import matplotlib
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import coarseNet, fineNet
import numpy as np

matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read image
file_name = "night_dark_transparent"
direct = '/home/xiule/Programming/p_workspace/depth_esitmation_lit_reviews/depth_estimation/depth-map-prediction/'
img_path = direct + 'imgs/' + file_name + '.jpg'
img = Image.open(img_path)

trans = transforms.Compose([
    transforms.Resize((228, 304)),
    transforms.ToTensor()
])
rgb = trans(img)
rgb = rgb.view(1, 3, 228, 304)
rgb.to(device)
rgb.requires_grad = False

# define the trained model to be used
model_folder = '/home/xiule/Programming/p_workspace/depth_esitmation_lit_reviews/depth_estimation/depth-map-prediction/models'
model_no = 500

# define the model
coarse_state_dict = torch.load(model_folder + "/coarse_model_" + str(model_no) + ".pth")
fine_state_dict = torch.load(model_folder + "/fine_model_" + str(model_no) + ".pth")

coarse_model = coarseNet()
fine_model = fineNet()
coarse_model.to(device)
fine_model.to(device)

coarse_model.load_state_dict(coarse_state_dict)
fine_model.load_state_dict(fine_state_dict)
coarse_model.eval()
fine_model.eval()

# inference
dtype = torch.cuda.FloatTensor

coarse_output = coarse_model(rgb.type(dtype))
fine_output = fine_model(rgb.type(dtype), coarse_output.type(dtype))

# plotting
fig = plt.figure(1)
plt.subplot(121)
plt.imshow(np.transpose(rgb[0].detach().cpu().numpy(), (1, 2, 0)), interpolation="nearest")
plt.subplot(122)
plt.imshow(np.transpose(fine_output[0][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")
plt.savefig(direct + "new_plots/" + "depth_" + file_name + ".pdf")
