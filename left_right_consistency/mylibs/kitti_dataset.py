import os
from skimage import io
from torch.utils.data import Dataset


class KITTIDataset(Dataset):
    def __init__(self, data_path, split='train', transforms=None, sanity_check=None, data_path_override=None):
        if data_path_override is not None:  # data_path_override is mostly for Google Colab
            self.data_path = data_path_override
        else:  # on local machine
            root = os.path.dirname(os.path.realpath(__file__))
            root = root.split(os.sep)
            while root[-1] != 'left_right_consistency':
                _ = root.pop(-1)  # remove mylibs from the root directory list
            root = os.path.join('/', *root)
            self.data_path = os.path.join(root, data_path)

        self.transforms = transforms
        if split == 'train':
            self.data_path = os.path.join(self.data_path, 'training')
        else:
            self.data_path = os.path.join(self.data_path, 'testing')
        left_im_path = os.path.join(self.data_path, 'colored_0')
        right_im_path = os.path.join(self.data_path, 'colored_1')

        if split == 'train':  # only the training folder comes with disparity from LiDAR
            disp_path = os.path.join(self.data_path, 'disp_noc')
            disp_file_name = os.listdir(disp_path)
            self.disparity = [os.path.join(disp_path, x) for x in disp_file_name]
        else:
            self.disparity = None

        img_file_name = os.listdir(left_im_path)
        self.left_img = [os.path.join(left_im_path, x) for x in img_file_name]
        self.right_img = [os.path.join(right_im_path, x) for x in img_file_name]
        if sanity_check:
            self.left_img = [self.left_img[sanity_check]]
            self.right_img = [self.right_img[sanity_check]]
            if self.disparity is not None:
                self.disparity = [self.disparity[sanity_check]]

    def __len__(self):
        return len(self.left_img)

    def __getitem__(self, item):
        left_img = io.imread(self.left_img[item])
        right_img = io.imread(self.right_img[item])
        if self.transforms is not None:
            for t in self.transforms:
                left_img, right_img = t(left_img, right_img)
        return left_img, right_img

    def get_transformed_img_pair(self, id):
        # get transformed img pair by id
        return self.__getitem__(id)

    def get_original_img_pair(self, id):
        # get original img pair by id
        left_img = io.imread(self.left_img[id])
        right_img = io.imread(self.right_img[id])
        return left_img, right_img

    def get_disp(self, id):
        # get disparity by id
        # note number of disparity files < img files
        # get the disparity file that is closest to the image file corresponding to id
        path = self.left_img[id]
        path = path.split(os.sep)
        disp_file = path[-1]
        if disp_file[-5] == '1':
            disp_file_mod = list(disp_file)
            disp_file_mod[-5] = '0'
            disp_file = "".join(disp_file_mod)
        disp_path = os.path.join(self.data_path, 'disp_noc', disp_file)
        return io.imread(disp_path)
