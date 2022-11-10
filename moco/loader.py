# Parts of codes are from https://github.com/facebookresearch/moco
from PIL import ImageFilter
import random
import numpy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import torch
import os
import cv2
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from torch.utils.data import SequentialSampler, RandomSampler




class TwoCropsTransform(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.rgb_filepath_list = []
        self.rgb_filename_list= []

        self.batch_generate(image_dir)

        self.transforms = transforms
        self.image_dir = image_dir


    def batch_generate(self, image_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]

        self.rgb_filepath_list += rgb_filepath_list
        self.rgb_filename_list = rgb_filename_list

    def __getitem__(self, idx):
        image = Image.open(self.rgb_filepath_list[idx]).convert("RGB")
        image_id = self.rgb_filepath_list[idx].split("/")[-1]
        if self.transforms is not None:
            q = self.transforms(image)
            k = self.transforms(image)

        return [q, k], image_id

    def __len__(self):
        return len(self.rgb_filepath_list)





class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
