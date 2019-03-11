import os
import cv2
import numpy as np
from random import uniform, randint

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from config import configer
from utiles import is_same_label, is_same_posit

def resizeMulti(image, dsize):
    """
    Params:
        image: {ndarray(H, W, C)}
        dsize: {tuple(H, W)}
    """
    c = image.shape[-1]
    ret = np.zeros(shape=(dsize[0], dsize[1], c))
    for i in range(c):
        ret[:, :, i] = cv2.resize(image[:, :, i], tuple(dsize)[::-1])
    return ret

def cropImage(image, croprate, offsetx, offsety):
    h, w, c = image.shape
    wc = w * croprate; hc = h * croprate
    xc = w * offsetx;  yc = h * offsety
    x1 = w/2 + xc - wc/2; x2 = x1 + wc
    y1 = h/2 + yc - hc/2; y2 = y1 + hc
    image = image[int(y1): int(y2), int(x1): int(x2), :]
    return image

def rotateImage(image, degree_max):
    h, w, c = image.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), np.random.randint(-degree_max, degree_max), 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    return image

class DeepIdData(Dataset):
    
    def __init__(self, mode='train', crop_max=0.7, rotate_max=45):
        self.mode = mode
        self.crop_max = crop_max
        self.rotate_max = rotate_max
        with open('../prepare_data/{}.txt'.format(mode), 'r') as f:
            self.pairlist = [pair.strip().split(' ') for pair in f.readlines()]
    
    def __getitem__(self, index):
        path1, path2 = self.pairlist[index]
        image1 = np.load(path1)[:, :, configer.use_channels]
        image2 = np.load(path2)[:, :, configer.use_channels]

        if self.mode == 'train':
            if np.random.random() > 0.5:                                        # 交替
                image1, image2 = image2, image1
            if is_same_posit(path1, path2):                                     # 如果是相同角度拍摄，则进行剪裁
                r = uniform(self.crop_max, 1.0)
                x = uniform(0.0, 1-r) - (1-r)/2
                y = uniform(0.0, 1-r) - (1-r)/2
                image1 = cropImage(image1, r, x, y)
                image2 = cropImage(image2, r, x, y)
            image1 = image1[:, ::-1, :] if np.random.random() > 0.5 else image1 # 镜像
            image2 = image2[:, ::-1, :] if np.random.random() > 0.5 else image2
            image1 = rotateImage(image1, self.rotate_max)                       # 旋转
            image2 = rotateImage(image2, self.rotate_max)

        image1 = resizeMulti(image1, configer.imgsize)
        image2 = resizeMulti(image2, configer.imgsize)
        image1 = ToTensor()(image1)
        image2 = ToTensor()(image2)
        label = int(is_same_label(path1, path2))
        return image1, image2, label
            
    def __len__(self):
        return len(self.pairlist)
