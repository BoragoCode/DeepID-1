import os
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class ClassifyDataset(Dataset):
    scale_ratio = {
        'S': 0.8,
        'M': 1.0,
        'L': 1.2,
    }

    def __init__(self, patch, dsize=None, scale='M'):

        with open('../data/lfw_classify/lfw_classify_{}.txt'.format(patch), 'r') as f:
            samples = f.readlines()

        ratio = self.scale_ratio[scale]
        self.dsize = dsize

        self.samples_list = []
        for sample in samples:
            sample = sample.strip().split(' ')
            path = sample[0]
            x1, y1, x2, y2, label = [int(i) for i in sample[1: ]]
            
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = (x2 - x1) * ratio
            h = (y2 - y1) * ratio

            x1 = int(x - w / 2)
            x1 = 0 if x1 < 0 else x1
            y1 = int(y - h / 2)
            y1 = 0 if y1 < 0 else y1
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            sample = [path, x1, y1, x2, y2, label]
            self.samples_list += [sample]
        
    def __getitem__(self, index):

        path, x1, y1, x2, y2, label = self.samples_list[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        
        h, w = image.shape[: 2]
        x2 = w if x2 > w else x2
        y2 = h if y2 > h else y2
        
        image = image[y1: y2, x1: x2]
        if self.dsize is not None:
            image = cv2.resize(image, self.dsize[:, :, -1])
        image = ToTensor(image)

        return image, label

    def __len__(self):
        return len(self.samples_list)
