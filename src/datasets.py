import os
import cv2
from numpy.random import rand

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def aug_bbox(bbox, ratio):
    """
    Params:
        bbox:   {list[x1, y1, x2, y2]}
        ratio:  {float}
    Returns:
        bbox:   {list[x1, y1, x2, y2]}
    """
    x1, y1, x2, y2 = bbox

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

    return [x1, y1, x2, y2]

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
        self.dsize = tuple(dsize)

        self.samples_list = []
        for sample in samples:
            sample = sample.strip().split(' ')

            path = sample[0]
            x1, y1, x2, y2, label = [int(i) for i in sample[1: ]]
            x1, y1, x2, y2 = aug_bbox([x1, y1, x2, y2], ratio)

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
            image = cv2.resize(image, self.dsize[::-1])

        image = ToTensor()(image)

        return image, label

    def __len__(self):
        return len(self.samples_list)



class ClassifyWithVerifyDataset(Dataset):
    scale_ratio = {
        'S': 0.8,
        'M': 1.0,
        'L': 1.2,
    }

    def __init__(self, patch, mode, dsize=None, scale='M'):

        with open('../data/lfw_classify_verify/lfw_classify_verify_{}_{}.txt'.format(patch, mode), 'r') as f:
            samples = f.readlines()

        ratio = self.scale_ratio[scale]
        self.dsize = tuple(dsize)

        self.samples_list = [{
                                'ver'   : -1, 
                                'path1' : "",
                                'bbox1' : [],
                                'cls1'  : -1,
                                'path2' : "",
                                'bbox2' : [],
                                'cls2'  : -1,
                            } for i in range(len(samples))]
        for i in range(len(samples)):
            sample = samples[i].strip().split(' ')
            self.samples_list[i]['ver']     = int(sample[0])
            self.samples_list[i]['path1']   = sample[1]
            self.samples_list[i]['bbox1']   = [int(s) for s in sample[2: 6]]
            self.samples_list[i]['bbox1']   = aug_bbox(self.samples_list[i]['bbox1'], ratio)
            self.samples_list[i]['cls1']    = int(sample[6])
            self.samples_list[i]['path2']   = sample[7]
            self.samples_list[i]['bbox2']   = [int(s) for s in sample[8: 12]]
            self.samples_list[i]['bbox2']   = aug_bbox(self.samples_list[i]['bbox2'], ratio)
            self.samples_list[i]['cls2']    = int(sample[12])

    def __getitem__(self, index):
        
        dict_sample = self.samples_list[index]

        image1 = cv2.imread(dict_sample['path1'], cv2.IMREAD_COLOR)
        h, w = image1.shape[: 2]
        x1, y1, x2, y2 = dict_sample['bbox1']
        x2 = w if x2 > w else x2
        y2 = h if y2 > h else y2
        image1 = image1[y1: y2, x1: x2]

        image2 = cv2.imread(dict_sample['path2'], cv2.IMREAD_COLOR)
        h, w = image2.shape[: 2]
        x1, y1, x2, y2 = dict_sample['bbox2']
        x2 = w if x2 > w else x2
        y2 = h if y2 > h else y2
        image2 = image2[y1: y2, x1: x2]

        if rand(1) > 0.5: iamge1, image2 = image2, image1

        if self.dsize is not None:
            image1 = cv2.resize(image1, self.dsize[::-1])
            image2 = cv2.resize(image2, self.dsize[::-1])

        image1 = ToTensor()(image1)
        image2 = ToTensor()(image2)

        return dict_sample['ver'], image1, dict_sample['cls1'], image2, dict_sample['cls2']
        

    def __len__(self):
        return len(self.samples_list)


class VerifyDataset(Dataset):

    def __init__(self, patch, scale, mode='train'):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass


if __name__ == "__main__":
    D = ClassifyDataset(0, (44, 33), 'M')
    X, y = D[0]