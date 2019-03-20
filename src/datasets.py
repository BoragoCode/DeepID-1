import os
import cv2
import numpy as np
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


def gen_patch(image, bbox, landmarks, patch_index, ratio):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        bbox:       {list[int]} x1, y1, x2, y2
        landmarks:  {list[int]} xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5
        patch_index:{int}       0~8
        scale:      {str}       'S', 'M', 'L'
    Returns:
        X:          {tensor(C(3), h, w)}
    """

    x1, y1, x2, y2 = bbox
    xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5 = landmarks

    if patch_index < 4:
        w = x2 - x1; h = y2 - y1
        ha = int(h*ratio); wa = int(0.75*ha)
    
        if patch_index == 0:
            dsize = (44, 33)
            x_ct = x1 + w // 2; y_ct = y1 + h // 2                  # 框中心为图片中心
        else:
            dsize = (25, 33)
            ha = int(wa*0.75)                                       # h: w = 3: 4
            if patch_index == 1:
                x_ct = (xx1 + xx2) // 2; y_ct = (yy1 + yy2) // 2    # 以两眼中心为图片中心
            elif patch_index == 2:
                x_ct = xx3; y_ct = yy3                              # 以鼻尖为中心
            elif patch_index == 3:
                x_ct = (xx4 + xx5) // 2; y_ct = (yy4 + yy5) // 2    # 以两嘴角中心为图片中心

    else:
        dsize = (25, 25)
        wa = x2 - x1; ha = wa                                       # h: w = 1: 1

        if patch_index == 4:
            x_ct = xx1; y_ct = yy1                                  # 左眼为图片中心
        elif patch_index == 5:
            x_ct = xx2; y_ct = yy2                                  # 右眼为图片中心
        elif patch_index == 6:
            x_ct = xx3; y_ct = yy3                                  # 鼻尖为图片中心
        elif patch_index == 7:
            x_ct = xx4; y_ct = yy4                                  # 左嘴角为图片中心
        elif patch_index == 8:
            x_ct = xx5; y_ct = yy5                                  # 左嘴角为图片中心
        

    x1a = x_ct - wa // 2
    x2a = x_ct + wa // 2
    y1a = y_ct - ha // 2
    y2a = y_ct + ha // 2

    x1a = 0 if x1a < 0 else x1a
    y1a = 0 if y1a < 0 else y1a
    imh, imw = image.shape[:-1]
    x2a = imw-1 if x2a >= imw else x2a
    y2a = imh-1 if y2a >= imh else y2a

    X = image[y1a:y2a, x1a:x2a, :]
    X = cv2.resize(X, dsize[::-1])
    X = ToTensor()(X)

    return X













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

















class ClassifyWithSimilarity(Dataset):
    scale_ratio = {
        'S': 0.8,
        'M': 1.0,
        'L': 1.2,
    }

    def __init__(self, patch, scale, mode='train', dsize=None):

        with open('../data/lfw_classify_similarity/lfw_classify_similarity_{}_{}.txt'.\
                                        format(patch, mode), 'r') as f:
            pairs = f.readlines()

        ratio = self.scale_ratio[scale]
        self.dsize = tuple(dsize)

        self.pairs = [[self.__parse(pairs[i], ratio), self.__parse(pairs[i+1], ratio)]\
                                        for i in range(len(pairs)//2)]

    def __parse(self, sample, ratio):
        """
        Params:
            sample: {str} path x1 y1 x2 y2 label\n
            ratio:  {float}
        """
        sample = sample.strip().split(' ')
        path = sample[0]
        x1, y1, x2, y2, label = [int(num) for num in sample[1:]]
        x1, y1, x2, y2 = aug_bbox([x1, y1, x2, y2], ratio)
        return [path, x1, y1, x2, y2, label]

    def __getitem__(self, index):
        
        pair = self.pairs[index]

        if rand(1) > 0.5: pair = pair[::-1]
        
        image1, label1 = self.__get_sample(pair[0])
        image2, label2 = self.__get_sample(pair[1])

        image1 = ToTensor()(image1)
        image2 = ToTensor()(image2)

        return image1, label1, image2, label2

    def __get_sample(self, sample):
        """
        Params:
            sample: {list[str, int, int, int, int, int]}
        """
        path, x1, y1, x2, y2, label = sample
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        
        h, w = image.shape[: 2]
        x2 = w if x2 > w else x2
        y2 = h if y2 > h else y2
        
        image = image[y1: y2, x1: x2]
        if self.dsize is not None:
            image = cv2.resize(image, self.dsize[::-1])

        return image, label

    def __len__(self):
        
        return len(self.pairs)
















class VerifyDataset(Dataset):

    __type = [[i, s] for i in range(9) for s in ['S', 'M', 'L']]

    def __init__(self, mode='train'):
        txtfile = '../data/lfw_verify/lfw_verify_{}.txt'.format(mode)
        with open(txtfile, 'r') as f:
            pairs = f.readlines()
        self.pairs = [[int(num) for num in pair.strip().split(' ')] for pair in pairs]
        self.features = self.__loadFeatures()

        self.X = dict()
        for patch, scale in self.__type:
            key = 'lfw_classify_{}_scale{}'.format(patch, scale)
            self.X[key] = np.load('../data/lfw_classify/features/{}.npy'.format(key))
        
    def __getitem__(self, index):
        """
        Params:
            index:  {int}
        Returns:
            X:      {tensor(n_groups(27), 2x160)}
        Notes:
            - {0~8} {S, M, L}
        """
        idx1, idx2, label = self.pairs[index]

        X1 = np.zeros(shape=(27, 160))
        X2 = np.zeros(shape=(27, 160))
        
        for i in range(27):
            patch, scale = self.__type[i]
            key = 'lfw_classify_{}_scale{}'.format(patch, scale)
            X1[i] = self.X[key][idx1]; X2[i] = self.X[key][idx2]
        
        if rand(1) > 0.5: X1, X2 = X2, X1
        
        X = np.concatenate([X1, X2], axis=1)
        
        return X, label

    def __len__(self):
        return len(self.pairs)
    
    def __loadFeatures(self):
        
        features = dict()
        for patch in range(9):
            for scale in ['S', 'M', 'L']:
                key = 'lfw_classify_{}_scale{}'.format(patch, scale)
                path = '../data/features/{}.npy'.format(key)
                features[key] = np.load(path)
        
        return features


















get_name_from_filepath = lambda filepath: filepath.split('/')[-2]

class DeepIdDataset(Dataset):
    """ Dataset for the whole net

    Attributes:
        scale_ratio:    {dict} 三个尺度的缩放比例
        aug:            {float} 检测出的框扩张一定比例作为patch0的框，其他框以patch0为基础产生
        pairs:          {list[list[str, list[], list[]], list[str, list[], list[]]]}
    """

    scale_ratio = {
        'S': 0.8,
        'M': 1.0,
        'L': 1.2,
    }

    def __init__(self, mode, aug=1.2):

        with open('../data/lfw_deepid_pair/lfw_deepid_pair_{}.txt'.format(mode), 'r') as f:
            pairs = f.readlines()
        self.pairs = [[self.__parse(pairs[i]), self.__parse(pairs[i+1])] for i in range(len(pairs)//2)]

        self.aug = aug


    def __parse(self, sample):
        """
        Params:
            sample: {str} 'filepath x1 y1 x2 y2 xx1 xx2 xx3 xx4 xx5 yy1 yy2 yy3 yy4 yy5'
        """
        sample = sample.strip().split(' ')
        path = sample[0]
        x1, y1, x2, y2, xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5 = [int(i) for i in sample[1:]]
        return [path, [x1, y1, x2, y2], [xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5]]


    def __getitem__(self, index):
        """
        Returns:
            X0: {tensor(2, 3, 3, 44, 33)}    patch0
            X1: {tensor(2, 3, 3, 25, 33)}    patch1
            X2: {tensor(2, 3, 3, 25, 33)}    patch2
            X3: {tensor(2, 3, 3, 25, 33)}    patch3
            X4: {tensor(2, 3, 3, 25, 25)}    patch4
            X5: {tensor(2, 3, 3, 25, 25)}    patch5
            X6: {tensor(2, 3, 3, 25, 25)}    patch6
            X7: {tensor(2, 3, 3, 25, 25)}    patch7
            X8: {tensor(2, 3, 3, 25, 25)}    patch8
        """

        pairs = self.pairs[index]
        if rand(1) > 0.5: pairs = pairs[::-1]
        
        X0 = torch.zeros(2, 3, 3, 44, 33)
        X1 = torch.zeros(2, 3, 3, 25, 33)
        X2 = torch.zeros(2, 3, 3, 25, 33)
        X3 = torch.zeros(2, 3, 3, 25, 33)
        X4 = torch.zeros(2, 3, 3, 25, 25)
        X5 = torch.zeros(2, 3, 3, 25, 25)
        X6 = torch.zeros(2, 3, 3, 25, 25)
        X7 = torch.zeros(2, 3, 3, 25, 25)
        X8 = torch.zeros(2, 3, 3, 25, 25)

        path1, bbox, landmarks = pairs[0]
        X0[0], X1[0], X2[0], X3[0], X4[0], X5[0], X6[0], X7[0], X8[0] = self.__gen_patches(cv2.imread(path1, cv2.IMREAD_COLOR), bbox, landmarks)

        path2, bbox, landmarks = pairs[1]
        X0[1], X1[1], X2[1], X3[1], X4[1], X5[1], X6[1], X7[1], X8[1] = self.__gen_patches(cv2.imread(path2, cv2.IMREAD_COLOR), bbox, landmarks)

        label = 1 if get_name_from_filepath(path1) == get_name_from_filepath(path2) else 0

        return X0, X1, X2, X3, X4, X5, X6, X7, X8, label
        
    def __gen_patches(self, image, bbox, landmarks):
        """
        Params:
            image:      {ndarray(H, W, 3)}
            bbox:       {list[int]} x1, y1, x2, y2
            landmarks:  {list[int]} xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5
        Returns:
            X0: {tensor(3, 3, 44, 33)}    patch0
            X1: {tensor(3, 3, 25, 33)}    patch1
            X2: {tensor(3, 3, 25, 33)}    patch2
            X3: {tensor(3, 3, 25, 33)}    patch3
            X4: {tensor(3, 3, 25, 25)}    patch4
            X5: {tensor(3, 3, 25, 25)}    patch5
            X6: {tensor(3, 3, 25, 25)}    patch6
            X7: {tensor(3, 3, 25, 25)}    patch7
            X8: {tensor(3, 3, 25, 25)}    patch8
        """
        
        X0 = torch.zeros(3, 3, 44, 33)
        X1 = torch.zeros(3, 3, 25, 33)
        X2 = torch.zeros(3, 3, 25, 33)
        X3 = torch.zeros(3, 3, 25, 33)
        X4 = torch.zeros(3, 3, 25, 25)
        X5 = torch.zeros(3, 3, 25, 25)
        X6 = torch.zeros(3, 3, 25, 25)
        X7 = torch.zeros(3, 3, 25, 25)
        X8 = torch.zeros(3, 3, 25, 25)

        scales = ['S', 'M', 'L']
        for i in range(len(scales)):
            X0[i] = gen_patch(image, bbox, landmarks, 0, self.scale_ratio[scales[i]])
            X1[i] = gen_patch(image, bbox, landmarks, 1, self.scale_ratio[scales[i]])
            X2[i] = gen_patch(image, bbox, landmarks, 2, self.scale_ratio[scales[i]])
            X3[i] = gen_patch(image, bbox, landmarks, 3, self.scale_ratio[scales[i]])
            X4[i] = gen_patch(image, bbox, landmarks, 4, self.scale_ratio[scales[i]])
            X5[i] = gen_patch(image, bbox, landmarks, 5, self.scale_ratio[scales[i]])
            X6[i] = gen_patch(image, bbox, landmarks, 6, self.scale_ratio[scales[i]])
            X7[i] = gen_patch(image, bbox, landmarks, 7, self.scale_ratio[scales[i]])
            X8[i] = gen_patch(image, bbox, landmarks, 8, self.scale_ratio[scales[i]])
        
        return X0, X1, X2, X3, X4, X5, X6, X7, X8

    


    def __len__(self):
        return len(self.pairs)












if __name__ == "__main__":
    # D = ClassifyDataset(0, (44, 33), 'M')
    # X, y = D[0]
    # D = VerifyDataset()
    # X, y = D[0]
    # D = ClassifyWithSimilarity
    D = DeepIdDataset('train')
    D[0]