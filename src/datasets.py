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


def gen_patch(image, bbox, landmarks, patch_index, scale, ratio=1.0):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        bbox:       {list[int]} x1, y1, x2, y2
        landmarks:  {list[int]} xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5
        patch_index:{int}       0~8
            - 0: 包括发型的脸，当前检测结果框扩大一定比例(1.2)，截取矩形(h, w; h: w=4: 3)
            - 1: 以两眼中心为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 2: 以鼻尖为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 3: 以嘴角中心为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 4：以左眼为图片中心，截取正方形(h', w; h' = 0.75*w)
            - 5：以右眼为图片中心，截取正方形(h", w; h" = w)
            - 6：以左嘴角为图片中心，截取正方形(h", w; h" = w)
            - 7：以右嘴角为图片中心，截取正方形(h", w; h" = w)
            - 8：以鼻尖为图片中心，截取正方形(h", w; h" = w)
        scale:  {str}   'S', 'M', 'L'
        ratio:  {float} 以检测结果框(x1, y1, x2, y2)扩张该倍数，作为patch0框
    Returns:
        X:          {tensor(C(3), h, w)}
    Notes:
        patch_index: {int}
        ratio:  {float} 以检测结果框(x1, y1, x2, y2)扩张该倍数，作为patch0框
    """

    ## 扩大bbox框
    x1, y1, x2, y2 = aug_bbox(bbox, ratio)
    xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5 = landmarks
    w = x2 - x1; h = y2 - y1

    ## 尺度
    dict_scale_ratio = {
        'S': 0.65,
        'M': 0.90,
        'L': 1.15,
    }
    scale_ratio = dict_scale_ratio[scale]

    if patch_index < 4:
        
        ha = int(h*scale_ratio); wa = int(0.75*ha)
    
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
        wa = int(w*scale_ratio*0.6); ha = wa                            # h: w = 1: 1

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

    ## 防止出界
    x1a = 0 if x1a < 0 else x1a
    y1a = 0 if y1a < 0 else y1a
    imh, imw = image.shape[:-1]
    x2a = imw-1 if x2a >= imw else x2a
    y2a = imh-1 if y2a >= imh else y2a

    ## resize
    X = image[y1a:y2a, x1a:x2a, :]
    X = cv2.resize(X, dsize[::-1])

    # cv2.imshow("{}_{}".format(patch_index, scale), X); cv2.waitKey(0); cv2.destroyAllWindows()

    X = ToTensor()(X)

    return X


def gen_patches(image, bbox, landmarks):
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
        X0[i] = gen_patch(image, bbox, landmarks, 0, scales[i])
        X1[i] = gen_patch(image, bbox, landmarks, 1, scales[i])
        X2[i] = gen_patch(image, bbox, landmarks, 2, scales[i])
        X3[i] = gen_patch(image, bbox, landmarks, 3, scales[i])
        X4[i] = gen_patch(image, bbox, landmarks, 4, scales[i])
        X5[i] = gen_patch(image, bbox, landmarks, 5, scales[i])
        X6[i] = gen_patch(image, bbox, landmarks, 6, scales[i])
        X7[i] = gen_patch(image, bbox, landmarks, 7, scales[i])
        X8[i] = gen_patch(image, bbox, landmarks, 8, scales[i])
        
    return X0, X1, X2, X3, X4, X5, X6, X7, X8



class ClassifyDataset(Dataset):
    """ datasets for classification

    Attributes:
        patch:  {int} 0~8
        dsize:  {list/tuple(H, W)}
        scale:  {str} 'S', 'M', 'L'
    """

    def __init__(self, patch, scale):

        self.patch = patch
        self.scale = scale

        if patch == 0:
            self.dsize = (44, 33)
        elif patch in [1, 2, 3]:
            self.dsize = (25, 33)
        else:
            self.dsize = (25, 25)

        with open('../data/celeba_classify/celeba_classify.txt', 'r') as f:
            samples = f.readlines()

        self.samples_list = []
        for sample in samples:
            sample = sample.strip().split(' ')

            path = sample[0]
            x1, y1, x2, y2, xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5, label = [int(i) for i in sample[1: ]]
            bbox = [x1, y1, x2, y2]
            landmarks = [xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5]
            sample = [path, bbox, landmarks, label-1]
            self.samples_list += [sample]
        
    def __getitem__(self, index):

        path, bbox, landmarks, label = self.samples_list[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = gen_patch(image, bbox, landmarks, self.patch, self.scale)

        return image, label

    def __len__(self):

        return len(self.samples_list)


class ClassifyPairsDataset(Dataset):

    def __init__(self, patch, scale, mode='train'):

        self.patch = patch
        self.scale = scale

        if patch == 0:
            self.dsize = (44, 33)
        elif patch in [1, 2, 3]:
            self.dsize = (25, 33)
        else:
            self.dsize = (25, 25)

        with open('../data/celeba_classify_pairs/{}.txt'.format(mode), 'r') as f:
            pairs = f.readlines()
        self.pairs = [[self.__parse(pairs[i]), self.__parse(pairs[i+1])] for i in range(len(pairs)//2)]

    def __parse(self, sample):
        """
        Params:
            sample: {str} path x1 y1 x2 y2 xx1 xx2 xx3 xx4 xx5 yy1 yy2 yy3 yy4 yy5 label\n
        """
        sample = sample.strip().split(' ')
        path = sample[0]
        x1, y1, x2, y2, xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5, label = [int(num) for num in sample[1:]]
        bbox = [x1, y1, x2, y2]
        landmarks = [xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5]
        return [path, bbox, landmarks, label]

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
            sample: {list[str, list[int], list[int], int]}
        """
        path, bbox, landmarks, label = sample
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        gen_patch(image, bbox, landmarks, self.patch, self.scale)

        return image, label

    def __len__(self):
        
        return len(self.pairs)









class DeepIdDataset(Dataset):
    """ Dataset for the whole net

    Attributes:
        pairs:          {list[
                            list[
                                str, list[int], list[int], int
                            ], 
                            list[
                                str, list[int], list[int], int
                            ]
                        ]}
    """

    def __init__(self, mode, database='celeba'):

        # with open('../data/celeba_classify_pairs/{}.txt'.format(mode), 'r') as f:
        with open('../data/lfw_classify_pairs/{}.txt'.format(mode), 'r') as f:
            pairs = f.readlines()
        self.pairs = [[self.__parse(pairs[i]), self.__parse(pairs[i+1])] for i in range(len(pairs)//2)]

    def __parse(self, sample):
        """
        Params:
            sample: {str} path x1 y1 x2 y2 xx1 xx2 xx3 xx4 xx5 yy1 yy2 yy3 yy4 yy5 label\n
        """
        sample = sample.strip().split(' ')
        path = sample[0]
        x1, y1, x2, y2, xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5, label = [int(num) for num in sample[1:]]
        bbox = [x1, y1, x2, y2]
        landmarks = [xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5]
        return [path, bbox, landmarks, label]

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

        path, bbox, landmarks, label1 = pairs[0]
        X0[0], X1[0], X2[0], X3[0], X4[0], X5[0], X6[0], X7[0], X8[0] = gen_patches(cv2.imread(path, cv2.IMREAD_COLOR), bbox, landmarks)

        path, bbox, landmarks, label2 = pairs[1]
        X0[1], X1[1], X2[1], X3[1], X4[1], X5[1], X6[1], X7[1], X8[1] = gen_patches(cv2.imread(path, cv2.IMREAD_COLOR), bbox, landmarks)

        y = 1 if label1 == label2 else 0

        return X0, X1, X2, X3, X4, X5, X6, X7, X8, y

    


    def __len__(self):
        return len(self.pairs)












if __name__ == "__main__":
    
    # for patch in range(9):
    #     for scale in ['S', 'M', 'L']:
    #         D = ClassifyDataset(patch, scale)
    #         D[0]
    
    # for patch in range(9):
    #     for scale in ['S', 'M', 'L']:
    #         D = ClassifyPairs(patch, scale, mode='train')
    #         D[0]    

    D = DeepIdDataset('train', database='lfw')
    # D = ClassifyDataset(1, 'S')
    for i in range(10):
        D[i]
