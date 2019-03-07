import os
import cv2
import torch
from torch.utils.data import Dataset

class DeepIdData(Dataset):
    def __init__(self, mode='train'):
        