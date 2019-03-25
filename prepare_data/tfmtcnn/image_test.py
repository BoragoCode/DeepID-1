# coding:utf-8
import os
import sys
import cv2
import numpy as np

from .detectors import Detector, FcnDetector, MtcnnDetector
from .models import P_Net, R_Net, O_Net

def initDetector():
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    shuffle = False
    detectors = [None, None, None]
    prefix = ['./tfmtcnn/modelfile/PNet/PNet', './tfmtcnn/modelfile/RNet/RNet', './tfmtcnn/modelfile/ONet/ONet']
    epoch = [18, 14, 16]


    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0]);       detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1]);   detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2]);   detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors,
                                    min_face_size=min_face_size,
                                    stride=stride, 
                                    threshold=thresh, 
                                    slide_window=slide_window)
    
    return mtcnn_detector


def detect_image(detector, image):
    """
    Params:
        image: {ndarray}
    """
    boxes_c, landmarks = detector.detect(image)
    return boxes_c, landmarks

