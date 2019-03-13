import os
from os.path import join
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import mtcnn

def detector():
    pnet, rnet, onet = mtcnn.init_detector()
    detect = lambda image: mtcnn.detect_faces(image, pnet, rnet, onet, min_face_size=40.0)
    return detect

def detect_lfw(datadir):
    detect_txt = join('/'.join(datadir.split('/')[:-1]), 'lfw_detect.txt')
    f = open(detect_txt, 'w')
    detect = detector()
    plt.figure("Image"); plt.ion()

    n_image = 13233
    i_image = 0
    t = time.time()

    for name in os.listdir(datadir):
        oridir_name = join(datadir, name)

        for pic in os.listdir(oridir_name):

            i_image += 1
            if i_image % 100 == 0:
                print('[{:5d}]/[{:5d}], {:.2%} of images done!, {:.6f} sec per image'.\
                                format(i_image, n_image, i_image / n_image, (time.time() - t)/ 100))
                t = time.time()

            oridir_name_pic = join(oridir_name, pic)
            image = Image.open(oridir_name_pic)
            bbox, landmark = detect(image)
            # plt.imshow(mtcnn.show_bboxes(image, bbox, landmark)); plt.pause(0.001)
            if bbox.shape[0] == 0: continue
            
            ct = np.array(image.size) / 2               # 图像中心点坐标
            dist = np.zeros(shape=(bbox.shape[0]))
            for i in range(bbox.shape[0]):
                x1, y1, x2, y2, _ = bbox[i]
                bboxct = np.array([x2-x1, y2-y1]) / 2   # 框中心点坐标
                dist[i] = np.linalg.norm(bboxct - ct)
            idx = np.argmin(dist)                       # 距离图像中心最近的框
            x1, y1, x2, y2, _ = bbox[idx].astype('int')
            xx1, yy1, xx2, yy2, xx3, yy3, xx4, yy4, xx5, yy5 = landmark[idx].astype('int')

            image_bbox_landmark = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.\
                    format(oridir_name_pic, x1, y1, x2, y2, xx1, yy1, xx2, yy2, xx3, yy3, xx4, yy4, xx5, yy5)
            f.write(image_bbox_landmark)
    f.close()
    plt.ioff()