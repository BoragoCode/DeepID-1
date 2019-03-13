import os
import numpy as np
from PIL import Image
from PIL import ImageDraw

get_name_from_filepath = lambda filepath: filepath.split('/')[-2]

def gen_labels(datadir):
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines()
    
    # 读取所有检测出人脸的图片对应人名
    names = []
    for detect in detect_list:
        imgpath = detect.split(' ')[0]
        name = get_name_from_filepath(imgpath)
        names += [name]
    # 去重，产生对应标签
    names = list(set(names))
    labels = [i for i in range(len(names))]
    
    # 保存标签
    labels_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_labels.txt')
    f = open(labels_txt, 'w')
    for name, label in zip(names, labels):
        name_label = name + ' ' + str(label) + '\n'
        f.write(name_label)
    f.close()

def gen_classify(datadir, patch_index):
    """
    Params:
        patch_index: {int}
            - 0: 包括发型的脸，当前检测结果框扩大一定比例，截取矩形
            - 1: 以两眼中心为图片中心，截取矩形
            - 2: 以鼻尖为图片中心，截取矩形
            - 3: 以嘴角中心为图片中心，截取矩形
            - 4：以左眼为图片中心，截取正方形
            - 5：以右眼为图片中心，截取正方形
            - 6：以左嘴角为图片中心，截取正方形
            - 7：以右嘴角为图片中心，截取正方形
            - 8：以鼻尖为图片中心，截取正方形
    """
    # 读取检测结果
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines() # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    # 读取标签
    labels_txt =  os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_labels.txt')
    with open(labels_txt, 'r') as f:
        labels_list = f.readlines() # name label
    labels_list = [l.split(' ') for l in labels_list]
    dict_name_label = {name_label[0]: name_label[1].strip() for name_label in labels_list}
    # 保存样本
    classify_dir = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_classify')
    if not os.path.exists(classify_dir): os.mkdir(classify_dir)
    classify_txt = os.path.join(classify_dir, 'lfw_classify_{}.txt'.format(patch_index))
    f = open(classify_txt, 'w')

    n_detect = len(detect_list)
    for i_detect in range(n_detect):

        filepath_bbox_landmark = detect_list[i_detect].strip()

        # 解析
        filepath_bbox_landmark = filepath_bbox_landmark.split(' ')
        filepath = filepath_bbox_landmark[0]
        bbox_landmark = [int(i) for i in filepath_bbox_landmark[1:]]
        x1, y1, x2, y2, xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5 = bbox_landmark
        label = dict_name_label[get_name_from_filepath(filepath)]

        # 生成样本
        if patch_index < 4:
            w = x2 - x1; h = y2 - y1
            ha = int(h*1.5); wa = int(0.75*ha)                          # h: w = 4: 3

            if patch_index == 0:
                x_ct = x1 + w // 2; y_ct = y1 + h // 2                  # 框中心为图片中心
            else:
                ha = int(wa*0.75)                                       # h: w = 3: 4
                if patch_index == 1:
                    x_ct = (xx1 + xx2) // 2; y_ct = (yy1 + yy2) // 2    # 以两眼中心为图片中心
                elif patch_index == 2:
                    x_ct = xx3; y_ct = yy3                              # 以鼻尖为中心
                elif patch_index == 3:
                    x_ct = (xx4 + xx5) // 2; y_ct = (yy4 + yy5) // 2    # 以两嘴角中心为图片中心

        else:
            wa = x2 - x1; ha = wa                                        # h: w = 1: 1

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
        
        # 计算生成的框坐标
        x1a = x_ct - wa // 2
        x2a = x_ct + wa // 2
        y1a = y_ct - ha // 2
        y2a = y_ct + ha // 2

        # 显示图片
        # img = Image.open(filepath)
        # draw = ImageDraw.Draw(img)
        # draw.rectangle([(x1a, y1a), (x2a, y2a)], outline='white')
        # draw.ellipse([(x_ct - 1.0, y_ct - 1.0), (x_ct + 1.0, y_ct + 1.0)], outline='blue')
        # img.show()
        
        image_bbox_label = '{} {} {} {} {} {}\n'.\
                    format(filepath, x1a, y1a, x2a, y2a, label)
        f.write(image_bbox_label)

    f.close()