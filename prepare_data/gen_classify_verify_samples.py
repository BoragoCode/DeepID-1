import os
from os.path import join
import numpy as np
from PIL import Image
from PIL import ImageDraw

get_name_from_filepath = lambda filepath: filepath.split('/')[-2]

def gen_aug_bbox(patch_index, ratio, detect):
    x1, y1, x2, y2, xx1, xx2, xx3, xx4, xx5, yy1, yy2, yy3, yy4, yy5 = [int(i) for i in detect.split(' ')]
    if patch_index < 4:
        w = x2 - x1; h = y2 - y1
        ha = int(h*ratio); wa = int(0.75*ha)                        # h: w = 4: 3

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

    return x1a, y1a, x2a, y2a



def gen_classify_verify_pairs(datadir, patch_index, ratio=1.2):
    """
    Params:
        datadir:    {str}
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
        ratio:  {float} 以检测结果框扩张该倍数，作为patch0框
    Notes:
        - 以LFW数据集已划分的训练集与测试集合为基础
    """
    # 读取标签
    labels_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_labels.txt')
    with open(labels_txt, 'r') as f:
        labels_list = f.readlines() # name label
    labels_list = [l.split(' ') for l in labels_list]
    dict_name_label = {name_label[0]: int(name_label[1].strip()) for name_label in labels_list}

    # 读取检测结果
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines() # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    detect_list = [detect.strip() for detect in detect_list]
    dict_filepath_detect = {detect.split(' ')[0]: ' '.join(detect.split(' ')[1: ]) \
                                            for detect in detect_list}
    
    # 保存结果文件夹
    classify_verify_dir = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_classify_verify')
    if not os.path.exists(classify_verify_dir): os.mkdir(classify_verify_dir)




    # 读取预划分训练数据集
    train_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view1/pairsDevTrain.txt')
    with open(train_txt, 'r') as f:
        train_list = f.readlines()[1:]  # name index1 index2 or name1 index1 name2 index2

    train_list_gen = []
    for train in train_list:
        train = train.strip().split('\t')

        if len(train) == 3: # 正样本, name index1 index2
            name, index1, index2 = train
            name1 = name; name2 = name
            label_bin = 1
        else:               # 负样本
            name1, index1, name2, index2 = train
            label_bin = 0
        
        index1 = int(index1); index2 = int(index2)

        filepath1 = join(datadir, '{}/{}_{:04d}.jpg'.format(name1, name1, index1))
        x1a1, y1a1, x2a1, y2a1 = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[filepath1])
        label1 = dict_name_label[get_name_from_filepath(filepath1)]

        filepath2 = join(datadir, '{}/{}_{:04d}.jpg'.format(name2, name2, index2))
        x1a2, y1a2, x2a2, y2a2 = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[filepath2])
        label2 = dict_name_label[get_name_from_filepath(filepath2)]
        
        image_bbox_label = '{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.\
                    format(label_bin, filepath1, x1a1, y1a1, x2a1, y2a1, label1, 
                            filepath2, x1a2, y1a2, x2a2, y2a2, label2)
        train_list_gen.append(image_bbox_label)
    # 保存训练样本
    classify_verify_train_txt = os.path.join(classify_verify_dir, 'lfw_classify_verify_{}_train.txt'.format(patch_index))
    with open(classify_verify_train_txt, 'w') as f:
        f.writelines(train_list_gen)





    # 读取预划分验证数据集
    valid_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view1/pairsDevTest.txt')
    with open(valid_txt, 'r') as f:
        valid_list = f.readlines()[1:]  # name index1 index2 or name1 index1 name2 index2

    valid_list_gen = []
    for valid in valid_list:
        valid = valid.strip().split('\t')

        if len(valid) == 3: # 正样本, name index1 index2
            name, index1, index2 = valid
            name1 = name; name2 = name
            label_bin = 1
        else:               # 负样本
            name1, index1, name2, index2 = valid
            label_bin = 0
        
        index1 = int(index1); index2 = int(index2)

        filepath1 = join(datadir, '{}/{}_{:04d}.jpg'.format(name1, name1, index1))
        x1a1, y1a1, x2a1, y2a1 = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[filepath1])
        label1 = dict_name_label[get_name_from_filepath(filepath1)]

        filepath2 = join(datadir, '{}/{}_{:04d}.jpg'.format(name2, name2, index2))
        x1a2, y1a2, x2a2, y2a2 = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[filepath2])
        label2 = dict_name_label[get_name_from_filepath(filepath2)]
        
        image_bbox_label = '{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.\
                    format(label_bin, filepath1, x1a1, y1a1, x2a1, y2a1, label1, 
                            filepath2, x1a2, y1a2, x2a2, y2a2, label2)
        valid_list_gen.append(image_bbox_label)
    # 保存验证样本
    classify_verify_valid_txt = os.path.join(classify_verify_dir, 'lfw_classify_verify_{}_valid.txt'.format(patch_index))
    with open(classify_verify_valid_txt, 'w') as f:
        f.writelines(valid_list_gen)





    # 读取预划分测试数据集
    test_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view2/pairs.txt')
    with open(test_txt, 'r') as f:
        test_list = f.readlines()[1:]  # name index1 index2 or name1 index1 name2 index2

    test_list_gen = []
    for test in test_list:
        test = test.strip().split('\t')

        if len(test) == 3: # 正样本, name index1 index2
            name, index1, index2 = test
            name1 = name; name2 = name
            label_bin = 1
        else:               # 负样本
            name1, index1, name2, index2 = test
            label_bin = 0
        
        index1 = int(index1); index2 = int(index2)

        filepath1 = join(datadir, '{}/{}_{:04d}.jpg'.format(name1, name1, index1))
        x1a1, y1a1, x2a1, y2a1 = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[filepath1])
        label1 = dict_name_label[get_name_from_filepath(filepath1)]

        filepath2 = join(datadir, '{}/{}_{:04d}.jpg'.format(name2, name2, index2))
        x1a2, y1a2, x2a2, y2a2 = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[filepath2])
        label2 = dict_name_label[get_name_from_filepath(filepath2)]
        
        image_bbox_label = '{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.\
                    format(label_bin, filepath1, x1a1, y1a1, x2a1, y2a1, label1, 
                            filepath2, x1a2, y1a2, x2a2, y2a2, label2)
        test_list_gen.append(image_bbox_label)
    # 保存测试样本
    classify_verify_test_txt = os.path.join(classify_verify_dir, 'lfw_classify_verify_{}_test.txt'.format(patch_index))
    with open(classify_verify_test_txt, 'w') as f:
        f.writelines(test_list_gen)
    



