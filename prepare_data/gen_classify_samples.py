import os
import numpy as np
from PIL import Image
from PIL import ImageDraw

get_name_from_filepath = lambda filepath: filepath.split('/')[-2]
gen_filepath = lambda name, index: '{}/{}_{:04d}.jpg'.format(name, name, index)

def gen_aug_bbox(patch_index, ratio, detect):
    """
    Params:
        patch_index:    {int} 0~8
        ratio:          {float}
        detect:         {str} x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    """
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



















def gen_classify(datadir, patch_index, ratio=1.2):
    """
    Params:
        datadir:    {str}
        patch_index: {int}
            - 0: 包括发型的脸，当前检测结果框扩大一定比例(1.2)，截取矩形(h, w; h: w=4: 3)
            - 1: 以两眼中心为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 2: 以鼻尖为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 3: 以嘴角中心为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 4：以左眼为图片中心，截取正方形(h', w; h' = 0.75*w)
            - 5：以右眼为图片中心，截取正方形(h", w; h" = w)
            - 6：以左嘴角为图片中心，截取正方形(h", w; h" = w)
            - 7：以右嘴角为图片中心，截取正方形(h", w; h" = w)
            - 8：以鼻尖为图片中心，截取正方形(h", w; h" = w)
        ratio:  {float} 以检测结果框(x1, y1, x2, y2)扩张该倍数，作为patch0框
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
























def gen_classify_similarity_pairs(datadir, patch_index, ratio=1.2):
    """
    Params:
        datadir:    {str}
        patch_index: {int}
            - 0: 包括发型的脸，当前检测结果框扩大一定比例(1.2)，截取矩形(h, w; h: w=4: 3)
            - 1: 以两眼中心为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 2: 以鼻尖为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 3: 以嘴角中心为图片中心，截取矩形(h', w; h' = 0.75*w)
            - 4：以左眼为图片中心，截取正方形(h', w; h' = 0.75*w)
            - 5：以右眼为图片中心，截取正方形(h", w; h" = w)
            - 6：以左嘴角为图片中心，截取正方形(h", w; h" = w)
            - 7：以右嘴角为图片中心，截取正方形(h", w; h" = w)
            - 8：以鼻尖为图片中心，截取正方形(h", w; h" = w)
        ratio:  {float} 以检测结果框(x1, y1, x2, y2)扩张该倍数，作为patch0框
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
    classify_similarity = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_classify_similarity')
    if not os.path.exists(classify_similarity): os.mkdir(classify_similarity)



    ## trainsets
    train_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view1', 'pairsDevTrain.txt')
    with open(train_txt, 'r') as f:
        train_list = f.readlines()[1:]  # name index1 index2 or name1 index1 name2 index2

    train_list_gen = []
    for train in train_list:

        train = train.strip().split('\t')
        if len(train)==3:   # 正样本            
            name, index1, index2 = train
            path1 = '{}/{}'.format(datadir, gen_filepath(name, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name, int(index2)))
            name1 = name; name2 = name
        else:               # 负样本
            name1, index1, name2, index2 = train
            path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))

        x1a, y1a, x2a, y2a = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[path1])
        path_bbox_label = '{} {} {} {} {} {}\n'.format(path1, int(x1a), int(y1a), int(x2a), int(y2a), dict_name_label[name1])
        train_list_gen += [path_bbox_label]
        x1a, y1a, x2a, y2a = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[path2])
        path_bbox_label = '{} {} {} {} {} {}\n'.format(path2, int(x1a), int(y1a), int(x2a), int(y2a), dict_name_label[name2])
        train_list_gen += [path_bbox_label]

    with open(os.path.join(classify_similarity, 'lfw_classify_similarity_{}_train.txt'.format(patch_index)), 'w') as f:
        f.writelines(train_list_gen)


    ## validsets
    valid_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view1', 'pairsDevTest.txt')
    with open(valid_txt, 'r') as f:
        valid_list = f.readlines()[1:]  # name index1 index2 or name1 index1 name2 index2

    valid_list_gen = []
    for valid in valid_list:

        valid = valid.strip().split('\t')
        if len(valid)==3:   # 正样本            
            name, index1, index2 = valid
            path1 = '{}/{}'.format(datadir, gen_filepath(name, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name, int(index2)))
            name1 = name; name2 = name
        else:               # 负样本
            name1, index1, name2, index2 = valid
            path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))

        x1a, y1a, x2a, y2a = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[path1])
        path_bbox_label = '{} {} {} {} {} {}\n'.format(path1, int(x1a), int(y1a), int(x2a), int(y2a), dict_name_label[name1])
        valid_list_gen += [path_bbox_label]
        x1a, y1a, x2a, y2a = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[path2])
        path_bbox_label = '{} {} {} {} {} {}\n'.format(path2, int(x1a), int(y1a), int(x2a), int(y2a), dict_name_label[name2])
        valid_list_gen += [path_bbox_label]

    with open(os.path.join(classify_similarity, 'lfw_classify_similarity_{}_valid.txt'.format(patch_index)), 'w') as f:
        f.writelines(valid_list_gen)



    ## testsets
    test_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view2', 'pairs.txt')
    with open(test_txt, 'r') as f:
        test_list = f.readlines()[1:]  # name index1 index2 or name1 index1 name2 index2

    test_list_gen = []
    for test in test_list:

        test = test.strip().split('\t')
        if len(test)==3:   # 正样本            
            name, index1, index2 = test
            path1 = '{}/{}'.format(datadir, gen_filepath(name, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name, int(index2)))
            name1 = name; name2 = name
        else:               # 负样本
            name1, index1, name2, index2 = test
            path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))

        x1a, y1a, x2a, y2a = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[path1])
        path_bbox_label = '{} {} {} {} {} {}\n'.format(path1, int(x1a), int(y1a), int(x2a), int(y2a), dict_name_label[name1])
        test_list_gen += [path_bbox_label]
        x1a, y1a, x2a, y2a = gen_aug_bbox(patch_index, ratio, dict_filepath_detect[path2])
        path_bbox_label = '{} {} {} {} {} {}\n'.format(path2, int(x1a), int(y1a), int(x2a), int(y2a), dict_name_label[name2])
        test_list_gen += [path_bbox_label]

    with open(os.path.join(classify_similarity, 'lfw_classify_similarity_{}_test.txt'.format(patch_index)), 'w') as f:
        f.writelines(test_list_gen)

