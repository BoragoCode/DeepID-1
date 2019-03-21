import os
import numpy as np
from PIL import Image
from PIL import ImageDraw

get_name_from_filepath = lambda filepath: filepath.split('/')[-2]
gen_filepath = lambda name, index: '{}/{}_{:04d}.jpg'.format(name, name, index)











def gen_classify(datadir):
    """
    Notes:
        以检测结果，生成分类样本
    """
    ## 读取检测结果
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines() # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    detect_list = [detect.strip() for detect in detect_list]
    dict_filepath_detect = {detect.split(' ')[0]: ' '.join(detect.split(' ')[1: ]) \
                                            for detect in detect_list}
    
    ## 读取标签
    labels_txt =  os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_labels.txt')
    with open(labels_txt, 'r') as f:
        labels_list = f.readlines() # name label
    labels_list = [l.split(' ') for l in labels_list]
    dict_name_label = {name_label[0]: name_label[1].strip() for name_label in labels_list}
    
    ## 保存样本
    classify_dir = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_classify')
    if not os.path.exists(classify_dir): os.mkdir(classify_dir)
    classify_txt = os.path.join(classify_dir, 'lfw_classify.txt')

    ## 生成样本
    samples = []
    for filepath, detect in dict_filepath_detect.items():
        label  = dict_name_label[get_name_from_filepath(filepath)]
        sample = '{} {} {}\n'.format(filepath, detect, label)
        samples += [sample]

    with open(classify_txt, 'w') as f:
        f.writelines(samples)






def gen_classify_pairs(datadir):
    """
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
    classify_pairs = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_classify_pairs')
    if not os.path.exists(classify_pairs): os.mkdir(classify_pairs)


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

        path_detect_label = '{} {} {}\n'.format(path1, dict_filepath_detect[path1], dict_name_label[name1])
        train_list_gen += [path_detect_label]
        path_detect_label = '{} {} {}\n'.format(path2, dict_filepath_detect[path2], dict_name_label[name2])
        train_list_gen += [path_detect_label]

    with open(os.path.join(classify_pairs, 'train.txt'), 'w') as f:
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

        path_detect_label = '{} {} {}\n'.format(path1, dict_filepath_detect[path1], dict_name_label[name1])
        valid_list_gen += [path_detect_label]
        path_detect_label = '{} {} {}\n'.format(path2, dict_filepath_detect[path2], dict_name_label[name2])
        valid_list_gen += [path_detect_label]

    with open(os.path.join(classify_pairs, 'valid.txt'), 'w') as f:
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

        path_detect_label = '{} {} {}\n'.format(path1, dict_filepath_detect[path1], dict_name_label[name1])
        test_list_gen += [path_detect_label]
        path_detect_label = '{} {} {}\n'.format(path2, dict_filepath_detect[path2], dict_name_label[name2])
        test_list_gen += [path_detect_label]

    with open(os.path.join(classify_pairs, 'test.txt'), 'w') as f:
        f.writelines(test_list_gen)
