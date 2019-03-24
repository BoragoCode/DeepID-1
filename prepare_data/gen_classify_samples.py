import os
import random
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


def gen_classify_celeba(datadir):
    """
    total: 10177
    use 80%(8000) to train classify models
    """
    ## 读取检测结果
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'celeba_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines() # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    detect_list = [detect.strip() for detect in detect_list]
    dict_filepath_detect = {detect.split(' ')[0]: ' '.join(detect.split(' ')[1: ]) \
                                            for detect in detect_list}

    ## 读取标签
    labels_txt =  os.path.join('/'.join(datadir.split('/')[:-1]), 'Anno/identity_CelebA.txt')
    with open(labels_txt, 'r') as f:
        list_img_label = f.readlines()
    dict_path_label = {'{}/{}'.format(datadir, img_label.strip().split(' ')[0]): int(img_label.strip().split(' ')[1]) \
                        for img_label in list_img_label}
    dict_path_label = {path: label for path, label in dict_path_label.items() if label <= 8000}

    ## 保存文件
    classify_dir = os.path.join('/'.join(datadir.split('/')[:-1]), 'celeba_classify')
    if not os.path.exists(classify_dir): os.mkdir(classify_dir)
    classify_txt = os.path.join(classify_dir, 'celeba_classify.txt')
    f = open(classify_txt, 'w')

    ## 生成样本
    for filepath, label in dict_path_label.items():

        detect = dict_filepath_detect[filepath]
        sample = '{} {} {}\n'.format(filepath, detect, label)
        f.write(sample)
    
    f.close()


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
    train_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view2', 'pairs.txt')
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
    test_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view1', 'pairsDevTrain.txt')
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


def gen_classify_pairs_celeba(datadir):
    """
    total: 10177
    use 20+%(2177) to train deepid model
    """
    ## 读取检测结果
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'celeba_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines() # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    detect_list = [detect.strip() for detect in detect_list]
    dict_filepath_detect = {detect.split(' ')[0]: ' '.join(detect.split(' ')[1: ]) \
                                            for detect in detect_list}

    ## 读取标签
    labels_txt =  os.path.join('/'.join(datadir.split('/')[:-1]), 'Anno/identity_CelebA.txt')
    with open(labels_txt, 'r') as f:
        list_img_label = f.readlines()
    dict_path_label = {'{}/{}'.format(datadir, img_label.strip().split(' ')[0]): int(img_label.strip().split(' ')[1]) \
                        for img_label in list_img_label}
    dict_path_label = {path: label for path, label in dict_path_label.items() if label > 8000}

    ## 统计文件
    dict_label_pathlist = {label: [] for label in list(set(dict_path_label.values()))}
    for filepath, label in dict_path_label.items():
        dict_label_pathlist[label] += [filepath]
    
    ## 统计数量
    n_file  = len(dict_path_label)
    n_label = len(dict_label_pathlist)

    ## 保存文件
    classify_dir = os.path.join('/'.join(datadir.split('/')[:-1]), 'celeba_classify_pairs')
    if not os.path.exists(classify_dir): os.mkdir(classify_dir)

    ## 训练集
    n_train = 10000
    train_txt = os.path.join(classify_dir, 'train.txt')
    ftrain = open(train_txt, 'w')
    for i_train in range(n_train):
        if np.random.randn(1) > 0.5:
            ### 正样本: 同类标签随机选取两个
            pathlist = random.sample(dict_label_pathlist, 1)
            path1, path2 = random.sample(pathlist, 2)
        else:
            ### 负样本
            pathlist1, pathlist2 = random.sample(dict_label_pathlist, 2)
            path1 = random.sample(pathlist1, 1)
            path2 = random.sample(pathlist2, 1)
        
        label1 = dict_path_label[path1]
        label2 = dict_path_label[path2]
        detect1 = dict_filepath_detect[path1]
        detect2 = dict_filepath_detect[path2]
        samplepair = '{} {} {}\n{} {} {}\n'.format(path1, detect1, label1, path2, detect2, label2)
        ftrain.write(samplepair)
    ftrain.close()

    ## 验证集
    n_valid = 3000
    valid_txt = os.path.join(classify_dir, 'valid.txt')
    fvalid = open(valid_txt, 'w')
    for i_valid in range(n_valid):
        if np.random.randn(1) > 0.5:
            ### 正样本: 同类标签随机选取两个
            pathlist = random.sample(dict_label_pathlist, 1)
            path1, path2 = random.sample(pathlist, 2)
        else:
            ### 负样本: 不同标签选取两个
            pathlist1, pathlist2 = random.sample(dict_label_pathlist, 2)
            path1 = random.sample(pathlist1, 1)
            path2 = random.sample(pathlist2, 1)
        
        label1 = dict_path_label[path1]
        label2 = dict_path_label[path2]
        detect1 = dict_filepath_detect[path1]
        detect2 = dict_filepath_detect[path2]
        samplepair = '{} {} {}\n{} {} {}\n'.format(path1, detect1, label1, path2, detect2, label2)
        fvalid.write(samplepair)
    fvalid.close()

    ## 测试集
    n_test = 1500
    test_txt = os.path.join(classify_dir, 'test.txt')
    ftest  = open(test_txt, 'w')
    for i_test in range(n_test):
        if np.random.randn(1) > 0.5:
            ### 正样本: 同类标签随机选取两个
            pathlist = random.sample(dict_label_pathlist, 1)
            path1, path2 = random.sample(pathlist, 2)
        else:
            ### 负样本: 不同标签选取两个
            pathlist1, pathlist2 = random.sample(dict_label_pathlist, 2)
            path1 = random.sample(pathlist1, 1)
            path2 = random.sample(pathlist2, 1)
        
        label1 = dict_path_label[path1]
        label2 = dict_path_label[path2]
        detect1 = dict_filepath_detect[path1]
        detect2 = dict_filepath_detect[path2]
        samplepair = '{} {} {}\n{} {} {}\n'.format(path1, detect1, label1, path2, detect2, label2)
        ftest.write(samplepair)
    ftest.close()
