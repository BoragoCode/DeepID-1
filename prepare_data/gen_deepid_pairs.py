import os
import numpy as np

get_name_from_filepath = lambda filepath: filepath.split('/')[-2]
gen_filepath = lambda name, index: '{}/{}_{:04d}.jpg'.format(name, name, index)

def gen_deepid_pair_samples(datadir):

    # 读取检测结果
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines() # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    detect_list = [detect.strip() for detect in detect_list]
    dict_filepath_detect = {detect.split(' ')[0]: ' '.join(detect.split(' ')[1: ]) \
                                            for detect in detect_list}

    # 保存结果文件夹
    deepid_pair = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_deepid_pair')
    if not os.path.exists(deepid_pair): os.mkdir(deepid_pair)

    
    ## trainsets
    train_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'view1', 'pairsDevTrain.txt')
    with open(train_txt, 'r') as f:
        train_list = f.readlines()[1:]  # name index1 index2 or name1 index1 name2 index2
    
    train_list_gen = []
    for train in train_list:

        train = train.strip().split('\t')
        if len(train)==3:   # 正样本            
            name, index1, index2 = train
            name1 = name; name2 = name
        else:               # 负样本
            name1, index1, name2, index2 = train
        path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
        path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))

        path_detect = '{} {}\n'.format(path1, dict_filepath_detect[path1])
        train_list_gen += [path_detect]
        path_detect = '{} {}\n'.format(path2, dict_filepath_detect[path2])
        train_list_gen += [path_detect]

    with open(os.path.join(deepid_pair, 'lfw_deepid_pair_train.txt'), 'w') as f:
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
            name1 = name; name2 = name
        else:               # 负样本
            name1, index1, name2, index2 = valid
        path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
        path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))

        path_detect = '{} {}\n'.format(path1, dict_filepath_detect[path1])
        valid_list_gen += [path_detect]
        path_detect = '{} {}\n'.format(path2, dict_filepath_detect[path2])
        valid_list_gen += [path_detect]

    with open(os.path.join(deepid_pair, 'lfw_deepid_pair_valid.txt'), 'w') as f:
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
            name1 = name; name2 = name
        else:               # 负样本
            name1, index1, name2, index2 = test
        path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
        path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))

        path_detect = '{} {}\n'.format(path1, dict_filepath_detect[path1])
        test_list_gen += [path_detect]
        path_detect = '{} {}\n'.format(path2, dict_filepath_detect[path2])
        test_list_gen += [path_detect]

    with open(os.path.join(deepid_pair, 'lfw_deepid_pair_test.txt'), 'w') as f:
        f.writelines(test_list_gen)