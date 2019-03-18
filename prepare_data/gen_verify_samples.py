import os
import numpy as np

get_name_from_filepath = lambda filepath: filepath.split('/')[-2]
gen_filepath = lambda name, index: '{}/{}_{:04d}.jpg'.format(name, name, index)

def gen_verify_pairs(datadir):
    
    verify_datadir = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_verify')
    if not os.path.exists(verify_datadir): os.mkdir(verify_datadir)

    classify_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_classify', 'lfw_classify_0.txt')
    with open(classify_txt, 'r') as f:
        classifies = f.readlines()
    dict_filepath_index = {classifies[i].split(' ')[0]: i for i in range(len(classifies))}

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
            label = 1
        else:               # 负样本
            name1, index1, name2, index2 = train
            path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))
            label = 0
        index1 = dict_filepath_index[path1]
        index2 = dict_filepath_index[path2]
        index1_index2_label = '{} {} {}\n'.format(index1, index2, label)
        train_list_gen += [index1_index2_label]

    with open(os.path.join(verify_datadir, 'lfw_verify_train.txt'), 'w') as f:
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
            label = 1
        else:               # 负样本
            name1, index1, name2, index2 = valid
            path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))
            label = 0
        index1 = dict_filepath_index[path1]
        index2 = dict_filepath_index[path2]
        index1_index2_label = '{} {} {}\n'.format(index1, index2, label)
        valid_list_gen += [index1_index2_label]

    with open(os.path.join(verify_datadir, 'lfw_verify_valid.txt'), 'w') as f:
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
            label = 1
        else:               # 负样本
            name1, index1, name2, index2 = test
            path1 = '{}/{}'.format(datadir, gen_filepath(name1, int(index1)))
            path2 = '{}/{}'.format(datadir, gen_filepath(name2, int(index2)))
            label = 0
        index1 = dict_filepath_index[path1]
        index2 = dict_filepath_index[path2]
        index1_index2_label = '{} {} {}\n'.format(index1, index2, label)
        test_list_gen += [index1_index2_label]

    with open(os.path.join(verify_datadir, 'lfw_verify_test.txt'), 'w') as f:
        f.writelines(test_list_gen)