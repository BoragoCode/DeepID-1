import os
from os.path import join
import random
import numpy as np

get_intlabel_from_path = lambda path: int(path[path.find('DATA') + len('DATAx/'):].split('/')[0])
get_obtype_from_path = lambda path: path[path.find('Multi') + len('Multi/'): path.find('Multi') + len('Multi/non-obtructive/')]
is_pos = lambda pair: get_intlabel_from_path(pair.strip().split(' ')[0])==get_intlabel_from_path(pair.strip().split(' ')[1])

def gen_image_list(datadir):
    imgtxt = 'image_list.txt'
    f = open(imgtxt, 'w')
    for vol in os.listdir(datadir):
        datadir_vol = join(datadir, vol)
        for sess in os.listdir(datadir_vol):
            datadir_vol_sess_multi = join(datadir_vol, sess, 'Multi')
            for obtype in ['non-obtructive', 'obtructive/ob1', 'obtructive/ob2']:
                datadir_vol_sess_multi_obtype = join(datadir_vol_sess_multi, obtype)
                for pair in os.listdir(datadir_vol_sess_multi_obtype):
                    imgpath = join(datadir_vol_sess_multi_obtype, pair)
                    f.write(imgpath + '\n')
    f.close()

def gen_pair_samples(n_samples, pos=0.5):
    ## get image list
    imgtxt = 'image_list.txt'
    with open(imgtxt, 'r') as f:
        imglist = f.readlines()
    imglist = [pair.strip() for pair in imglist]
    n_imglist = len(imglist)

    ## split image list
    n_class = 33
    imglist_class = [[] for i in range(n_class)]
    for i in range(n_class):
        imglist_class[i] = [pair for pair in imglist if get_intlabel_from_path(pair)==(i+1)]

    ## generate pos samples
    poslist = []
    for i_pos in range(int(n_samples*pos)):
        classidx = np.random.randint(n_class)
        i, j = np.random.randint(len(imglist_class[classidx]), size=2)
        poslist += [imglist_class[classidx][i] + ' ' + imglist_class[classidx][j]]
    n_pos = len(poslist)

    ## generate neg samples
    neglist = []
    for i_neg in range(n_samples - n_pos):
        classi, classj = 0, 0
        while classi == classj:
            classi, classj = np.random.randint(n_class, size=2)
        i, j = np.random.randint(len(imglist_class[classi])), np.random.randint(len(imglist_class[classj]))
        neglist += [imglist_class[classi][i] + ' ' + imglist_class[classj][j]]
    n_neg = len(neglist)

    ## save
    imglist = poslist + neglist
    random.shuffle(imglist)
    pairtxt = 'pair_list.txt'
    with open(pairtxt, 'w') as f:
        imglist = [pair + '\n' for pair in imglist]
        f.writelines(imglist)

    ## statistic
    print('total samples: ', n_samples)
    print('pos samples: ', n_pos)
    print('neg samples: ', n_neg)

def gen_split(train=0.7, valid=0.2, test=0.1):
    pairtxt = 'pair_list.txt'
    with open(pairtxt, 'r') as f:
        imglist = f.readlines()
    
    n_samples = len(imglist)
    n_train = int(n_samples * train)
    n_valid = int(n_samples * valid)
    n_test  = n_samples - n_train - n_valid

    imglist_valid_test = random.sample(imglist, n_test + n_valid)
    imglist_test       = random.sample(imglist_valid_test, n_test)
    imglist_valid      = [pair for pair in imglist_valid_test if pair not in imglist_test]
    imglist_train      = [pair for pair in imglist if pair not in imglist_valid_test]

    n_pos = len([pair for pair in imglist_train if is_pos(pair)])
    print('train {} | pos: {}, neg: {}'.format(n_train, n_pos, len(imglist_train) - n_pos))
    n_pos = len([pair for pair in imglist_valid if is_pos(pair)])
    print('valid {} | pos: {}, neg: {}'.format(n_valid, n_pos, len(imglist_valid) - n_pos))
    n_pos = len([pair for pair in imglist_test if is_pos(pair)])
    print('test  {} | pos: {}, neg: {}'.format(n_test,  n_pos, len(imglist_test) - n_pos))

    with open('train.txt', 'w') as f:
        f.writelines(imglist_train)
    with open('valid.txt', 'w') as f:
        f.writelines(imglist_valid)
    with open('test.txt', 'w') as f:
        f.writelines(imglist_test)

if __name__ == "__main__":
    DATADIR = '/home/louishsu/Work/Workspace/ECUST2019_NPY'

    ## step 1. generate image lists
    # gen_image_list(DATADIR)

    ## step 2. generate pair samples
    # gen_pair_samples(200000)

    ## step 3. split dataset
    gen_split(0.7, 0.2, 0.1)