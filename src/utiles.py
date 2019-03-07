import os
import torch
import time

get_intlabel_from_path = lambda path: int(path[path.find('DATA') + len('DATAx/'):].split('/')[0])
get_intposit_from_path = lambda path: int(path[path.find('Multi_') + len('Multi_'):].split('_')[0])

is_same_label = lambda path1, path2: get_intlabel_from_path(path1) == get_intlabel_from_path(path2)
is_same_posit = lambda path1, path2: get_intposit_from_path(path1) == get_intposit_from_path(path2)

get_time = lambda: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def accuracy(y_pred, y_true):
    y_pred[y_pred >  0.5] = 1.0
    y_pred[y_pred <= 0.5] = 0.0
    return torch.mean(y_pred == y_true)