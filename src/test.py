import os
import time
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utiles import getTime
from datasets import ClassifyDataset
from metrics import IdentifyLoss, accuracy_mul
from models import Classifier

def iniClassifyModel(in_channels, n_classes, modeldir):

    assert os.path.exists(modeldir), 'please train first! '

    model = Classifier(in_channels, n_classes, modeldir)
    model.load()
    return model

def test_classify_only(configer, save_features=False):

    testset     = ClassifyDataset(configer.patch, configer.imsize, configer.scale)
    testloader  = DataLoader(testset, configer.batchsize, shuffle=False)
    model       = iniClassifyModel(configer.in_channels, configer.n_classes, configer.modeldir)
    if configer.cuda: model.cuda()
    metric      = IdentifyLoss() 

    elapsed_time = 0; total_time = 0
    start_time = time.time()

    loss = []
    acc  = []

    if save_features: 
        features = None
        features_dir = '../data/lfw_classify/features'
        if not os.path.exists(features_dir): os.mkdir(features_dir)
        features_path = '{}/lfw_classify_{}_scale{}.npy'.\
                                    format(features_dir, configer.patch, configer.scale)

    model.eval()
    for i_batch, (X, y_true) in enumerate(testloader):

        # load data
        X = Variable(X.float())
        y_true = Variable(y_true)
        if configer.cuda: 
            X = X.cuda()
            y_true = y_true.cuda()
        
        # forward
        X, y_pred_prob = model(X)

        # save feature
        if save_features:
            X = X.detach().numpy()
            if features is None:
                features = X
            else:
                features = np.concatenate([features, X], axis=0)

        # calculate loss and accuracy
        loss_i = metric(y_pred_prob, y_true)
        acc_i  = accuracy_mul(y_pred_prob, y_true)
        loss += [loss_i.detach().cpu().numpy()]
        acc  += [acc_i.cpu().numpy()]

        # time
        duration_time = time.time() - start_time
        start_time = time.time()
        elapsed_time += duration_time
        total_time = duration_time * len(testset) // configer.batchsize
       
        # print
        print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Batch: [{:3d}]/[{:3d}] || loss_i: {:.4f} acc_i: {:.2%}".\
            format(getTime(), elapsed_time/3600, (total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                    i_batch, len(testset) // configer.batchsize, 
                    loss_i, acc_i)
        print(print_log)

    loss = np.mean(np.array(loss))
    acc  = np.mean(np.array(acc))
    print_log = "{} || Elapsed: {:.4f}h || loss_i: {:.4f} acc_i: {:.2%}".\
        format(getTime(), elapsed_time/3600, loss, acc)
    print(print_log)

    np.save(features_path, features)