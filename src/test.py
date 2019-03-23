import os
import time
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utiles import getTime
from datasets import ClassifyDataset, ClassifyPairsDataset, DeepIdDataset
from metrics import IdentifyLoss, VerifyBinLoss, TotalLoss, accuracy_mul, accuracy_bin
from models import Classifier, DeepID


def test_deepid_net(configer):

    testset = DeepIdDataset('test')
    testloader = DataLoader(testset, configer.batchsize, shuffle=False)
    model = DeepID(configer.in_channels, prefix='../modelfile/classify')
    if configer.cuda: model.cuda()
    metric = VerifyBinLoss()

    loss_test = []; acc_test = []
    model.eval()
    for i_batch, (X0, X1, X2, X3, X4, X5, X6, X7, X8, y_true) in enumerate(testloader):

        ## load data
        X0 = Variable(X0.float())
        X1 = Variable(X1.float())
        X2 = Variable(X2.float())
        X3 = Variable(X3.float())
        X4 = Variable(X4.float())
        X5 = Variable(X5.float())
        X6 = Variable(X6.float())
        X7 = Variable(X7.float())
        X8 = Variable(X8.float())
        y_true = Variable(y_true.float())
        if configer.cuda: 
            X0 = X0.cuda()
            X1 = X1.cuda()
            X2 = X2.cuda()
            X3 = X3.cuda()
            X4 = X4.cuda()
            X5 = X5.cuda()
            X6 = X6.cuda()
            X7 = X7.cuda()
            X8 = X8.cuda()
            y_true = y_true.cuda()

        ## forward
        y_pred = model(X0, X1, X2, X3, X4, X5, X6, X7, X8)

        ## calculate loss
        loss_i = metric(y_pred, y_true)

        ## calculate accuracy
        acc_i  = accuracy_bin(y_pred, y_true)

        ## log
        print_log = "{} || Batch: [{:3d}]/[{:3d}] || loss_i: {:.4f} acc_i: {:.2%} p: {:.4f}".\
            format(getTime(), i_batch, len(testset) // configer.batchsize, loss_i, acc_i, torch.exp(-loss_i))
        print(print_log)

        ## save
        loss_test += [loss_i.detach().cpu().numpy()]
        acc_test  += [acc_i.cpu().numpy()]

    loss_test = np.mean(np.array(loss_test))
    acc_test = np.mean(np.array(acc_test))

    print_log = "{} || loss: {:.4f} acc: {:.2%} p: {:.4f}".format(getTime(), loss_test, acc_test, np.exp(-loss_test))
    print(print_log)