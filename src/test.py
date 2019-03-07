import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config  import configer
from dataset import DeepIdData
from utiles  import accuracy

def initModel():
    modelpath = os.path.join('../modelfile', '{}.pkl'.format(configer.modelname))
    assert os.path.exists(modelpath), 'please check model file! '
    model = torch.load(modelpath)
    return model

def test():
    testsets = DeepIdData(mode='test')
    testloader = DataLoader(testsets, configer.batchsize)
    model = initModel().eval()
    loss = nn.BCELoss()

    loss_test = []
    acc_test  = []

    for i_batch, (X1, X2, y) in enumerate(testloader):
        X1 = Variable(X1.float())
        X2 = Variable(X2.float())
        y_pred_prob = model(X1, X2)

        loss_test_batch = loss(y_pred_prob, y)
        acc_test_batch  = accuracy(y_pred_prob, y)

        print_log = 'testing...  batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                    format(i_batch+1, len(testsets)//configer.batchsize, acc_test_batch, loss_test_batch)
        print(print_log)

        loss_test += [loss_test_batch.detach().numpy()]
        acc_test  += [acc_test_batch.numpy()]

    loss_test = np.mean(loss_test)
    acc_test  = np.mean(acc_test )

    print_log = 'testing: accuracy: {:2.2%}, loss: {:4.4f}'.\
            format(acc_test, loss_test)
    print(print_log)