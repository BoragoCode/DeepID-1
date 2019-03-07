import os
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config  import configer
from dataset import DeepIdData
from models  import VGGFeatures, DeepIdModel
from utiles  import accuracy, get_time

def initLog():
    logdir = os.path.join('../logfile', configer.modelname)
    if not os.path.exists(logdir): os.mkdir(logdir)
    logger = SummaryWriter(logdir)
    return logger

def initModel():
    modelpath = os.path.join('../modelfile', '{}.pkl'.format(configer.modelname))
    if os.path.exists(modelpath):
        model = torch.load(modelpath)
    else:
        model = DeepIdModel(
            lambda in_channels, out_features: VGGFeatures(in_channels, out_features, configer.modelbase), 
            configer.n_channels,  configer.n_features
            )
        torch.save(model, modelpath)
    return model, modelpath

def train():
    trainsets = DeepIdData(mode='train')
    trainloader = DataLoader(trainsets, configer.batchsize, shuffle=True)
    validsets = DeepIdData(mode='valid')
    validloader = DataLoader(validsets, configer.batchsize)
    logger = initLog()
    model,  modelpath = initModel()
    loss = nn.BCELoss()
    optimizor = optim.Adam(model.parameters(), configer.learningrate, betas=(0.9, 0.95), weight_decay=0.0005)

    acc_train = 0.; acc_valid = 0.
    loss_train = float('inf')
    loss_valid = float('inf')
    loss_valid_last = float('inf')


    for i_epoch in range(configer.n_epoch):

        acc_train = []; acc_valid = []
        loss_train = []; loss_valid = []


        model.train()
        for i_batch, (X1, X2, y) in enumerate(trainloader):
            X1 = Variable(X1.float())
            X2 = Variable(X2.float())
            y  = y.float()

            y_pred_prob = model(X1, X2)
            loss_train_batch = loss(y_pred_prob, y)
            optimizor.zero_grad()
            loss_train_batch.backward() 
            optimizor.step()

            acc_train_batch  = accuracy(y_pred_prob, y)
            print_log = '{} || training...    epoch [{:3d}]/[{:3d}] | batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(get_time(), i_epoch+1, configer.n_epoch, i_batch+1, len(trainsets)//configer.batchsize, acc_train_batch, loss_train_batch)
            print(print_log)

            acc_train.append(acc_train_batch.numpy())
            loss_train.append(loss_train_batch.detach().numpy())
        
        
        model.eval()
        for i_batch, (X1, X2, y) in enumerate(validloader):
            X1 = Variable(X1.float())
            X2 = Variable(X2.float())
            y_pred_prob = model(X1, X2)

            loss_valid_batch = loss(y_pred_prob, y)
            acc_valid_batch  = accuracy(y_pred_prob, y)
            print_log = '{} || validating...  epoch [{:3d}]/[{:3d}] | batch [{:2d}]/[{:2d}] || accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(get_time(), i_epoch+1, configer.n_epoch, i_batch+1, len(validsets)//configer.batchsize, acc_valid_batch, loss_valid_batch)
            print(print_log)

            acc_valid.append(acc_valid_batch.numpy())
            loss_valid.append(loss_valid_batch.detach().numpy())
        

        
        acc_train = np.mean(np.array(acc_train))
        loss_train = np.mean(np.array(loss_train))
        acc_valid = np.mean(np.array(acc_valid))
        loss_valid = np.mean(np.array(loss_valid))

        logger.add_scalars('accuracy', {'train': acc_train,  'valid': acc_valid},  i_epoch)
        logger.add_scalars('logloss',  {'train': loss_train, 'valid': loss_valid}, i_epoch)

        print_log = '--------------------------------------------------------------------'
        print(print_log)
        print_log = '{} || epoch [{:3d}]/[{:3d}] || training: accuracy: {:2.2%}, loss: {:4.4f} | validing: accuracy: {:2.2%}, loss: {:4.4f}'.\
                        format(get_time(), i_epoch, configer.n_epoch, acc_train, loss_train, acc_valid, loss_valid)
        print(print_log)


        if loss_valid_last > loss_valid:
            torch.save(model, modelpath)
            loss_valid_last = loss_valid
            print_log = '{} || model saved @ {}'.format(get_time(), modelpath)
            print(print_log)


        print_log = '===================================================================='
        print(print_log)