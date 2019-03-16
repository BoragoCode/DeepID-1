import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utiles   import getTime
from datasets import ClassifyDataset, ClassifyWithVerifyDataset
from metrics  import IdentifyLoss, TotalLoss, accuracy_mul, accuracy_bin
from models   import Classifier

def iniClassifyModel(in_channels, n_classes, modeldir):

    model = Classifier(in_channels, n_classes, modeldir)

    if os.path.exists(modeldir): 
        model.load()
    else:
        os.mkdir(modeldir)
        model.save()
    
    return model

def iniLogger(logdir):
    
    if not os.path.exists(logdir): os.mkdir(logdir)
    logger = SummaryWriter(logdir)

    return logger

def train_classify_only(configer):

    trainset    = ClassifyDataset(configer.patch, configer.imsize, configer.scale)
    trainloader = DataLoader(trainset, configer.batchsize, shuffle=True)
    model       = iniClassifyModel(configer.in_channels, configer.n_classes, configer.modeldir)
    if configer.cuda: model.cuda()
    metric      = IdentifyLoss()
    optimizer   = optim.Adam(model.parameters(), configer.lrbase,  betas=(0.9, 0.95), weight_decay=0.0005)
    scheduler   = lr_scheduler.StepLR(optimizer, configer.stepsize, configer.gamma)
    logger      = iniLogger(configer.logdir)

    cur_batch = 0
    elapsed_time = 0; total_time = 0


    for i_epoch in range(configer.n_epoch):

        if configer.cuda: torch.cuda.empty_cache()
        scheduler.step(i_epoch)

        start_time = time.time()
        
        model.train()
        for i_batch, (X, y_true) in enumerate(trainloader):
            

            # load data
            X = Variable(X.float())
            y_true = Variable(y_true)
            if configer.cuda: 
                X = X.cuda()
                y_true = y_true.cuda()

            # forward
            _, y_pred_prob = model(X)

            # calculate loss and accuracy
            loss_i = metric(y_pred_prob, y_true)
            acc_i  = accuracy_mul(y_pred_prob, y_true)

            # backward
            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()

            # time
            duration_time = time.time() - start_time
            start_time = time.time()
            elapsed_time += duration_time

            cur_batch += 1
            if cur_batch % 10 == 0:
                total_time = duration_time * configer.n_epoch * len(trainset) // configer.batchsize
                print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] | cur: [{:6d}] || lr: {:.6f} | loss_i: {:.4f} acc_i: {:.2%}".\
                    format(getTime(), elapsed_time/3600, (total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                            i_epoch, configer.n_epoch, i_batch, len(trainset) // configer.batchsize, cur_batch, 
                            scheduler.get_lr()[-1], loss_i, acc_i)
                print(print_log)
                model.save()

            # log
            logger.add_scalar('accuracy', acc_i,  cur_batch)
            logger.add_scalar('loss',     loss_i, cur_batch)
            logger.add_scalar('lr',       scheduler.get_lr()[-1], cur_batch)



def train_classify_with_verify(configer):
    modeldir = '_'.join(configer.modeldir.split('_')[: -2])
    assert os.path.exists(modeldir), "please train classify model first! "

    trainset    = ClassifyWithVerifyDataset(configer.patch, 'train', configer.imsize, configer.scale)
    trainloader = DataLoader(trainset, configer.batchsize // 2, shuffle=True)
    validset    = ClassifyWithVerifyDataset(configer.patch, 'valid', configer.imsize, configer.scale)
    validloader = DataLoader(validset, configer.batchsize // 2, shuffle=True)
    model       = iniClassifyModel(configer.in_channels, configer.n_classes, configer.modeldir)
    if configer.cuda: model.cuda()
    metric      = TotalLoss(configer.verify_weight)

    params  = [
            {'params': model.parameters(), 'lr': configer.lrbase * 0.1},
            {'params': metric.verify.parameters()},
    ]
    optimizer   = optim.Adam(params, configer.lrbase,  betas=(0.9, 0.95), weight_decay=0.0005)
    scheduler   = lr_scheduler.StepLR(optimizer, configer.stepsize, configer.gamma)
    logger      = iniLogger(configer.logdir)

    cur_batch = 0
    elapsed_time = 0; total_time = 0
    valid_loss_last = float('inf')

    for i_epoch in range(configer.n_epoch):

        if configer.cuda: torch.cuda.empty_cache()
        scheduler.step(i_epoch)

        ident_epoch_train = []
        verif_epoch_train = []
        total_epoch_train = []
        acc_epoch_train   = []
        
        start_time = time.time()

        model.train()
        for i_batch, (y, X1, y1_true, X2, y2_true) in enumerate(trainloader):
            

            # load data
            y  = Variable(y)
            X1 = Variable(X1.float())
            y1_true = Variable(y1_true)
            X2 = Variable(X2.float())
            y2_true = Variable(y2_true)
            if configer.cuda: 
                y = y.cuda()
                X1 = X1.cuda()
                y1_true = y1_true.cuda()
                X2 = X2.cuda()
                y2_true = y2_true.cuda()

            # forward
            X1, y1_pred_prob = model(X1)
            X2, y2_pred_prob = model(X2)

            # calculate loss and accuracy
            ident_i, verif_i, total_i = metric(X1, X2, y1_pred_prob, y2_pred_prob, y1_true, y2_true)
            acc_i  = (accuracy_mul(y1_pred_prob, y1_true) + accuracy_mul(y2_pred_prob, y2_true)) / 2

            ident_epoch_train += [ident_i.detach().cpu().numpy()]
            verif_epoch_train += [verif_i.detach().cpu().numpy()]
            total_epoch_train += [total_i.detach().cpu().numpy()]
            acc_epoch_train   += [acc_i.cpu().numpy()]

            # backward
            optimizer.zero_grad()
            total_i.backward()
            optimizer.step()

            # time
            duration_time = time.time() - start_time
            start_time = time.time()
            elapsed_time += duration_time

            cur_batch += 1
            if cur_batch % 10 == 0:
                total_time = duration_time * configer.n_epoch * len(trainset) // configer.batchsize
                print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] | cur: [{:6d}] || lr: {:.6f} | ident_i: {:.3f}, verif_i: {:.3f}, total_i: {:.3f} | acc_i: {:.2%}".\
                            format(getTime(), elapsed_time/3600, 2*(total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                                    i_epoch, configer.n_epoch, i_batch, len(trainset) // configer.batchsize, cur_batch, 
                                    scheduler.get_lr()[-1], ident_i, verif_i, total_i, acc_i)
                print(print_log)

        ident_epoch_train = np.mean(np.array(ident_epoch_train))
        verif_epoch_train = np.mean(np.array(verif_epoch_train))
        total_epoch_train = np.mean(np.array(total_epoch_train))
        acc_epoch_train   = np.mean(np.array(acc_epoch_train))

        print('-----------------------------------------------------------------------------------------------------')

        ident_epoch_valid = []
        verif_epoch_valid = []
        total_epoch_valid = []
        acc_epoch_valid   = []

        model.eval()
        for i_batch, (y, X1, y1_true, X2, y2_true) in enumerate(validloader):

            y  = Variable(y)
            X1 = Variable(X1.float())
            y1_true = Variable(y1_true)
            X2 = Variable(X2.float())
            y2_true = Variable(y2_true)
            if configer.cuda: 
                y = y.cuda()
                X1 = X1.cuda()
                y1_true = y1_true.cuda()
                X2 = X2.cuda()
                y2_true = y2_true.cuda()

            X1, y1_pred_prob = model(X1)
            X2, y2_pred_prob = model(X2)

            ident_i, verif_i, total_i = metric(X1, X2, y1_pred_prob, y2_pred_prob, y1_true, y2_true)
            acc_i  = (accuracy_mul(y1_pred_prob, y1_true) + accuracy_mul(y2_pred_prob, y2_true)) / 2

            ident_epoch_valid += [ident_i.detach().cpu().numpy()]
            verif_epoch_valid += [verif_i.detach().cpu().numpy()]
            total_epoch_valid += [total_i.detach().cpu().numpy()]
            acc_epoch_valid   += [acc_i.cpu().numpy()]

            if i_batch % 10 == 0:
                print_log = "{} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || ident_i: {:.3f}, verif_i: {:.3f}, total_i: {:.3f} | acc_i: {:.2%}".\
                            format(getTime(), i_epoch, configer.n_epoch, i_batch, len(validset) // configer.batchsize, 
                                    ident_i, verif_i, total_i, acc_i)
                print(print_log)
        

        ident_epoch_valid = np.mean(np.array(ident_epoch_valid))
        verif_epoch_valid = np.mean(np.array(verif_epoch_valid))
        total_epoch_valid = np.mean(np.array(total_epoch_valid))
        acc_epoch_valid   = np.mean(np.array(acc_epoch_valid))

        print('-----------------------------------------------------------------------------------------------------')
        
        print_log = "{} || Epoch: [{:3d}]/[{:3d}] || train | ident_i: {:.3f}, verif_i: {:.3f}, total_i: {:.3f} | acc_i: {:.2%} || valid | ident_i: {:.3f}, verif_i: {:.3f}, total_i: {:.3f} | acc_i: {:.2%}".\
                            format(getTime(), i_epoch, configer.n_epoch, 
                                    ident_epoch_train, verif_epoch_train, total_epoch_train, acc_epoch_train,
                                    ident_epoch_valid, verif_epoch_valid, total_epoch_valid, acc_epoch_valid)
        print(print_log)

        logger.add_scalars('identify loss', {'train': ident_epoch_train, 'valid': ident_epoch_valid}, i_epoch)
        logger.add_scalars('verify loss',   {'train': verif_epoch_train, 'valid': verif_epoch_valid}, i_epoch)
        logger.add_scalars('total loss',    {'train': total_epoch_train, 'valid': total_epoch_valid}, i_epoch)
        logger.add_scalars('accuracy',      {'train': acc_epoch_train,   'valid': acc_epoch_valid},   i_epoch)
        logger.add_scalar('lr', scheduler.get_lr()[-1], i_epoch)

        print('-----------------------------------------------------------------------------------------------------')
        
        if total_epoch_valid < valid_loss_last:
            valid_loss_last = total_epoch_valid
            model.save()

        print('=====================================================================================================')