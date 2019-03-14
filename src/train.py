import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utiles   import getTime
from datasets import ClassifyDataset
from metrics  import IdentifyLoss, TotalLoss, accuracy_mul, accuracy_bin
from models   import Classifier

def iniClassifyModel(in_channels, n_classes, modeldir):

    model = Classifier(in_channels, n_classes, modeldir)

    if os.path.exists(modeldir): 
        model.load()
    else:
        os.mkdir(modeldir)
        model.save()
    
    if torch.cuda.is_available():
        model.cuda()
    
    return model

def iniLogger(logdir):
    
    if not os.path.exists(logdir): os.mkdir(logdir)
    logger = SummaryWriter(logdir)

    return logger

def train_classify_only(configer):

    trainset    = ClassifyDataset(configer.patch, configer.imsize, configer.scale)
    trainloader = DataLoader(trainset, configer.batchsize, shuffle=True)
    model       = iniClassifyModel(configer.in_channels, configer.n_classes, configer.modeldir)
    metric      = IdentifyLoss()
    optimizer   = optim.Adam(model.parameters(), configer.lrbase,  betas=(0.9, 0.95), weight_decay=0.0005)
    scheduler   = lr_scheduler.StepLR(optimizer, configer.stepsize, configer.gamma)
    logger      = iniLogger(configer.logdir)

    cur_batch = 0
    elapsed_time = 0; total_time = 0


    for i_epoch in range(configer.n_epoch):

        scheduler.step(i_epoch)

        model.train()
        for i_batch, (X, y_true) in enumerate(trainloader):
            
            start_time = time.time()

            # load data
            X = Variable(X.float())
            y_true = Variable(y_true)
            if torch.cuda.is_available(): X.cuda(); y_true.cuda()

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
            elapsed_time += duration_time

            # validating
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
    
    assert os.path.exists(configer.modeldir), "please train classify model first! "

    
