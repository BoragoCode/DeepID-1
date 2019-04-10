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
from datasets import ClassifyDataset, ClassifyPairsDataset, DeepIdDataset
from metrics  import IdentifyLoss, TotalLoss, VerifyBinLoss, accuracy_mul, accuracy_bin
from models   import Classifier, DeepID



def iniLogger(logdir):
    
    if not os.path.exists(logdir): os.makedirs(logdir)
    logger = SummaryWriter(logdir)

    return logger



# ------------------------------------------------------------------------------------------------------------
def train_classify_only(configer):

    trainset    = ClassifyDataset(configer.patch, configer.scale)
    trainloader = DataLoader(trainset, configer.batchsize, shuffle=True)
    model       = Classifier(configer.in_channels, configer.n_classes)
    if configer.cuda: model.cuda()
    metric      = IdentifyLoss()
    optimizer   = optim.Adam(model.parameters(), configer.lrbase)
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

            # log
            cur_batch += 1
            if cur_batch % 20 == 0:
                total_time = duration_time * configer.n_epoch * len(trainset) // configer.batchsize
                print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] | cur: [{:6d}] || lr: {:.6f} | loss_i: {:.4f} acc_i: {:.2%}".\
                    format(getTime(), elapsed_time/3600, (total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                            i_epoch, configer.n_epoch, i_batch, len(trainset) // configer.batchsize, cur_batch, 
                            scheduler.get_lr()[-1], loss_i, acc_i)
                print(print_log)
                model.save('../modelfile/celeba_classify/{}'.format(configer.modelname))

            logger.add_scalar('accuracy', acc_i,  cur_batch)
            logger.add_scalar('loss',     loss_i, cur_batch)
            logger.add_scalar('lr',       scheduler.get_lr()[-1], cur_batch)




# ------------------------------------------------------------------------------------------------------------
def train_classify_similarity(configer, finetune=False):

    trainset    = ClassifyPairsDataset(configer.patch, configer.scale, 'train')
    trainloader = DataLoader(trainset, configer.batchszie, shuffle=True)
    validset    = ClassifyPairsDataset(configer.patch, configer.scale, 'valid')
    validloader = DataLoader(validset, configer.batchszie, shuffle=False)
    model       = Classifier(configer.in_channels, configer.n_classes, '../modelfile/classify_similarity/{}'.format(configer.modelname))
    if finetune: model.load(prefix='../modelfile/classify/{}'.format(configer.modelname))
    if configer.cuda: model.cuda()
    metric      = TotalLoss(k=0.01)
    optimizer   = optim.Adam(model.parameters(), configer.lrbase)
    scheduler   = lr_scheduler.StepLR(optimizer, configer.stepsize, configer.gamma)
    logger      = iniLogger(configer.logdir)
    
    elapsed_time = 0; total_time = 0

    for i_epoch in range(configer.n_epoch):

        if configer.cuda: torch.cuda.empty_cache()
        scheduler.step(i_epoch)

        ident_train = []
        similar_train = []
        total_train = []
        acc_train = []

        start_time = time.time()
        model.train()
        for i_batch, (X1, y1_true, X2, y2_true) in enumerate(trainloader):

            ## load data
            X1 = Variable(X1.float())
            y1_true = Variable(y1_true)
            X2 = Variable(X2.float())
            y2_true = Variable(y2_true)
            if configer.cuda: 
                X1 = X1.cuda()
                X2 = X2.cuda()
                y1_true = y1_true.cuda()
                y2_true = y2_true.cuda()

            ## forward
            X1, y1_pred_prob = model(X1)
            X2, y2_pred_prob = model(X2)

            ## calculate loss
            ident_i, similar_i, total_i = metric(X1, X2, y1_pred_prob, y2_pred_prob, y1_true, y2_true)

            ## backward
            optimizer.zero_grad()
            total_i.backward()
            optimizer.step()

            ## calculate accuracy
            acc_i = (accuracy_bin(y1_pred_prob, y1_true) + accuracy_bin(y2_pred_prob, y2_true)) / 2

            ## time
            duration_time = time.time() - start_time
            start_time = time.time()
            elapsed_time += duration_time
            total_time = duration_time * configer.n_epoch * len(trainset) // configer.batchsize

            ## log
            print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || lr: {:.6f} | ident_i: {:.4f} similar_i: {:.4f} total_i: {:.4f} acc_i: {:.2%}".\
                    format(getTime(), elapsed_time/3600, (total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                            i_epoch, configer.n_epoch, i_batch, len(trainset) // configer.batchsize, 
                            scheduler.get_lr()[-1], ident_i, similar_i, total_i, acc_i)
            print(print_log)


            ## save
            ident_train += [ident_i.detach().cpu().numpy()]
            similar_train = [similar_i.detach().cpu().numpy()]
            total_train = [total_i.detach().cpu().numpy()]
            acc_train = [acc_i.cpu().numpy()]

        ident_train = np.mean(np.array(ident_train))
        similar_train = np.mean(np.array(similar_train))
        total_train = np.mean(np.array(total_train))
        acc_train  = np.mean(np.array(acc_train))

        print('-----------------------------------------------------------------------------------------------------')

        ident_valid = []
        similar_valid = []
        total_valid = []
        acc_valid = []

        model.eval()
        for i_batch, (X1, y1_true, X2, y2_true) in enumerate(validloader):

            ## load data
            X1 = Variable(X1.float())
            y1_true = Variable(y1_true)
            X2 = Variable(X2.float())
            y2_true = Variable(y2_true)
            if configer.cuda: 
                X1 = X1.cuda()
                X2 = X2.cuda()
                y1_true = y1_true.cuda()
                y2_true = y2_true.cuda()

            ## forward
            X1, y1_pred_prob = model(X1)
            X2, y2_pred_prob = model(X2)

            ## calculate loss
            ident_i, similar_i, total_i = metric(X1, X2, y1_pred_prob, y2_pred_prob, y1_true, y2_true)

            ## calculate accuracy
            acc_i = (accuracy_bin(y1_pred_prob, y1_true) + accuracy_bin(y2_pred_prob, y2_true)) / 2

            ## log
            print_log = "{} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || lr: {:.6f} | ident_i: {:.4f} similar_i: {:.4f} total_i: {:.4f} acc_i: {:.2%}".\
                    format(getTime(), i_epoch, configer.n_epoch, i_batch, len(validset) // configer.batchsize, 
                            scheduler.get_lr()[-1], ident_i, similar_i, total_i, acc_i)
            print(print_log)

            ## save
            ident_valid += [ident_i.detach().cpu().numpy()]
            similar_valid = [similar_i.detach().cpu().numpy()]
            total_valid = [total_i.detach().cpu().numpy()]
            acc_valid = [acc_i.cpu().numpy()]

        ident_valid = np.mean(np.array(ident_valid))
        similar_valid = np.mean(np.array(similar_valid))
        total_valid = np.mean(np.array(total_valid))
        acc_valid  = np.mean(np.array(acc_valid))

        print('-----------------------------------------------------------------------------------------------------')


        logger.add_scalars('identity loss',     {'train': ident_train,  'valid': ident_valid},      i_epoch)
        logger.add_scalars('similarity loss',   {'train': similar_train,'valid': similar_valid},    i_epoch)
        logger.add_scalars('total loss',        {'train': total_train,  'valid': total_valid},      i_epoch)
        logger.add_scalars('accuracy',          {'train': acc_train,    'valid': acc_valid},        i_epoch)


        print_log = "{} || Epoch: [{:3d}]/[{:3d}] || lr: {:.6f} || train | ident: {:.4f} similar: {:.4f} total: {:.4f} acc: {:.2%} | valid | ident: {:.4f} similar: {:.4f} total: {:.4f} acc: {:.2%}".\
            format(getTime(), i_epoch, configer.n_epoch, scheduler.get_lr()[-1], 
                    ident_train, similar_train, total_train, acc_train,
                    ident_valid, similar_valid, total_valid, acc_valid)
        print(print_log)

        print('=====================================================================================================')





# ------------------------------------------------------------------------------------------------------------
def train_deepid_net(configer):

    trainset = DeepIdDataset('train', database='celeba')
    trainloader = DataLoader(trainset, configer.batchsize, shuffle=True)
    validset = DeepIdDataset('valid', database='celeba')
    validloader = DataLoader(validset, configer.batchsize, shuffle=False)
    model = DeepID(configer.in_channels, prefix='../modelfile/celeba_classify')
    if configer.cuda: model.cuda()
    metric = VerifyBinLoss()

    params = [{
                'params': features.parameters(), 
                'lr': configer.lrbase * configer.finetune_lr,
                }\
            for features in model.features.values()]
    params += [{'params': model.verifier.parameters()}]
    optimizer   = optim.Adam(params, configer.lrbase, weight_decay=0.001)
    
    scheduler   = lr_scheduler.StepLR(optimizer, configer.stepsize, configer.gamma)
    logger      = iniLogger('../logfile/deepid_lr_{}'.format(configer.finetune_lr))




    start_time = 0; duration_time = 0; elapsed_time = 0; total_time = 0
    loss_valid_last = float('inf')

    for i_epoch in range(configer.n_epoch):

        if configer.cuda: torch.cuda.empty_cache()
        scheduler.step(i_epoch)

        loss_train = []; acc_train = []
        model.train()
        start_time = time.time()
        for i_batch, (X0, X1, X2, X3, X4, X5, X6, X7, X8, y_true) in enumerate(trainloader):

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

            ## backward
            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()

            ## calculate accuracy
            acc_i  = accuracy_bin(y_pred, y_true)

            ## time
            duration_time = time.time() - start_time
            start_time = time.time()
            elapsed_time += duration_time
            total_time = duration_time * configer.n_epoch * len(trainset) // configer.batchsize

            ## log
            print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || lr: {:.6f} | loss_i: {:.4f} acc_i: {:.2%} p: {:.4f}".\
                format(getTime(), elapsed_time/3600, (total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                        i_epoch, configer.n_epoch, i_batch, len(trainset) // configer.batchsize, 
                        scheduler.get_lr()[-1], loss_i, acc_i, torch.exp(-loss_i))
            print(print_log)

            ## save
            loss_train += [loss_i.detach().cpu().numpy()]
            acc_train  += [acc_i.cpu().numpy()]

        loss_train = np.mean(np.array(loss_train))
        acc_train  = np.mean(np.array(acc_train))

        print('-----------------------------------------------------------------------------------------------------')


        loss_valid = []; acc_valid = []
        model.eval()
        for i_batch, (X0, X1, X2, X3, X4, X5, X6, X7, X8, y_true) in enumerate(validloader):

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

            ## calculate loss and accuracy
            loss_i = metric(y_pred, y_true)
            acc_i  = accuracy_bin(y_pred, y_true)

            ## log
            print_log = "{} || Epoch: [{:3d}]/[{:3d}] | Batch: [{:3d}]/[{:3d}] || loss_i: {:.4f} acc_i: {:.2%} p: {:.4f}".\
                format(getTime(), i_epoch, configer.n_epoch, i_batch, len(validset) // configer.batchsize, loss_i, acc_i, torch.exp(-loss_i))
            print(print_log)

            ## save
            loss_valid += [loss_i.detach().cpu().numpy()]
            acc_valid  += [acc_i.cpu().numpy()]

        loss_valid = np.mean(np.array(loss_valid))
        acc_valid  = np.mean(np.array(acc_valid))

        print('-----------------------------------------------------------------------------------------------------')

        logger.add_scalars('loss',      {'train': loss_train, 'valid': loss_valid}, i_epoch)
        logger.add_scalars('accuracy',  {'train': acc_train,  'valid': acc_valid},  i_epoch)
        logger.add_scalar ('lr', scheduler.get_lr()[-1], i_epoch)

        print_log = "{} || Epoch: [{:3d}]/[{:3d}] || lr: {:.6f} || train | loss: {:.4f} acc: {:.2%} p: {:.4f} | valid | loss: {:.4f} acc: {:.2%} p: {:.4f}".\
            format(getTime(), i_epoch, configer.n_epoch, scheduler.get_lr()[-1], 
                    loss_train, acc_train, np.exp(-loss_train),
                    loss_valid, acc_valid, np.exp(-loss_valid))
        print(print_log)
        
        if loss_valid < loss_valid_last:
            loss_valid_last = loss_valid
            model.save(prefix='../modelfile/deepid_lr_{}'.format(configer.finetune_lr))
        
        print('=====================================================================================================')
