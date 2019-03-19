import os
import time
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utiles import getTime
from datasets import ClassifyDataset, VerifyDataset, ClassifyWithSimilarity
from metrics import IdentifyLoss, VerifyBinLoss, TotalLoss, accuracy_mul, accuracy_bin
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
        features_dir = '../data/features'
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

    if save_features:
        np.save(features_path, features)







































def iniVerifyModel():
    modelpath = '../modelfile/verify.pkl'
    assert os.path.exists(modelpath), 'please train first! '

    model = torch.load(modelpath)
    return model

def test_verify(configer):

    testset = VerifyDataset('test')
    testloader = DataLoader(testset, configer.batchsize, shuffle=False)
    
    model = iniVerifyModel()
    if configer.cuda: model.cuda()
    metric = VerifyBinLoss()

    elapsed_time = 0; total_time = 0
    start_time = time.time()

    loss = []
    acc  = []

    model.eval()
    for i_batch, (X, y_true) in enumerate(testloader):

        # load data
        X = Variable(X.float())
        y_true = Variable(y_true.float())
        if configer.cuda: 
            X = X.cuda()
            y_true = y_true.cuda()
        
        # forward
        y_pred_prob = model(X)

        # calculate loss and accuracy
        loss_i = metric(y_pred_prob, y_true)
        acc_i  = accuracy_bin(y_pred_prob, y_true)
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






















def iniClassifyFinetunedModel(in_channels, n_classes, modeldir):

    assert os.path.exists(modeldir), 'please train first! '

    model = Classifier(in_channels, n_classes, modeldir)
    model.load(True)
    return model




def test_classify_with_similarity(configer):

    testset    = ClassifyWithSimilarity(configer.patch, configer.scale, 'test', configer.imsize)
    testloader = DataLoader(testset, configer.batchsize, shuffle=False)
    model      = iniClassifyFinetunedModel(configer.in_channels, configer.n_classes, '_'.join(configer.modeldir.split('_')[:-1]))
    if configer.cuda: model.cuda()
    metric      = TotalLoss(0.1)


    elapsed_time = 0; total_time = 0
    start_time = time.time()


    ident_loss = []
    similar_loss = []
    total_loss = []
    acc  = []

    model.eval()
    for i_batch, (X1, y1_true, X2, y2_true) in enumerate(testloader):

        # load data
        X1 = Variable(X1.float())
        y1_true = Variable(y1_true)
        X2 = Variable(X2.float())
        y2_true = Variable(y2_true)
        if configer.cuda: 
            X1 = X1.cuda()
            y1_true = y1_true.cuda()
            X2 = X2.cuda()
            y2_true = y2_true.cuda()

        # forward
        X1, y1_pred_prob = model(X1)
        X2, y2_pred_prob = model(X2)

        # calculate loss and accuracy
        ident_i, similar_i, total_i = metric(X1, X2, y1_pred_prob, y2_pred_prob, y1_true, y2_true)
        acc_i  = (accuracy_mul(y1_pred_prob, y1_true) + accuracy_mul(y2_pred_prob, y2_true)) / 2
        ident_loss += [ident_i.detach().cpu().numpy()]
        similar_loss += [similar_i.detach().cpu().numpy()]
        total_loss += [total_i.detach().cpu().numpy()]
        acc  += [acc_i.cpu().numpy()]

        # time
        duration_time = time.time() - start_time
        start_time = time.time()
        elapsed_time += duration_time
        total_time = duration_time * len(testset) // configer.batchsize
       
        # print
        print_log = "{} || Elapsed: {:.4f}h | Left: {:.4f}h | FPS: {:4.2f} || Batch: [{:3d}]/[{:3d}] || ident_i: {:.4f} similar_i: {:.4f} total_i: {:.4f} acc_i: {:.2%}".\
            format(getTime(), elapsed_time/3600, (total_time - elapsed_time)/3600, configer.batchsize / duration_time,
                    i_batch, len(testset) // configer.batchsize, 
                    ident_i, similar_i, total_i, acc_i)
        print(print_log)


    ident_loss = np.mean(np.array(ident_loss))
    similar_loss = np.mean(np.array(similar_loss))
    total_loss = np.mean(np.array(total_loss))
    acc  = np.mean(np.array(acc))

    print_log = "{} || Elapsed: {:.4f}h || ident_i: {:.4f} similar_i: {:.4f} total_i: {:.4f} acc_i: {:.2%}".\
        format(getTime(), elapsed_time/3600, ident_loss, similar_loss, total_loss, acc)
    print(print_log)