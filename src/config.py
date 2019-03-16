from torch.cuda import is_available
from easydict import EasyDict

configer = EasyDict()

configer.patch = 1              # 0 ~ 8
configer.scale = 'L'            # 'S', 'M', 'L'
configer.with_verify = False




configer.in_channels = 3
configer.n_classes = 5749

configer.lrbase = 0.001
configer.stepsize = 50
configer.gamma = 0.9

configer.verify_weight = 0.1 

configer.batchsize = 128
configer.n_epoch = 200
configer.valid_batch = 100

configer.cuda = is_available()






if configer.patch == 0:
    configer.imsize = (44, 33)  # h: w = 4: 3
elif configer.patch in [1, 2, 3]:
    configer.imsize = (25, 33)  # h: w = 3: 4
else:
    configer.imsize = (25, 25)  # h: w = 1: 1

configer.modelname = 'classify_patch{}_scale{}'.format(configer.patch, configer.scale)
configer.modeldir  = '../modelfile/{}'.format(configer.modelname)
configer.logdir    = '../logfile/{}'.format(configer.modelname)
if configer.with_verify:
    configer.modeldir  = '../modelfile/{}_with_verify'.format(configer.modelname)
    configer.logdir = '{}_with_verify'.format(configer.logdir)









def updateConfiger(configer):
    
    if configer.patch == 0:
        configer.imsize = (44, 33)  # h: w = 4: 3
    elif configer.patch in [1, 2, 3]:
        configer.imsize = (25, 33)  # h: w = 3: 4
    else:
        configer.imsize = (25, 25)  # h: w = 1: 1

    configer.modelname = 'classify_patch{}_scale{}'.format(configer.patch, configer.scale)
    configer.modeldir  = '../modelfile/{}'.format(configer.modelname)
    configer.logdir    = '../logfile/{}'.format(configer.modelname)
    if configer.with_verify:
        configer.modeldir  = '../modelfile/{}_with_verify'.format(configer.modelname)
        configer.logdir = '{}_with_verify'.format(configer.logdir)