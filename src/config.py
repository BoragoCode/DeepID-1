from torch.cuda import is_available
from easydict import EasyDict


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

configer = EasyDict()

configer.patch = 1              # 0 ~ 8
configer.scale = 'L'            # 'S', 'M', 'L'
configer.with_verify = False




configer.in_channels = 3
configer.n_classes = 5749

configer.lrbase = 0.001
configer.stepsize = 200
configer.gamma = 0.1

configer.verify_weight = 0.1 

configer.batchsize = 256
configer.n_epoch = 500
configer.valid_batch = 100

configer.cuda = is_available()


configer.imsize = None
configer.modelname = None
configer.modeldir = None
configer.logdir = None

updateConfiger(configer)




