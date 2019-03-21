from torch.cuda import is_available
from easydict import EasyDict


def updateConfiger(configer):

    configer.modelname = 'classify_patch{}_scale{}'.format(configer.patch, configer.scale)
    configer.modeldir  = '../modelfile/classify/{}'.format(configer.modelname)
    configer.logdir    = '../logfile/classify/{}'.format(configer.modelname)


configer = EasyDict()

configer.patch = 0              # 0 ~ 8
configer.scale = 'S'            # 'S', 'M', 'L'

## for classification
configer.in_channels = 3
configer.n_classes = 5749

## optimizer
configer.lrbase = 0.005
configer.stepsize = 50
configer.gamma = 0.1

configer.verify_weight = 0.1 

configer.batchsize = 64
configer.n_epoch = 120
configer.valid_batch = 100

configer.cuda = is_available()


configer.modelname = None
configer.modeldir = None
configer.logdir = None

updateConfiger(configer)




