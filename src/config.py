from torch.cuda import is_available
from easydict import EasyDict


def updateConfiger(configer):

    configer.modelname = 'classify_patch{}_scale{}'.format(configer.patch, configer.scale)
    configer.modeldir  = '../modelfile/classify/{}'.format(configer.modelname)
    configer.logdir    = '../logfile/classify/{}'.format(configer.modelname)


configer = EasyDict()

configer.patch = 2              # 0 ~ 8
configer.scale = 'S'            # 'S', 'M', 'L'

## for classification
configer.in_channels = 3
configer.n_classes = 5749
configer.cuda = is_available()




## optimizer: classification
# configer.lrbase = 0.005
# configer.stepsize = 50
# configer.gamma = 0.1
# configer.batchsize = 64
# configer.n_epoch = 120
# configer.valid_batch = 100

## optimizer: deepid
configer.lrbase = 0.001
configer.stepsize = 100
configer.gamma = 0.1
configer.batchsize = 64
configer.n_epoch = 250
configer.valid_batch = 100






configer.modelname = None
configer.modeldir = None
configer.logdir = None

updateConfiger(configer)




