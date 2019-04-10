from torch.cuda import is_available
from easydict import EasyDict


def updateConfiger(configer):

    configer.modelname = 'classify_patch{}_scale{}'.format(configer.patch, configer.scale)
    configer.modeldir  = '../modelfile/celeba_classify/{}'.format(configer.modelname)
    configer.logdir    = '../logfile/celeba_classify/{}'.format(configer.modelname)


configer = EasyDict()

configer.patch = 0              # 0 ~ 8
configer.scale = 'S'            # 'S', 'M', 'L'

## for classification
configer.in_channels = 3
# configer.n_classes = 5749         # lfw
configer.n_classes = 8000          # celeba
configer.cuda = is_available()
# configer.cuda = False




## optimizer: classification
# configer.lrbase = 0.0005
# configer.stepsize = 120
# configer.gamma = 0.2
# configer.batchsize = 256
# configer.n_epoch = 200
# configer.valid_batch = 100


## optimizer: deepid
configer.lrbase = 0.001
configer.stepsize = 30
configer.gamma = 0.1
configer.finetune_lr = 0.01

configer.batchsize = 128
configer.n_epoch = 50
configer.valid_batch = 100






configer.modelname = None
configer.modeldir = None
configer.logdir = None

updateConfiger(configer)




