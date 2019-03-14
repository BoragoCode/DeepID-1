from easydict import EasyDict

configer = EasyDict()

configer.patch = 0              # 0 ~ 8
configer.scale = 'M'            # 'S', 'M', 'L'

if configer.patch == 0:
    configer.imsize = (44, 33)  # h: w = 4: 3
elif configer.patch in [1, 2, 3]:
    configer.imsize = (25, 33)  # h: w = 3: 4
else:
    configer.imsize = (25, 25)  # h: w = 1: 1


configer.in_channels = 3
configer.n_classes = 5749
configer.modelname = 'classify_patch{}_scale{}'.format(configer.patch, configer.scale)
configer.with_verify = True
configer.modeldir  = '../modelfile/{}'.format(configer.modelname)
configer.logdir    = '../logfile/{}'.format(configer.modelname)
if configer.with_verify:
    configer.logdir = '{}_with_verify'.format(configer.logdir)


configer.lrbase = 0.001
configer.stepsize = 100
configer.gamma = 0.9

configer.verify_weight = 0.1 

configer.batchsize = 128
configer.n_epoch = 500
configer.valid_batch = 100
