from easydict import EasyDict

configer = EasyDict()

configer.patch = 0
configer.scale = 'M'

if configer.patch == 0:
    configer.imsize = (44, 33)  # h: w = 4: 3
elif configer.patch in [1, 2, 3]:
    configer.imsize = (25, 33)  # h: w = 3: 4
else:
    configer.imsize = (25, 25)  # h: w = 1: 1
    
configer.n_classes = 5749

configer.modelname = 'classify_patch{}_scale{}'.\
                        format(configer.patch, configer.scale)
configer.lrbase = 0.001
configer.batchsize = 64
configer.n_epoch = 300
