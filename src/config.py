from easydict import EasyDict

configer = EasyDict()

configer.imgsize = (96, 96)
configer.use_channels = [_ for _ in range(30, 40)]
configer.n_channels = len(configer.use_channels)
configer.n_features = 512

configer.modelbase = 'vgg16_bn'
configer.modelname = 'deepid_{}_{}chs_{}feats'.\
                format(configer.modelbase, configer.n_channels, configer.n_features)

configer.learningrate  = 1e-3
configer.batchsize     = 32
configer.n_epoch       = 300
configer.valid_epoch   = 100