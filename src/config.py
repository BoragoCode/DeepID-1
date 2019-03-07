from easydict import EasyDict

configer = EasyDict()

configer.imgsize = (64, 64)
configer.use_channels = [_ for _ in range(30, 40)]
configer.n_channels = len(configer.use_channels)

configer.modelbase = 'vgg16_bn'
configer.modelname = '{}_{}chs'.format(configer.modelbase, configer.use_channels)
