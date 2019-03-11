import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as tvmdl

from config import configer

class VGGFeatures(nn.Module):
    vggs = {
        'vgg11_bn': tvmdl.vgg11_bn(True),
        # 'vgg11_bn': tvmdl.vgg11_bn(True),
    }
    def __init__(self, in_channels, out_features, type='vgg11_bn'):
        super(VGGFeatures, self).__init__()
        basemodel = self.vggs[type]
        self.base = basemodel.features
        conv1 = self.base[0]
        self.base[0] = nn.Conv2d(in_channels, conv1.out_channels, conv1.kernel_size, conv1.stride)
        self.vect = nn.Linear(2*2*512, out_features)
    
    def forward(self, x):
        x = self.base(x)
        x = x.view(x.shape[0], -1)
        x = self.vect(x)
        return x

class DeepId2Features(nn.Module):

    def __init__(self, in_channels, out_features):
        super(DeepId2Features, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 20, 3, padding=1),   # 96 x 96 x 20
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 48 x 48 x 20
            nn.Conv2d(20, 40, 3, padding=1),            # 48 x 48 x 40
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # 24 x 24 x 40
            nn.Conv2d(40, 60, 3, padding=1),            # 12 x 12 x 60
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         #  6 x  6 x 60
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(60, 80, 3),                       #  4 x  4 x 80
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
        )
        self.vect  = nn.Linear(6*6*60+4*4*80, out_features)
    
    def forward(self, x):
        x  = self.features(x)
        x_ = self.conv4(x)
        x  = x.view(x.shape[0],)
        x_ = x_.view(x.shape[0],)
        x  = torch.cat([x, x_], 1)
        x  = self.vect(x)
        return x

class DeepIdModel(nn.Module):
    
    def __init__(self, features, in_channels, out_features):
        super(DeepIdModel, self).__init__()
        self.features = features(in_channels, out_features)
        self.classifier = nn.Sequential(
            nn.Linear(out_features*2, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 1),
        )

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x = torch.cat([x1, x2], 1)
        y = self.classifier(x)
        y = nn.Sigmoid()(y)
        y = y.view(y.shape[0])
        return y

modeldicts = {
    'vgg11_bn': DeepIdModel(lambda n1, n2: VGGFeatures(n1, n2, 'vgg11_bn'), configer.n_channels, configer.n_features),
    'deepid2':  DeepIdModel(lambda n1, n2: DeepId2Features(n1, n2),         configer.n_channels, configer.n_features),
}

if __name__ == "__main__":
    x1 = torch.zeros([32, 10, 96, 96])
    x2 = torch.zeros([32, 10, 96, 96])

    net = DeepIdModel(lambda in_channels, out_features: VGGFeatures(in_channels, out_features, 'vgg11_bn'), 10, 512)
    y = net(x1, x2)
    
    print(y.size)