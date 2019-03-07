import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as tvmdl

from config import configer

class VGGFeatures(nn.Module):
    vggs = {
        'vgg11': tvmdl.vgg11(True),
        'vgg11_bn': tvmdl.vgg11_bn(True),
        'vgg16': tvmdl.vgg16(True),
        'vgg16_bn': tvmdl.vgg16_bn(True),
    }
    def __init__(self, in_channels, out_features, type='vgg11_bn'):
        super(VGGFeatures, self).__init__()
        basemodel = self.vggs[type]
        self.features = basemodel.features
        conv1 = self.features[0]
        self.features[0] = nn.Conv2d(in_channels, conv1.out_channels, conv1.kernel_size, conv1.stride)
        self.classifier = nn.Linear(2*2*512, out_features)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class DeepIdModel(nn.Module):
    
    def __init__(self, features, in_channels, out_features):
        super(DeepIdModel, self).__init__()
        self.features = features(in_channels, out_features)
        self.classifier = nn.Sequential(
            nn.Linear(out_features*2, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 1),
        )

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x = torch.cat([x1, x2], 1)
        y = self.classifier(x)
        y = nn.Sigmoid()(y)
        y = y.view(y.shape[0])
        return y

_models = {
    'deepid_vgg11_bn_1chs_128feats':    DeepIdModel(lambda in_channels, out_features: VGGFeatures(in_channels, out_features, 'vgg11_bn'), 1,  128),
    'deepid_vgg11_10chs_512feats':      DeepIdModel(lambda in_channels, out_features: VGGFeatures(in_channels, out_features, 'vgg11')   , 10, 512),
    'deepid_vgg16_bn_1chs_128feats':    DeepIdModel(lambda in_channels, out_features: VGGFeatures(in_channels, out_features, 'vgg16_bn'), 1,  128),
    'deepid_vgg16_10chs_512feats':      DeepIdModel(lambda in_channels, out_features: VGGFeatures(in_channels, out_features, 'vgg16')   , 10, 512),
}

if __name__ == "__main__":
    x1 = torch.zeros([32, 10, 96, 96])
    x2 = torch.zeros([32, 10, 96, 96])

    net = DeepIdModel(lambda in_channels, out_features: VGGFeatures(in_channels, out_features, 'vgg11_bn'), 10, 512)
    y = net(x1, x2)
    
    print(y.size)