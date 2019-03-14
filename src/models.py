import os
import torch
import torch.nn as nn

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Params:
            x:  {tensor(N, H, W, C)}
        Returns:
            x:  {tensor(N, H * W * C)}
        """
        return x.view(x.shape[0], -1)


class DeepIdFeatures(nn.Module):

    def __init__(self, in_channels):
        super(DeepIdFeatures, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 20, 4, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 40, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(40, 60, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(60, 80, 2),
            nn.ReLU(True),
        )
        self.features = None

    def forward(self, x):
        """
        Params:
            x:  {tensor(N, H, W, C)}
        Returns:
            x:  {tensor(N, 160)}
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x  = torch.cat([x1, x2], 1)

        if self.features is None:
            self.features = nn.Linear(x.shape[-1], 160)
        x = self.features(x)
        
        x = self.norm(x)

        return x
    
    def norm(self, x, mode='l2'):
        """
        Params:
            x:      {tensor(N, 160)}
            mode:   {str}   'l1' ,'l2'
        """
        if mode == 'l1':
            _x_ = torch.sum(torch.abs(x), dim=1)
        elif mode == 'l2':
            _x_ = torch.sqrt(torch.sum(x**2, dim=1))
        
        x = x / torch.unsqueeze(_x_, dim=1)
        return x

class DeepIdClassifier(nn.Module):

    def __init__(self, n_classes):
        super(DeepIdClassifier, self).__init__()
        
        self.classifier = nn.Linear(160, n_classes)
    
    def forward(self, x):
        """
        Params:
            x:  {tensor(N, 160)}
        Returns:
            x:  {tensor(N, n_classes)}
        """
        x = self.classifier(x)

        return x

class Classifier(nn.Module):

    def __init__(self, in_channels, n_classes, modeldir):
        super(Classifier, self).__init__()

        self.features = DeepIdFeatures(in_channels)
        self.classifier = DeepIdClassifier(n_classes)

        self.modeldir = modeldir
        self.featurespath = os.path.join(modeldir, 'features.pkl')
        self.classifierpath = os.path.join(modeldir, 'classifier.pkl')

    def forward(self, x):
        """
        Params:
            x:  {tensor(N, H, W, C)}
        Returns:
            x:  {tensor(N, n_classes)}
        """
        x = self.features(x)
        y = self.classifier(x)

        return x, y

    def save(self):

        # torch.save(self.features.state_dict(), self.featurespath)
        # torch.save(self.classifier.state_dict(), self.classifierpath)
        
        torch.save(self.features, self.featurespath)
        torch.save(self.classifier, self.classifierpath)

        print("model saved at {} ! ".format(self.modeldir))
    
    def load(self):
        # state_dict = torch.load(self.featurespath)
        # self.features.load_state_dict(state_dict)
        # state_dict = torch.load(self.classifierpath)
        # self.classifier.load_state_dict(state_dict)

        self.features = torch.load(self.featurespath)
        self.classifier = torch.load(self.classifierpath)

        print("model loaded from {} ! ".format(self.modeldir))
        