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

    def save(self, finetune=False):
        
        featurespath = self.featurespath
        classifierpath = self.classifierpath

        if finetune:
            featurespath = featurespath.split('.')
            featurespath[-2] += '_finetune'
            featurespath = '.'.join(featurespath)
            classifierpath = classifierpath.split('.')
            classifierpath[-2] += '_finetune'
            classifierpath = '.'.join(classifierpath)

        torch.save(self.features, featurespath)
        torch.save(self.classifier, classifierpath)

        print("model saved at {} ! ".format(self.modeldir))
    
    def load(self, finetune=False):

        featurespath = self.featurespath
        classifierpath = self.classifierpath

        if finetune:
            featurespath = featurespath.split('.')
            featurespath[-2] += '_finetune'
            featurespath = '.'.join(featurespath)
            classifierpath = classifierpath.split('.')
            classifierpath[-2] += '_finetune'
            classifierpath = '.'.join(classifierpath)

        self.features = torch.load(featurespath)
        self.classifier = torch.load(classifierpath)

        print("model loaded from {} ! ".format(self.modeldir))


class Verifier(nn.Module):
    """
    Attributes:
        features:   {dict{'classify_patch{patch}_scale{scale}': nn.Sequential(nn.Linear(2*160, 80), nn.ReLU(True))}}
    """
    __type = [[i, s] for i in range(9) for s in ['S', 'M', 'L']]

    def __init__(self):
        super(Verifier, self).__init__()
        
        self.features = dict()
        self._load_features()

        self.classifier = nn.Sequential(
            nn.Linear(27*80, 4800),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(4800, 1),
            nn.Sigmoid(),
        )

    def _load_features(self):

        for patch, scale in self.__type:
            key = 'classify_patch{}_scale{}'.format(patch, scale)
            self.features[key] = nn.Sequential(
                nn.Linear(2*160, 80),
                nn.ReLU(True),
            )

    def forward(self, X):
        """
        Params:
            X:  {tensor(N, n_groups(27), 160x2)}
        Returns:
            x:  {tensor(N)}
        """
        N, g, n = X.shape
        
        ## locally-connected layer
        x = None
        for i in range(len(self.__type)):
            patch, scale = self.__type[i]
            key = 'classify_patch{}_scale{}'.format(patch, scale)
            if x is None:
                x = self.features[key](X[:, i])
            else:
                x = torch.cat([x, self.features[key](X[:, i])], dim=1)

        ## fully connected layer
        x = self.classifier(x).view(-1)

        return x






































class DeepID(nn.Module):
    """ Whole model

    Attributes:
        features(27):   {dict[key]=DeepIdFeatures} classify_patch{}_scale{}: DeepIdFeatures
                        input       --->    output
                        N x 3 x H x W       N x 160

        verifier:       {Verifier}
                        input       --->    output
                        N x 27 x (160x2)    N x 1

    """
    
    __type = [[i, s] for i in range(9) for s in ['S', 'M', 'L']]

    def __init__(self, type=None):
        """
        Params:
            type:   {str} None or `finetune`, `deepid`
        """
        super(DeepID, self).__init__()

        self.type = type
        self.load()

    def forward(self, X0, X1, X2, X3, X4, X5, X6, X7, X8):
        """
        Params:
            X0: {tensor(N, 2, 3, 3, 44, 33)}    patch0
            X1: {tensor(N, 2, 3, 3, 25, 33)}    patch1
            X2: {tensor(N, 2, 3, 3, 25, 33)}    patch2
            X3: {tensor(N, 2, 3, 3, 25, 33)}    patch3
            X4: {tensor(N, 2, 3, 3, 25, 25)}    patch4
            X5: {tensor(N, 2, 3, 3, 25, 25)}    patch5
            X6: {tensor(N, 2, 3, 3, 25, 25)}    patch6
            X7: {tensor(N, 2, 3, 3, 25, 25)}    patch7
            X8: {tensor(N, 2, 3, 3, 25, 25)}    patch8
        Notes:
            2 people; 3 scales: 'S', 'M', 'L'
        """
        scales = ['S', 'M', 'L']
        patches = [X0, X1, X2, X3, X4, X5, X6, X7, X8]
        
        features = torch.zeros(X0.shape[0], 27, 160*2)
        for patch, scale in self.__type:
            key = 'classify_patch{}_scale{}'.format(patch, scale)
            idx_s = scales.index(scale)
            X = patches[patch][:, :, idx_s]     # {tensor(N, 2, 3, 44, 33)}
            X1 = self.features[key](X[:, 0])    # {tensor(N, 3, h, w)} ---> {tensor(N, 160)}
            X2 = self.features[key](X[:, 1])    # {tensor(N, 3, h, w)} ---> {tensor(N, 160)}
            features[:, patch*3 + idx_s] = torch.cat([X1, X2], dim=1)   # {tensor(N, 160x2)}
        
        y = self.verifier(features)
        return y

    def load(self):
        """
        Params:
            type:   {str} None or `finetune`, `deepid`
        """

        self.features = dict()
        for patch, scale in self.__type:
            key = 'classify_patch{}_scale{}'.format(patch, scale)
            featurename = 'features' if self.type is None else 'features_{}'.format(self.type)
            self.features[key] = torch.load('../modelfile/{}/{}.pkl'.format(key, featurename))
        verifiername = 'verify' if self.type is None else 'verify_{}'.format(self.type)
        self.verifier = torch.load('../modelfile/{}.pkl'.format(verifiername))

    def save(self, total=False):

        if total:
            torch.save(self, '../modelfile/deepid.pkl')
        else:
            for patch, scale in self.__type:
                key = 'classify_patch{}_scale{}'.format(patch, scale)
                torch.save(self.features[key], '../modelfile/{}/features_deepid.pkl'.format(key))
            torch.save(self.verifier, '../modelfile/verify_deepid.pkl')
























if __name__ == "__main__":
    # M = Verifier()
    # X = torch.randn(128, 27, 160*2)
    # y = M(X)
    M = DeepID('finetune')
    X0 = torch.zeros(32, 2, 3, 3, 44, 33)
    X1 = torch.zeros(32, 2, 3, 3, 25, 33)
    X2 = torch.zeros(32, 2, 3, 3, 25, 33)
    X3 = torch.zeros(32, 2, 3, 3, 25, 33)
    X4 = torch.zeros(32, 2, 3, 3, 25, 25)
    X5 = torch.zeros(32, 2, 3, 3, 25, 25)
    X6 = torch.zeros(32, 2, 3, 3, 25, 25)
    X7 = torch.zeros(32, 2, 3, 3, 25, 25)
    X8 = torch.zeros(32, 2, 3, 3, 25, 25)
    y = M(X0, X1, X2, X3, X4, X5, X6, X7, X8)