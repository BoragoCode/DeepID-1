import torch
import torch.nn as nn

class IdentifyLoss(nn.Module):

    def __init__(self):
        super(IdentifyLoss, self).__init__()

        self.identify = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        """
        Params:
            y_pred: {tensor(N, n_class)}
            y_true: {tensor(N)}
        Returns:
            loss:   {tensor(1)}
        """
        return self.identify(y_pred, y_true)

class VerifyLoss(nn.Module):

    def __init__(self):
        super(VerifyLoss, self).__init__()
        
    def forward(self, x1, x2, y):
        """
        Params:
            x1: {tensor(N, 160)}
            x2: {tensor(N, 160)}
            y:  {tensor(N)}      bool
        Returns:
            loss:   {tensor(1)}
        """
        pass

class TotalLoss(nn.Module):

    def __init__(self, k):
        super(TotalLoss, self).__init__()
        
        self.k = k
        self.identify = IdentifyLoss()
        self.verify   = VerifyLoss()

    def forward(self, x1, x2, y1_pred, y2_pred, y1_true, y2_true):
        """
        Params:
            x1:      {tensor(N, 160)}
            x2:      {tensor(N, 160)}
            y1_pred: {tensor(N, n_class)}
            y2_pred: {tensor(N, n_class)}
            y1_true: {tensor(N)}
            y2_true: {tensor(N)}
        Returns:
            loss:   {tensor(1)}
        """
        ident = self.identify(y1_pred, y1_true) + self.identify(y2_pred, y2_true)
        verif = self.verify(x1, x2, y1_true==y2_true)

        return ident + self.k * verif