import torch
import torch.nn as nn


accuracy = lambda y_pred, y_true: torch.mean((y_pred == y_true).float())

def accuracy_mul(y_pred_prob, y_true):
    """
    Params:
        y_pred_prob: {tensor(N, n_class)}
        y_true:      {tensor(N)}
    Returns:
        acc:         {tensor(1)}
    """
    y_pred = torch.argmax(y_pred_prob, dim=1)
    acc = accuracy(y_pred, y_true)
    return acc

def accuracy_bin(y_pred_prob, y_true):
    """
    Params:
        y_pred_prob: {tensor(N)}
        y_true:      {tensor(N)}
    Returns:
        acc:         {tensor(1)}
    """
    y_pred_prob[y_pred_prob > 0.5] = 1.0
    y_pred_prob[y_pred_prob < 0.5] = 0.0
    acc = accuracy(y_pred_prob, y_true)
    return acc
    
    
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

        self.m = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x1, x2, y):
        """
        Params:
            x1: {tensor(N, 160)}
            x2: {tensor(N, 160)}
            y:  {tensor(N)}      bool
        Returns:
            loss:   {tensor(1)}
        """
        d2 = torch.sum((x1 - x2)**2, dim=1)
        a = self.m - d2[y==False]
        b = torch.clamp(self.m - d2[y==False], 0, float('inf'))
        d2[y==False] = torch.clamp(self.m - d2[y==False], 0, float('inf'))

        return torch.mean(d2, dim=1)

class TotalLoss(nn.Module):

    def __init__(self, k=0.1):
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
