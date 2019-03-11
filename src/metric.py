import torch
import torch.nn as nn
import torch.functional as F

class DeepIdLoss(nn.Module):

    def __init__(self, ident, verif, m):
        super(DeepIdLoss, self).__init__()
        self.ident = ident / (ident + verif)
        self.verif = verif / (ident + verif)
        self.m = m
        self.bceloss = nn.BCELoss()
    
    def forward(self, y_true, y_pred, x1_feat, x2_feat):
        """
        Params:
            y_true: {tensor(N,)} `0` or `1`
            y_pred: {tensor(N, 2)}
            x1_feat:{tensor(N, n_features)}
            x2_feat:{tensor(N, n_features)}
        """
        diff = (x1_feat - x2_feat)**2
        diff = torch.sum(diff, dim=1)
        diff[y_true==0] = torch.clamp(self.m - diff[y_true==0], 0, float('inf'))
        verif = torch.mean(diff)
        ident = self.bceloss(y_pred, y_true)
        total = self.ident * ident + self.verif * verif
        return total