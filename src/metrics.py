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
    """ for classification
    """
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



class SimilarityLoss(nn.Module):
    """ similarity loss for classification
    Notes:
        .. math::
            \text{loss}(f_i, f_j, y_{ij}, \theta_{ve}) = 
                \begin{cases}
                    \frac{1}{2} ||f_i - f_j||_2^2           & if y_{ij} =  1 \\
                    \frac{1}{2} max(0, m-||f_i - f_j||_2)^2 & if y_{ij} = -1 
                \end{cases}
    """
    def __init__(self):
        super(SimilarityLoss, self).__init__()

        self.m = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x1, x2, y):
        """
        Params:
            x1: {tensor(N, 160)} unitized vector
            x2: {tensor(N, 160)} unitized vector
            y:  {tensor(N)}      bool
        Returns:
            l:  {tensor(1)}
        """
        ## l1 loss
        l = torch.sum(torch.abs(x1 - x2), dim=1)
        l_pos = torch.mean(l[y==1])
        l_neg = torch.mean(torch.clamp(self.m - l[y==0], 0, float('inf')))

        ## l2 loss
        # l = torch.sqrt(torch.sum((x1 - x2)**2, dim=1))
        # l_pos = torch.mean(l[y==1]**2)
        # l_neg = torch.mean(torch.clamp(self.m - l[y==0], 0, float('inf'))**2)

        l = l_pos + l_neg

        return l



class TotalLoss(nn.Module):
    """ for classification, with similarity loss
    Attributes:
        k:          {float}         weight of similarity loss
        identify:   {IdentifyLoss}  classification loss
    """
    def __init__(self, k=0.01):
        super(TotalLoss, self).__init__()
        
        self.k = k
        self.identify = IdentifyLoss()
        self.similarity = SimilarityLoss()

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
        ident = (self.identify(y1_pred, y1_true) + self.identify(y2_pred, y2_true)) / 2
        similar = self.similarity(x1, x2, y1_true==y2_true)

        total = ident + self.k * similar
        return ident, similar, total



class VerifyBinLoss(nn.Module):
    """ for verification
    """

    def __init__(self):
        super(VerifyBinLoss, self).__init__()

        self.lossfunc = nn.BCELoss()
    
    def forward(self, y_pred, y_true):
        """
        Params:
            y_pred: {tensor(N)}
            y_true: {tensor(N)}
        """
        return self.lossfunc(y_pred, y_true)
        