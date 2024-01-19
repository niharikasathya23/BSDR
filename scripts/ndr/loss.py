import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

# experiment: use cross entropy loss for classification
class MyClassificationLoss(nn.Module):

    def __init__(self):
        super(MyClassificationLoss, self).__init__()

    def forward(self, x, y):
        func = nn.CrossEntropyLoss()
        return func(x, y)

class MLPClassificationLoss(nn.Module):

    def __init__(self, sigma_squared = 2, max_disp = 256):
        super(MLPClassificationLoss, self).__init__()
        self.sigma_squared = sigma_squared
        self.max_disp = max_disp

    def forward(self, x, y, lbl=None):
        # x = probability, y = disparity map
        gt_i = torch.arange(0, self.max_disp).to(x.device)
        gt_i = torch.reshape(gt_i, (1, self.max_disp, 1)) # added a dim at the end to match the shape of gt
        gt = torch.exp(-0.5 * torch.abs(y - gt_i) ** 2 / self.sigma_squared) / self.sigma_squared
        ce = torch.where(
            x > 0, -(gt * torch.log(x + 1e-10)), torch.zeros_like(x)
        )
        if lbl is None:
            ce = torch.mean(ce, dim=2)
            ce = torch.mean(ce, dim=0)
            classification_loss = torch.sum(ce)
        else:
            ce = torch.sum(ce, dim=1)
            classification_loss = torch.mean(ce*lbl)

        return classification_loss

class MLPSegmentationLoss(nn.Module):
    def __init__(self):
        super(MLPSegmentationLoss, self).__init__()

    def forward(self, x, y, lbl=None):
        # x = D_tilda, y = disparity map
        regression_loss = torch.nn.functional.l1_loss(
            x, y, reduction="none"
        )
        regression_loss = torch.where(
            regression_loss > 1, torch.zeros_like(regression_loss), regression_loss
        )
        if lbl is None:
            regression_loss = torch.mean(regression_loss)
        else:
            regression_loss = torch.mean(regression_loss*lbl)

        return regression_loss

def DisparityLoss(D_tilda, probs, y, lam_cls=1., lam_reg=1., max_disp=256, lbl=None):

    # D_tilda = torch.reshape(D_tilda, (D_tilda.size(0), D_tilda.size(1), D_tilda.size(2)*D_tilda.size(3)))
    # probs = torch.reshape(probs, (probs.size(0), probs.size(1), probs.size(2)*probs.size(3)))
    y = torch.reshape(y, (y.size(0), y.size(1), y.size(2)*y.size(3)))

    # Original classification loss
    cls_loss_fun = MLPClassificationLoss(max_disp=max_disp)
    cls_loss = cls_loss_fun(probs, y, lbl=lbl)

    # experiment: use one-hot encodings and cross entropy loss for classification
    # y_int = torch.floor(y).long()[:,0,:,:] - 1 # represent classes as the class + some positive offset
    # y_probs = nn.functional.one_hot(y_int, max_disp).float() # a one-hot encoding of integer disparity values
    # y_probs = torch.transpose(y_probs, 3, 1)
    # cls_loss_fun = MyClassificationLoss()
    # cls_loss = cls_loss_fun(probs, y_probs)

    # Original regression loss
    reg_loss_fun = MLPSegmentationLoss()
    reg_loss = reg_loss_fun(D_tilda, y, lbl=lbl)

    # experiment: omit regression loss to start
    # reg_loss = torch.Tensor([0]).cuda()
    # reg_loss = torch.sum(reg_loss)

    return lam_cls*cls_loss + lam_reg*reg_loss, cls_loss, reg_loss

# experiment: use L1 loss, which is one used in Tabel 1 of paper
def SimpleL1Loss(D_tilda, y):
    return nn.functional.l1_loss(D_tilda, y, reduction='mean')

# experiment: use L1 loss, which is one used in Tabel 1 of paper
def SimpleL2Loss(D_tilda, y):
    return torch.linalg.norm(D_tilda-y)
