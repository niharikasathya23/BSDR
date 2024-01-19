import torch
from torch import Tensor
from typing import Tuple
import numpy as np

class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)


def multi_class_FPr_TPr(conf_matrix, target_names=[]):

    n_classes = conf_matrix.shape[0]
    report = {}

    for i in range(n_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[i, :i].sum() + conf_matrix[i, i+1:].sum()

        mask = np.ones_like(conf_matrix)
        mask[i, :] = 0
        mask[:, i] = 0
        TN = np.where(mask, conf_matrix, 0).sum()
        FN = conf_matrix[:i, i].sum() + conf_matrix[i+1:, i].sum()

        P = TP / (TP + FP + 1)
        R = TP / (TP + FN + 1)

        TPr = TP / (TP + FN + 1)
        FPr = FP / (FP + TN + 1)

        class_name = i
        if i < len(target_names):
            class_name = target_names[i]
        
        report[class_name] = {
            "TPr": round(TPr, 2),
            "FPr": round(FPr, 2)
        }
    
    return report