import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath('/Users/rithik/Desktop/bsdr/scripts/ndr/loss')))
sys.path.append(parent_dir)
from ndr.loss import DisparityLoss

def water_obstacle_separation_loss(features, gt_mask):
    """Computes the water-obstacle separation loss from intermediate features.

    Args:
        features (torch.tensor): Features tensor
        gt_mask (torch.tensor): Ground truth tensor
        clipping_value (float): Clip loss at clipping_value * sigma
    """
    epsilon_watercost = 0.01
    min_samples = 5

    # Resize gt mask to match the extracted features shape (x,y)
    feature_size = (features.size(2), features.size(3))
    gt_mask = F.interpolate(gt_mask, size=feature_size, mode='area')

    # Create water and obstacles masks.
    # The masks should be of type float so we can multiply it later in order to mask the elements
    mask_water = gt_mask[:,1].unsqueeze(1)

    mask_obstacles = gt_mask[:,0].unsqueeze(1)

    # Count number of water and obstacle pixels, clamp to at least 1 (for numerical stability)
    elements_water = mask_water.sum((0,2,3), keepdim=True).clamp(min=1.)
    elements_obstacles = mask_obstacles.sum((0,2,3), keepdim=True)

    # Zero loss if number of samples for any class is smaller than min_samples
    if elements_obstacles.squeeze() < min_samples or elements_water.squeeze() < min_samples:
        return torch.tensor(0.)

    # Only keep water and obstacle pixels. Set the rest to 0.
    water_pixels = mask_water * features
    obstacle_pixels = mask_obstacles * features

    # Mean value of water pixels per feature (batch average)
    mean_water = water_pixels.sum((0,2,3), keepdim=True) / elements_water

    # Mean water value matrices for water and obstacle pixels
    mean_water_wat = mean_water * mask_water
    mean_water_obs = mean_water * mask_obstacles

    # Variance of water pixels (per channel, batch average)
    var_water = (water_pixels - mean_water_wat).pow(2).sum((0,2,3), keepdim=True) / elements_water

    # Average quare difference of obstacle pixels and mean water values (per channel)
    difference_obs_wat = (obstacle_pixels - mean_water_obs).pow(2).sum((0,2,3), keepdim=True)

    # Compute the separation
    loss_c = elements_obstacles * var_water / (difference_obs_wat + epsilon_watercost)

    var_cost = loss_c.mean()

    return var_cost


def focal_loss(logits, labels, gamma=2.0, alpha=4.0, target_scale='labels'):
    """Focal loss of the segmentation output `logits` and ground truth `labels`."""


    epsilon = 1.e-9

    logits_sm = torch.softmax(logits, 1)

    # Focal loss
    fl = -labels * torch.log(logits_sm + epsilon) * (1. - logits_sm) ** gamma
    fl = fl.sum(1) # Sum focal loss along channel dimension

    # Return mean of the focal loss along spatial and batch dimensions
    return fl.mean()

def separation_loss(logits, labels, gt_disp, disp, separation_lambda=0.05, ignore_label=255):

    logits, aux = logits[0], logits[1]

    # need this for one_hot
    loss_labels = labels.clone()
    loss_labels[loss_labels==255] = 0
    labels_one_hot = torch.nn.functional.one_hot(
        loss_labels
    ).permute(0, 3, 1, 2).float()


    fl = focal_loss(logits, labels_one_hot)

    if aux is None:
        return fl

    separation_loss = water_obstacle_separation_loss(aux, labels_one_hot)

    separation_loss = separation_lambda * separation_loss

    loss = fl + separation_loss

    return loss

# def disp_loss(refined, err, labels, disp):
#     """ MSE loss after masking human """
#     mse_fun = nn.MSELoss()
#
#     refined = refined.reshape(refined.size(0), refined.size(2), refined.size(3))
#     refined = disp+refined # NEW!
#
#     labels[disp == 0] = 0 # filter out noise from occlusions
#
#     yhat = refined[labels == 1]
#     y = err[labels == 1]
#
#     # import numpy as np
#     # import cv2
#     # # print(disp.shape)
#     # disp_test = disp[0].detach().cpu().numpy()
#     # dp2 = np.zeros_like(disp_test).astype(np.uint8)
#     # dp2[disp_test == 0] = 255
#     # # print(disp_test.min(), disp_test.max())
#     # disp_show = ((disp_test)*255).astype(np.uint8)
#     # disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
#     # cv2.imshow('disp', disp_show)
#     # cv2.imshow('occ', dp2)
#     # cv2.waitKey(0)
#
#     return torch.mean(torch.abs(yhat-y))*190 # average disparities we are off by, most interpretable

# def disp_loss(refined, err, labels, disp):

#    # print("refined shape:", refined.shape)
#    # print("err shape:", err.shape)
#    # print("labels shape:", labels.shape)
#    # print("disp shape:", disp.shape)
#     """ MSE loss after masking human """
#     # mse_fun = nn.MSELoss()
#     loss_fun = nn.L1Loss()
#     refined = refined.mean(dim=1)

#     # refined = refined.reshape(refined.size(0), refined.size(2), refined.size(3))
#     # refined = disp+refined # NEW!

#     # labels[disp == 0] = 0 # filter out noise from occlusions

    

#    #batch_size, channels, height, width = labels.shape[0], refined.shape[1], labels.shape[1], labels.shape[2]
#    # refined = refined.view(batch_size, channels, height, width)
#      # Ensure the refined_avg tensor now matches the spatial dimensions of labels
#     print("Adjusted refined_avg shape:", refined.shape)

#     # Flatten the tensors for compatible indexing
#     refined_flat = refined.view(refined.size(0), -1)
#     err_flat = err.view(err.size(0), -1)
#     labels_flat = labels.view(labels.size(0), -1)

#     # Create a mask for indexing
#     mask = labels_flat > 0


#     yhat = refined_flat[mask]
#     y = err_flat[mask]

#     # import numpy as np
#     # import cv2
#     # # print(disp.shape)
#     # disp_test = disp[0].detach().cpu().numpy()
#     # dp2 = np.zeros_like(disp_test).astype(np.uint8)
#     # dp2[disp_test == 0] = 255
#     # # print(disp_test.min(), disp_test.max())
#     # disp_show = ((disp_test)*255).astype(np.uint8)
#     # disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
#     # cv2.imshow('disp', disp_show)
#     # cv2.imshow('occ', dp2)
#     # cv2.waitKey(0)

#     # return torch.mean(torch.abs(yhat-y))*190 # average disparities we are off by, most interpretable
#     return loss_fun(yhat, y)

def disp_loss(refined, err, labels, disp):
    loss_fun = nn.L1Loss()

    # Reshape 'refined' to [batch_size, channels, height, width]
    refined_reshaped = refined.view(refined.size(0), refined.size(1), 384, 640)

    # Create the mask with channel dimension and expand it
    mask = (labels > 0).unsqueeze(1).expand_as(refined_reshaped)

    # Apply mask and maintain dimensions
    yhat = refined_reshaped * mask

    # Sum across channel dimension and then apply the mask
    yhat_summed = yhat.sum(dim=1)[labels > 0]

    # Apply the mask to 'err' and flatten
    y = err[labels > 0]

    # Reshape yhat_summed to match the shape of y
    yhat_summed = yhat_summed.view(-1)

    return loss_fun(yhat_summed, y)


def disp_only_loss(refined, err, disp):
    """ MSE loss after masking human """
    mse_fun = nn.MSELoss()

    refined = refined.reshape(refined.size(0), refined.size(2), refined.size(3))
    refined = disp+refined # NEW!

    yhat = refined[disp > 0]
    y = err[disp > 0]

    return torch.mean(torch.abs(yhat-y))*190 # average disparities we are off by, most interpretable

def separation_disp_loss(logits, labels, err, disp, separation_lambda=0.05, ignore_label=255):

    logits, aux, refined = logits[0], logits[1], logits[-1]

    # need this for one_hot
    loss_labels = labels.clone()
    loss_labels[loss_labels==255] = 0
    labels_one_hot = torch.nn.functional.one_hot(
        loss_labels
    ).permute(0, 3, 1, 2).float()


    fl = focal_loss(logits, labels_one_hot)

    if aux is None:
        return fl

    separation_loss = water_obstacle_separation_loss(aux, labels_one_hot)

    separation_loss = separation_lambda * separation_loss

    dl = disp_loss(refined, err, labels, disp)

    loss = fl + separation_loss + dl

    return loss

def separation_ndr_loss(logits, labels, gt_disp, disp, separation_lambda=0.05, ignore_label=255):

    logits, aux, D_tilda, probs = logits[0], logits[1], logits[-2], logits[-1]

    # need this for one_hot
    loss_labels = labels.clone()
    loss_labels[loss_labels==255] = 0
    labels_one_hot = torch.nn.functional.one_hot(
        loss_labels
    ).permute(0, 3, 1, 2).float()


    fl = focal_loss(logits, labels_one_hot)

    if aux is None:
        return fl

    separation_loss = water_obstacle_separation_loss(aux, labels_one_hot)

    separation_loss = separation_lambda * separation_loss

    gt_disp = gt_disp.unsqueeze(1)
    lbl = (labels>0).reshape(labels.size(0), -1)

    # import cv2
    # import numpy as np
    # test = lbl[0].detach().cpu().numpy()
    # cv2.imshow('test', (test.reshape(384,640)*255).astype(np.uint8))
    # cv2.waitKey(0)

    # exp1
    # lbl[lbl == 2] = 2 # experiment: try weighing the object more
    dl, _, _ = DisparityLoss(D_tilda, probs, gt_disp, lbl=lbl)
    # exp2
    # refined = D_tilda.reshape(D_tilda.size(0), 384, 640)
    # gt_disp = gt_disp.squeeze()
    # dl = disp_loss(refined, gt_disp, labels, disp)
    # exp=3
    # dl, _, _ = DisparityLoss(D_tilda, probs, gt_disp, lbl=None)

    loss = fl + separation_loss + dl

    return loss


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'SeparationLoss', 'SeparationDispLoss', 'DispLoss', 'SeparationNDRLoss']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    elif loss_fn_name == 'SeparationLoss':
        return separation_loss
    elif loss_fn_name == 'SeparationDispLoss':
        return separation_disp_loss
    elif loss_fn_name == 'DispLoss':
        return disp_only_loss
    elif loss_fn_name == 'SeparationNDRLoss':
        return separation_ndr_loss

    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)
